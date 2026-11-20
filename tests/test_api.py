"""API 集成测试：完整取回流程，含坏副本停用、重排、迟到/重复回执与并发快照。"""
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

DEMO_JOB = {
    "files": ["A", "B", "C", "D", "E", "F"],
    "copies": [
        {"file": "A", "volume": 1, "position": 2},
        {"file": "A", "volume": 2, "position": 5},
        {"file": "B", "volume": 1, "position": 6},
        {"file": "C", "volume": 2, "position": 1},
        {"file": "C", "volume": 3, "position": 4},
        {"file": "D", "volume": 3, "position": 1},
        {"file": "E", "volume": 3, "position": 6},
        {"file": "F", "volume": 1, "position": 9},
        {"file": "F", "volume": 3, "position": 8},
    ],
}


@pytest.fixture(autouse=True)
def reset_job():
    client.delete("/job")
    yield
    client.delete("/job")


def create_demo_job():
    resp = client.post("/jobs", json=DEMO_JOB)
    assert resp.status_code == 201, resp.text
    return resp.json()


def report(file, volume, position, result, job_id=None):
    if job_id is None:
        job_id = client.get("/state").json()["job_id"]
    return client.post(
        "/receipts",
        json={
            "job_id": job_id,
            "file": file,
            "volume": volume,
            "position": position,
            "result": result,
        },
    )


def test_initial_plan_is_optimal():
    body = create_demo_job()
    plan = body["plan"]
    assert plan["job_id"] == body["state"]["job_id"] == 1
    assert plan["cost"] == 2
    assert plan["mounts"] == 2 and plan["rewinds"] == 0
    assert plan["next_step"] == {
        "seq": 2, "action": "read", "volume": 1, "file": "A", "position": 2,
    }
    state = body["state"]
    assert state["current_volume"] is None and state["head"] == 0
    assert state["pending"] == list("ABCDEF")
    # 创建响应里的计划以本次创建刚完成时的进度为准（无已取、无停用、计数为 0）
    assert state["done"] == [] and state["disabled_copies"] == []
    assert state["executed"] == {"mounts": 0, "rewinds": 0, "reads_ok": 0, "reads_failed": 0}
    # 再次创建会分配新的 job_id（旧作业被覆盖）
    second = create_demo_job()
    assert second["state"]["job_id"] == 2
    assert second["plan"]["job_id"] == 2


def test_receipt_must_match_first_step():
    create_demo_job()
    # 不按计划第一步回报（第一步是 A@1:2，这里报 B）→ 409，进度不动
    resp = report("B", 1, 6, "ok")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["expected"] == {"file": "A", "volume": 1, "position": 2}
    state = client.get("/state").json()
    assert state["pending"] == list("ABCDEF")
    assert state["current_volume"] is None


def test_failed_copy_disabled_and_replanned():
    create_demo_job()
    # 第一步 A@卷1:2 读失败：带头推进到卷1@2，副本停用，A 还有卷2 的副本
    resp = report("A", 1, 2, "failed")
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"]["copy_disabled"] is True
    state = body["state"]
    assert state["current_volume"] == 1 and state["head"] == 2
    assert state["disabled_copies"] == [{"file": "A", "volume": 1, "position": 2}]
    assert state["pending"] == list("ABCDEF")  # A 未取到，仍在待取
    assert state["unretrievable"] == []
    plan = body["plan"]
    # 重排：A 只能用卷2 的副本；卷1 是当前卷免装带
    assert plan["cost"] == 2 and plan["mounts"] == 2 and plan["rewinds"] == 0
    assert plan["next_step"]["file"] == "B"


def test_full_retrieval_flow_with_two_failures():
    create_demo_job()

    # 1) A@1:2 失败 → 停用，重排
    plan = report("A", 1, 2, "failed").json()["plan"]
    # 2) B@1:6 成功（当前卷继续读，不装带）
    body = report("B", 1, 6, "ok").json()
    assert body["state"]["head"] == 6
    assert body["state"]["done"] == ["B"]
    # 3) 下一步在卷2：C@2:1 成功（装带计入累计：卷1、卷2 各一次）
    body = report("C", 2, 1, "ok").json()
    assert body["state"]["current_volume"] == 2 and body["state"]["head"] == 1
    assert body["state"]["executed"]["mounts"] == 2
    # 4) A@2:5 失败 → A 副本耗尽，进入无副本可读清单
    body = report("A", 2, 5, "failed").json()
    assert body["state"]["unretrievable"] == ["A"]
    assert body["state"]["pending"] == ["A", "D", "E", "F"]
    plan = body["plan"]
    assert plan["unretrievable"] == ["A"]
    assert plan["mounts"] == 1 and plan["rewinds"] == 0  # 只剩卷3 要装
    # 5) 依次取回 D、E、F（卷3 升序）
    for f, p in [("D", 1), ("E", 6), ("F", 8)]:
        body = report(f, 3, p, "ok").json()
    state = body["state"]
    assert state["done"] == ["B", "C", "D", "E", "F"]
    assert state["unretrievable"] == ["A"]
    assert state["executed"] == {"mounts": 3, "rewinds": 0, "reads_ok": 5, "reads_failed": 2}
    # 计划清空：没有可回报的步骤
    assert body["plan"]["steps"] == []
    assert body["plan"]["next_step"] is None
    # 再回报 → 409
    assert report("F", 3, 8, "ok").status_code == 409


def test_live_flow_never_needs_rewind():
    """不变式：当前卷上不存在“带头之下的可用副本”。

    计划按序号升序调度，且失败副本立即停用，所以带头不可能越过
    仍处于可用状态的副本 —— 实际流程中重排永远不需要回卷。
    （回卷分支只在理论平局裁决中出现，见 tests/test_planner.py。）
    """
    create_demo_job()
    plan = report("A", 1, 2, "failed").json()["plan"]
    assert plan["rewinds"] == 0
    plan = report("B", 1, 6, "ok").json()["plan"]
    assert plan["rewinds"] == 0
    plan = report("C", 2, 1, "ok").json()["plan"]
    assert plan["rewinds"] == 0
    plan = report("A", 2, 5, "failed").json()["plan"]
    assert plan["rewinds"] == 0


def test_validation_errors():
    # 超过 12 卷
    resp = client.post("/jobs", json={
        "files": ["A"],
        "copies": [{"file": "A", "volume": v, "position": 1} for v in range(1, 14)],
    })
    assert resp.status_code == 422
    # 未知文件
    resp = client.post("/jobs", json={
        "files": ["A"],
        "copies": [{"file": "B", "volume": 1, "position": 1}],
    })
    assert resp.status_code == 422
    # 重复副本
    resp = client.post("/jobs", json={
        "files": ["A"],
        "copies": [
            {"file": "A", "volume": 1, "position": 1},
            {"file": "A", "volume": 1, "position": 1},
        ],
    })
    assert resp.status_code == 422
    # 非法序号
    resp = client.post("/jobs", json={
        "files": ["A"],
        "copies": [{"file": "A", "volume": 1, "position": 0}],
    })
    assert resp.status_code == 422


def test_endpoints_require_job():
    assert client.get("/state").status_code == 404
    assert client.get("/plan").status_code == 404
    assert report("A", 1, 1, "ok", job_id=1).status_code == 404
    assert client.delete("/job").status_code == 204


def test_late_receipt_after_job_replacement_is_rejected():
    """旧作业被覆盖后，即便首步 (file,volume,position) 完全相同，
    旧 job_id 的迟到回执也必须拒绝，新作业的任何进度都不被推进。"""
    old = create_demo_job()
    old_id = old["state"]["job_id"]
    assert old["plan"]["next_step"] == {
        "seq": 2, "action": "read", "volume": 1, "file": "A", "position": 2,
    }

    # 覆盖为同清单的新作业：首步仍是 A@1:2，但 job_id 已变
    new = create_demo_job()
    new_id = new["state"]["job_id"]
    assert new_id != old_id

    # 旧作业的迟到回执 → 409，新作业状态原样不动
    resp = report("A", 1, 2, "ok", job_id=old_id)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["current_job_id"] == new_id
    state = client.get("/state").json()
    assert state["job_id"] == new_id
    assert state["pending"] == list("ABCDEF")
    assert state["done"] == []
    assert state["current_volume"] is None and state["head"] == 0
    assert state["executed"] == {"mounts": 0, "rewinds": 0, "reads_ok": 0, "reads_failed": 0}
    assert state["history"] == []
    # 新作业自己的回执正常生效，响应反映的是新作业的进度
    ok = report("A", 1, 2, "ok", job_id=new_id)
    assert ok.status_code == 200
    assert ok.json()["state"]["done"] == ["A"]
    assert ok.json()["plan"]["job_id"] == new_id


def test_late_receipt_after_job_delete_is_rejected():
    old = create_demo_job()
    old_id = old["state"]["job_id"]
    client.delete("/job")
    # 已删除作业的迟到回执 → 404；重新创建后旧 id 同样不可用
    assert report("A", 1, 2, "ok", job_id=old_id).status_code == 404
    create_demo_job()
    assert report("A", 1, 2, "ok", job_id=old_id).status_code == 409
    # 新作业进度未被迟到回执触碰
    assert client.get("/state").json()["done"] == []


def test_duplicate_receipt_for_same_step_is_rejected():
    create_demo_job()
    job_id = client.get("/state").json()["job_id"]
    first = report("A", 1, 2, "ok", job_id=job_id)
    assert first.status_code == 200
    snap1 = first.json()
    assert snap1["state"]["done"] == ["A"]
    assert snap1["state"]["executed"]["reads_ok"] == 1

    # 同一步骤再次回报（失败也算）→ 409，带头、副本、计数、历史全部保持原样
    dup = report("A", 1, 2, "failed", job_id=job_id)
    assert dup.status_code == 409
    assert "已经入账" in dup.json()["detail"]["error"]
    state = client.get("/state").json()
    assert state["current_volume"] == 1 and state["head"] == 2
    assert state["done"] == ["A"]
    assert state["disabled_copies"] == []  # 失败回执未入账，副本未被停用
    assert state["executed"] == {"mounts": 1, "rewinds": 0, "reads_ok": 1, "reads_failed": 0}
    assert len(state["history"]) == 1


def test_response_snapshot_is_consistent_after_commit():
    """入账响应里的状态与计划必须同属本次回执刚完成时的进度（不可变快照）。"""
    create_demo_job()
    job_id = client.get("/state").json()["job_id"]
    body = report("A", 1, 2, "failed", job_id=job_id).json()
    state, plan = body["state"], body["plan"]
    # 状态与计划携带同一 job_id
    assert state["job_id"] == plan["job_id"] == job_id
    # 计划下一步与状态中的待取/带头自洽：A 仍待取，B 是新首步，当前卷 1 带头 2
    assert state["head"] == 2 and state["current_volume"] == 1
    assert plan["next_step"]["file"] == "B"
    assert "A" in state["pending"]
    # 后续继续推进，旧响应快照内容不被改写（快照独立于可变作业）
    report("B", 1, 6, "ok", job_id=job_id)
    assert body["state"]["head"] == 2
    assert body["state"]["done"] == []
    assert body["plan"]["next_step"]["file"] == "B"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_concurrent_receipts_each_response_reflects_its_own_commit():
    """两步回报并发交错：每张成功响应只能带自己的重排计划与自己的进度，
    后一步的进度绝不串进先入账的响应；被拒的回执不产生任何副作用。

    锁把两张回执串成两种合法顺序：
    - A 先 B 后：两张都 200（B 入账时已成为计划首步）；
    - B 先到：B 与首步不符 → 409，A 200。
    """
    create_demo_job()
    job_id = client.get("/state").json()["job_id"]

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                httpx.get(base + "/docs", timeout=1)
                break
            except httpx.TransportError:
                time.sleep(0.05)
        else:  # pragma: no cover - 服务起不来时才走这里
            raise AssertionError("测试用 uvicorn 服务未在限时内就绪")

        # 同一时刻发出两张针对连续两步的回执：首步 A@1:2、次步 B@1:6
        payload_a = {"job_id": job_id, "file": "A", "volume": 1, "position": 2, "result": "ok"}
        payload_b = {"job_id": job_id, "file": "B", "volume": 1, "position": 6, "result": "ok"}
        barrier = threading.Barrier(2)
        results: dict[str, httpx.Response] = {}

        def send(name: str, payload: dict) -> None:
            barrier.wait()
            with httpx.Client(base_url=base, timeout=10) as http:
                results[name] = http.post("/receipts", json=payload)

        t1 = threading.Thread(target=send, args=("a", payload_a))
        t2 = threading.Thread(target=send, args=("b", payload_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        ra, rb = results["a"], results["b"]
        statuses = sorted((ra.status_code, rb.status_code))
        assert statuses in ([200, 200], [200, 409])

        # 校验每张成功响应：状态与计划严格自洽，且只反映本回执入账那一刻的进度
        for resp in (ra, rb):
            if resp.status_code != 200:
                continue
            body = resp.json()
            st, pl = body["state"], body["plan"]
            assert st["job_id"] == pl["job_id"] == job_id
            # 状态里的已取集合必须与累计成功次数、历史条数完全一致
            assert st["executed"]["reads_ok"] == len(st["done"]) == len(st["history"])
            # 计划首步指向的文件必须仍在待取清单中（状态与计划不得各说各话）
            nxt = pl["next_step"]
            if nxt is not None:
                assert nxt["file"] in st["pending"]
            # 响应快照必须与按其历史重放得到的进度一致（快照不被后续提交改写）
            assert st["head"] == st["history"][-1]["position"]
            assert st["current_volume"] == st["history"][-1]["volume"]

        body_a = ra.json()
        if ra.status_code == 200:
            # A 的响应：无论它先还是后入账，done 里有 A；若 B 已先入账则会是 409，
            # 因此 200 时 A 必然是第一笔，done 恰为 ["A"]，计划首步是 B
            assert body_a["state"]["done"] == ["A"]
            assert body_a["state"]["head"] == 2
            assert "B" not in body_a["state"]["done"]
            assert body_a["plan"]["next_step"] == {
                "seq": 1, "action": "read", "volume": 1, "file": "B", "position": 6,
            }
        else:
            # B 先到被拒：期望首步仍是 A，且无任何副作用
            assert rb.status_code == 200
            assert ra.json()["detail"]["expected"]["file"] == "A"

        # 最终状态：被拒的一步未生效，成功的都已生效；补报后流程继续
        final = client.get("/state").json()
        if rb.status_code == 409:
            assert final["done"] == ["A"] and final["head"] == 2
            follow = report("B", 1, 6, "ok", job_id=job_id)
            assert follow.status_code == 200
            assert follow.json()["state"]["done"] == ["A", "B"]
        else:
            assert final["done"] == ["A", "B"] and final["head"] == 6
            assert final["executed"]["reads_ok"] == 2
    finally:
        server.should_exit = True
        thread.join(timeout=10)
