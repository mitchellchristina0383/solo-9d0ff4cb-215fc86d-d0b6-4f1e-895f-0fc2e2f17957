"""FastAPI 入口：模拟磁带机，支撑值班员逐步取回文件。"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import RedirectResponse

from .planner import MAX_VOLUMES, Plan, Step
from .schemas import JobIn, JobOut, PlanOut, ReceiptIn, ReceiptOut, StateOut, StepOut
from .state import (
    CopyRecord,
    DuplicateStepError,
    JobSnapshot,
    MismatchError,
    NoJobError,
    NoPendingStepError,
    StaleJobError,
    store,
)

app = FastAPI(
    title="磁带机取回模拟器",
    description=(
        "接收待取文件清单与副本目录（至多 12 卷），给出装带 + 回卷总次数最少的读取顺序。"
        "值班员逐步回报读带结果：失败同样推进带头并停用出错副本；"
        "每步按带头新位置重排剩余文件，并列出已无副本可读的文件。"
        "回执须携带作业 id 且匹配当前计划第一步，迟到/重复回执一律拒绝。"
    ),
    version="1.1.0",
)


# ---------- 序列化（只从锁内生成的不可变快照取值，杜绝并发串扰） ----------

def _step_out(step: Step) -> StepOut:
    return StepOut(
        seq=step.seq,
        action=step.action,
        volume=step.volume,
        file=step.file,
        position=step.position,
    )


def _plan_out(job_id: int, plan: Plan) -> PlanOut:
    nxt = plan.next_read
    return PlanOut(
        job_id=job_id,
        volumes_used=plan.volumes_used,
        mounts=plan.mounts,
        rewinds=plan.rewinds,
        cost=plan.cost,
        next_step=_step_out(nxt) if nxt else None,
        steps=[_step_out(s) for s in plan.steps],
        unretrievable=plan.unretrievable,
    )


def _state_out(snap: JobSnapshot) -> StateOut:
    return StateOut(
        job_id=snap.job_id,
        current_volume=snap.current_volume,
        head=snap.head,
        pending=snap.pending,
        done=snap.done,
        unretrievable=snap.unretrievable,
        disabled_copies=snap.disabled_copies,
        executed=snap.executed,
        history=snap.history,
    )


# ---------- 接口 ----------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.post(
    "/jobs",
    status_code=status.HTTP_201_CREATED,
    response_model=JobOut,
    summary="创建取回作业（覆盖旧作业），返回初始状态与最优计划",
)
def create_job(body: JobIn) -> JobOut:
    files = [f.strip() for f in body.files]
    if any(not f for f in files):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "文件名不能为空")
    if len(set(files)) != len(files):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "文件清单存在重名")

    known = set(files)
    seen: set[tuple[str, int, int]] = set()
    copies: list[CopyRecord] = []
    for c in body.copies:
        if c.file not in known:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"副本指向未知文件: {c.file!r}",
            )
        key = (c.file, c.volume, c.position)
        if key in seen:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"重复副本: file={c.file} volume={c.volume} position={c.position}",
            )
        seen.add(key)
        copies.append(CopyRecord(file=c.file, volume=c.volume, position=c.position))

    volumes = {c.volume for c in copies}
    if len(volumes) > MAX_VOLUMES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"副本目录涉及 {len(volumes)} 卷，超过上限 {MAX_VOLUMES} 卷",
        )

    # 快照在锁内生成：响应中的状态与计划同为本次创建刚完成时的进度
    snap = store.create(files, copies)
    return JobOut(state=_state_out(snap), plan=_plan_out(snap.job_id, snap.plan))


@app.get("/state", response_model=StateOut, summary="查看当前状态（带头位置、待取/已取/无副本文件等）")
def get_state() -> StateOut:
    try:
        return _state_out(store.snapshot())
    except NoJobError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "尚未创建取回作业，请先 POST /jobs")


@app.get("/plan", response_model=PlanOut, summary="按当前带头位置重算最优读取计划")
def get_plan() -> PlanOut:
    try:
        snap = store.snapshot()
    except NoJobError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "尚未创建取回作业，请先 POST /jobs")
    return _plan_out(snap.job_id, snap.plan)


@app.post(
    "/receipts",
    response_model=ReceiptOut,
    summary="回报当前计划第一步的读带结果，并按带头新位置重排剩余文件",
    responses={
        404: {"description": "尚未创建作业（或作业已删除）"},
        409: {"description": "回执属于已替换作业/步骤重复/无待回报步骤/与当前计划第一步不符"},
    },
)
def post_receipt(body: ReceiptIn) -> ReceiptOut:
    try:
        # 核对作业、确认首步、入账、重排与快照在同一把锁内一次生效
        snap = store.apply_receipt(
            body.job_id, body.file, body.volume, body.position, body.result
        )
    except NoJobError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "尚未创建取回作业，请先 POST /jobs")
    except StaleJobError as exc:
        if exc.current_job_id is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                {"error": "回执属于已删除的旧作业，进度未更新"},
            )
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {
                "error": "回执所属作业已被新作业替换，迟到回执被拒绝，进度未更新",
                "current_job_id": exc.current_job_id,
            },
        )
    except DuplicateStepError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"error": "该读带步骤已经入账，重复回执被拒绝，进度未更新"},
        )
    except NoPendingStepError:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "当前计划没有待回报的读带步骤（已全部完成或无副本可读）"
        )
    except MismatchError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {
                "error": "回执与当前计划第一步不符，进度未更新",
                "expected": {
                    "file": exc.expected.file,
                    "volume": exc.expected.volume,
                    "position": exc.expected.position,
                },
            },
        )
    return ReceiptOut(
        applied={
            "job_id": snap.job_id,
            "file": body.file,
            "volume": body.volume,
            "position": body.position,
            "result": body.result,
            "copy_disabled": body.result == "failed",
        },
        state=_state_out(snap),
        plan=_plan_out(snap.job_id, snap.plan),
    )


@app.delete(
    "/job",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="作废当前作业（复位磁带机）",
)
def delete_job() -> Response:
    store.reset()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
