"""作业状态：文件清单、副本目录、带头位置，以及回执的应用。

约定：
- 回执必须携带其所属作业的 job_id；作业被替换（POST /jobs 覆盖）或删除
  （DELETE /job）后，旧作业的迟到回执一律拒绝（StaleJobError），不触碰新作业。
- 只有“当前计划第一步（第一个 read 动作）”的回执才会被接受并推进进度；
  其余回执一律拒绝（MismatchError），不更新任何状态。
- 同一读带步骤只入账一次：已出现在历史中的步骤再次回报视为重复回执
  （DuplicateStepError），拒绝且状态不变。
- 无论读带成功还是失败，带头都推进到该步骤的卷内序号；
  若该步骤在另一卷上，说明装带已发生，当前卷随之切换。
- 失败的副本立即停用，不再参与后续规划。
- 核对作业、确认首步、入账与重排在同一把锁内一次生效，并在锁内生成
  不可变快照（JobSnapshot）；接口层只从快照序列化，避免并发下响应里
  的状态与计划来自不同进度。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .planner import Copy, Plan, Step, build_plan


class NoJobError(Exception):
    """尚未创建作业。"""


class NoPendingStepError(Exception):
    """当前计划没有可回报的读带步骤（已全部完成或全部无副本可读）。"""


class StaleJobError(Exception):
    """回执属于已被替换或删除的旧作业。"""

    def __init__(self, current_job_id: int | None):
        super().__init__("receipt refers to a job that has been replaced or deleted")
        # 当前作业 id；作业已删除时为 None
        self.current_job_id = current_job_id


class DuplicateStepError(Exception):
    """该读带步骤已经入账，重复回执被拒绝。"""


class MismatchError(Exception):
    """回执与当前计划第一步不符。"""

    def __init__(self, expected: Step):
        super().__init__("receipt does not match the first step of the current plan")
        self.expected = expected


@dataclass
class CopyRecord:
    file: str
    volume: int
    position: int
    enabled: bool = True


@dataclass
class ReceiptEvent:
    seq: int
    file: str
    volume: int
    position: int
    result: str  # "ok" | "failed"


@dataclass
class Job:
    id: int
    files: list[str]
    copies: list[CopyRecord]
    done: set[str] = field(default_factory=set)
    current_volume: int | None = None  # 当前已挂载的卷（初始为空机）
    head: int = 0  # 带头所在的卷内序号
    mounts_done: int = 0
    rewinds_done: int = 0
    reads_ok: int = 0
    reads_failed: int = 0
    history: list[ReceiptEvent] = field(default_factory=list)

    def enabled_copies(self) -> dict[str, list[Copy]]:
        result: dict[str, list[Copy]] = {}
        for c in self.copies:
            if c.enabled:
                result.setdefault(c.file, []).append(Copy(c.file, c.volume, c.position))
        return result

    def pending(self) -> list[str]:
        return [f for f in self.files if f not in self.done]

    def unretrievable(self) -> list[str]:
        enabled = self.enabled_copies()
        return sorted(f for f in self.pending() if not enabled.get(f))

    def plan(self) -> Plan:
        return build_plan(self.pending(), self.enabled_copies(), self.current_volume, self.head)

    def processed_steps(self) -> set[tuple[str, int, int]]:
        """已经入账的读带步骤（file, volume, position）。"""
        return {(e.file, e.volume, e.position) for e in self.history}


@dataclass(frozen=True)
class JobSnapshot:
    """锁内生成的不可变进度快照：状态与计划取自同一时刻，互不串扰。"""

    job_id: int
    current_volume: int | None
    head: int
    pending: list[str]
    done: list[str]
    unretrievable: list[str]
    disabled_copies: list[dict]
    executed: dict[str, int]
    history: list[dict]
    plan: Plan


def _snapshot(job: Job) -> JobSnapshot:
    """必须在持有 JobStore 锁时调用：把可变作业凝固成不可变快照。"""
    return JobSnapshot(
        job_id=job.id,
        current_volume=job.current_volume,
        head=job.head,
        pending=sorted(job.pending()),
        done=sorted(job.done),
        unretrievable=job.unretrievable(),
        disabled_copies=[
            {"file": c.file, "volume": c.volume, "position": c.position}
            for c in job.copies
            if not c.enabled
        ],
        executed={
            "mounts": job.mounts_done,
            "rewinds": job.rewinds_done,
            "reads_ok": job.reads_ok,
            "reads_failed": job.reads_failed,
        },
        history=[vars(e) for e in job.history],
        plan=job.plan(),
    )


class JobStore:
    """内存作业仓库：同一时刻只处理一个取回作业。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._next_id = 1

    def create(self, files: list[str], copies: list[CopyRecord]) -> JobSnapshot:
        """创建（覆盖）作业，并在锁内返回初始进度快照。"""
        with self._lock:
            job = Job(id=self._next_id, files=files, copies=copies)
            self._next_id += 1
            self._job = job
            return _snapshot(job)

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            if self._job is None:
                raise NoJobError
            return _snapshot(self._job)

    def reset(self) -> bool:
        with self._lock:
            existed = self._job is not None
            self._job = None
            return existed

    def apply_receipt(
        self,
        job_id: int,
        file: str,
        volume: int,
        position: int,
        result: str,
    ) -> JobSnapshot:
        """应用一张回执，全流程在同一把锁内完成，返回入账后的进度快照。

        任何拒绝条件命中时抛异常且作业状态原样不变。
        """
        with self._lock:
            # 1) 核对作业：作业已删除 → NoJobError；已被新作业替换 → StaleJobError
            if self._job is None:
                raise NoJobError
            job = self._job
            if job_id != job.id:
                raise StaleJobError(job.id)

            # 2) 该步骤已处理过 → 重复回执，拒绝
            if (file, volume, position) in job.processed_steps():
                raise DuplicateStepError

            # 3) 确认当前计划第一步
            before = job.plan()
            step = before.next_read
            if step is None:
                raise NoPendingStepError
            if (step.file, step.volume, step.position) != (file, volume, position):
                raise MismatchError(step)

            # 4) 入账：该读带步骤之前的装带/回卷动作随回执一并确认
            for s in before.steps:
                if s.action == "read":
                    break
                if s.action == "mount":
                    job.mounts_done += 1
                elif s.action == "rewind":
                    job.rewinds_done += 1

            # 成功或失败都推进带头；跨卷步骤意味着装带已发生
            job.current_volume = volume
            job.head = position

            if result == "ok":
                job.done.add(file)
                job.reads_ok += 1
            else:
                for c in job.copies:
                    if c.file == file and c.volume == volume and c.position == position:
                        c.enabled = False  # 停用出错的副本
                        break
                job.reads_failed += 1

            job.history.append(
                ReceiptEvent(
                    seq=len(job.history) + 1,
                    file=file,
                    volume=volume,
                    position=position,
                    result=result,
                )
            )

            # 5) 锁内重排并凝固快照，保证响应中的状态与计划同属本次入账后的进度
            return _snapshot(job)


store = JobStore()
