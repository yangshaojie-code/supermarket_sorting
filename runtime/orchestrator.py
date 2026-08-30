"""Pick-one then deliver-one mission loop. No scene reset, no motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from runtime.scene_zones import TIME_LIMIT_S
from runtime.task_protocol import Target, TaskList, parse_task_payload


class MissionState(str, Enum):
    WAITING_TASK = "WAITING_TASK"
    PICKING = "PICKING"
    DELIVERING = "DELIVERING"
    DONE = "DONE"
    TIMEOUT = "TIMEOUT"


class MissionStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class CycleRecord:
    target_id: str
    kind: str
    phase: str
    success: bool
    reason: Optional[str] = None


@dataclass
class MissionOrchestrator:
    """Enforce 取一件→送一件→再取下一件 without resetting physics."""

    time_limit_seconds: float = TIME_LIMIT_S
    chooser: Optional[Callable[[List[Target]], Target]] = None
    task: Optional[TaskList] = None
    state: MissionState = MissionState.WAITING_TASK
    pending: List[Target] = field(default_factory=list)
    current: Optional[Target] = None
    holding: bool = False
    elapsed_seconds: float = 0.0
    records: List[CycleRecord] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def load_task(self, payload: Any) -> TaskList:
        if self.state in (MissionState.PICKING, MissionState.DELIVERING):
            raise MissionStateError("cannot replace the task list during an open pick-deliver cycle")
        task = payload if isinstance(payload, TaskList) else parse_task_payload(payload)
        self.task = task
        self.state = MissionState.WAITING_TASK
        self.pending = list(task.targets)
        self.current = None
        self.holding = False
        self.elapsed_seconds = 0.0
        self.records.clear()
        self.context = {"run_prefix": task.run_prefix}
        return task

    def remaining(self) -> List[Target]:
        return list(self.pending)

    def tick(self, elapsed_seconds: float) -> MissionState:
        self.elapsed_seconds = float(elapsed_seconds)
        if self.state not in (MissionState.DONE, MissionState.TIMEOUT):
            if self.elapsed_seconds >= self.time_limit_seconds:
                self.state = MissionState.TIMEOUT
        return self.state

    def start_pick(self, target: Optional[Target] = None) -> Target:
        if self.task is None:
            raise MissionStateError("no /supermarket_sorting/task loaded")
        if self.state == MissionState.TIMEOUT:
            raise MissionStateError("time limit reached")
        if self.holding or self.state in (MissionState.PICKING, MissionState.DELIVERING):
            raise MissionStateError("must finish the open pick-deliver cycle before picking another")
        if self.state == MissionState.DONE:
            raise MissionStateError("all targets already finished")
        if not self.pending:
            self.state = MissionState.DONE
            raise MissionStateError("no remaining targets")
        chosen = target if target is not None else self._choose()
        if chosen.id not in {item.id for item in self.pending}:
            raise MissionStateError(f"target {chosen.id} is not remaining")
        self.current = chosen
        self.holding = False
        self.state = MissionState.PICKING
        self.context["current_id"] = chosen.id
        self.context["current_kind"] = chosen.kind
        return chosen

    def complete_pick(self, success: bool, reason: Optional[str] = None) -> None:
        if self.state != MissionState.PICKING or self.current is None:
            raise MissionStateError("complete_pick requires an active pick")
        self.records.append(CycleRecord(self.current.id, self.current.kind, "pick", success, reason))
        if success:
            self.holding = True
            self.state = MissionState.DELIVERING
        else:
            # Stay on the same remaining target. Physics is not reset.
            self.holding = False
            self.state = MissionState.WAITING_TASK

    def complete_deliver(self, success: bool, reason: Optional[str] = None) -> Optional[Target]:
        if self.state != MissionState.DELIVERING or self.current is None or not self.holding:
            raise MissionStateError("complete_deliver requires a held item")
        self.records.append(CycleRecord(self.current.id, self.current.kind, "deliver", success, reason))
        finished = self.current
        if success:
            self.pending = [item for item in self.pending if item.id != finished.id]
            self.holding = False
            self.current = None
            self.state = MissionState.DONE if not self.pending else MissionState.WAITING_TASK
            return finished
        # Drop / misplace: cycle scores 0, item may still be somewhere in the scene.
        self.holding = False
        self.state = MissionState.WAITING_TASK
        return None

    def planned_cycles(self) -> List[dict]:
        return [
            {
                "index": item.index,
                "id": item.id,
                "kind": item.kind,
                "pick_then_deliver": True,
            }
            for item in self.pending
        ]

    def _choose(self) -> Target:
        if self.chooser is not None:
            return self.chooser(list(self.pending))
        return self.pending[0]
