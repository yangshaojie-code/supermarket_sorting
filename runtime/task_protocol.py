"""Parse /supermarket_sorting/task without using unpublished location fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

KNOWN_KINDS = (
    "sanmingzhi",
    "heweidao",
    "shupian",
    "zhijin",
    "maidong",
    "kouxiangtang",
    "pingguo",
    "chengzi",
    "kele",
)


class TaskProtocolError(ValueError):
    """Raised when the Server task payload cannot be used."""


@dataclass(frozen=True)
class Target:
    id: str
    kind: str
    index: int


@dataclass(frozen=True)
class TaskList:
    schema_version: int
    run_prefix: str
    targets: Tuple[Target, ...]

    @property
    def count(self) -> int:
        return len(self.targets)

    def kinds(self) -> Tuple[str, ...]:
        return tuple(item.kind for item in self.targets)


def _as_mapping(payload: Any) -> dict:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TaskProtocolError("task payload must be a JSON object")
    return payload


def parse_task_payload(payload: Any) -> TaskList:
    data = _as_mapping(payload)
    try:
        schema_version = int(data["schema_version"])
        run_prefix = str(data["run_prefix"])
        raw_targets = data["targets"]
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskProtocolError("task JSON missing schema_version/run_prefix/targets") from exc
    if schema_version != 1:
        raise TaskProtocolError(f"unsupported schema_version: {schema_version}")
    if not run_prefix:
        raise TaskProtocolError("run_prefix must be a non-empty string")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise TaskProtocolError("targets must be a non-empty list")

    declared = data.get("count")
    if declared is not None and int(declared) != len(raw_targets):
        raise TaskProtocolError(f"count={declared} does not match targets length {len(raw_targets)}")

    targets = []
    seen_ids = set()
    for index, item in enumerate(raw_targets):
        if not isinstance(item, dict):
            raise TaskProtocolError(f"target {index} must be an object")
        target_id = str(item.get("id") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not target_id or not kind:
            raise TaskProtocolError(f"target {index} needs id and kind")
        if target_id in seen_ids:
            raise TaskProtocolError(f"duplicate target id: {target_id}")
        # Location fields from the old contest brief are not published and must
        # not be treated as ground truth even if a debug Server adds them.
        seen_ids.add(target_id)
        targets.append(Target(id=target_id, kind=kind, index=index))
    return TaskList(schema_version=schema_version, run_prefix=run_prefix, targets=tuple(targets))


def unknown_kinds(task: TaskList, known: Iterable[str] = KNOWN_KINDS) -> Tuple[str, ...]:
    allowed = set(known)
    return tuple(sorted({item.kind for item in task.targets if item.kind not in allowed}))
