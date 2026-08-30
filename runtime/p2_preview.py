#!/usr/bin/env python3
"""P2 preview: parse /supermarket_sorting/task and walk pick-deliver cycles.

Does not publish motion. Use this to verify the task JSON and the
取一件→送一件 loop before navigation or grasping exist.
"""

from __future__ import annotations

import argparse
import json
import time

from runtime.orchestrator import MissionOrchestrator
from runtime.scene_zones import DELIVERY_APPROACH_XY, DELIVERY_TARGET_XYZ, TIME_LIMIT_S
from runtime.task_protocol import parse_task_payload, unknown_kinds


def wait_for_task(timeout_sec: float = 10.0) -> str:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    from runtime.ros_contract import TASK_TOPIC

    rclpy.init()
    node = rclpy.create_node("supermarket_p2_preview")
    box = {"data": None}
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def _cb(message):
        box["data"] = message.data

    node.create_subscription(String, TASK_TOPIC, _cb, qos)
    deadline = time.monotonic() + timeout_sec
    try:
        while box["data"] is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for /supermarket_sorting/task")
            rclpy.spin_once(node, timeout_sec=0.1)
        return box["data"]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def preview_dict(raw: str, dry_run: bool) -> dict:
    task = parse_task_payload(raw)
    orch = MissionOrchestrator()
    orch.load_task(task)
    dry_records = []
    if dry_run:
        while orch.remaining():
            target = orch.start_pick()
            orch.complete_pick(True, "dry-run")
            orch.complete_deliver(True, "dry-run")
            dry_records.append({"id": target.id, "kind": target.kind, "result": "picked_then_delivered"})
    return {
        "run_prefix": task.run_prefix,
        "count": task.count,
        "unknown_kinds": list(unknown_kinds(task)),
        "time_limit_s": TIME_LIMIT_S,
        "delivery_target_xyz": list(DELIVERY_TARGET_XYZ),
        "delivery_approach_xy": list(DELIVERY_APPROACH_XY),
        "planned_cycles": orch.planned_cycles() if not dry_run else dry_records,
        "state": orch.state.value,
        "rule": "pick one then deliver one; never pick the next item while holding",
        "raw": raw,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preview supermarket pick-deliver cycles")
    parser.add_argument("--dry-run", action="store_true", help="walk all cycles as successful without motion")
    parser.add_argument("--payload", help="task JSON string; default waits on the ROS topic")
    args = parser.parse_args(argv)
    raw = args.payload if args.payload is not None else wait_for_task()
    print(json.dumps(preview_dict(raw, args.dry_run), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
