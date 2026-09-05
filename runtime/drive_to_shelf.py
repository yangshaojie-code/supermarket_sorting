#!/usr/bin/env python3
"""Drive MMK2 from the delivery spawn to the shelf aisle, then stop.

Thin wrapper around P4 corridor nav. Publishes only through RosRobotController.
"""

from __future__ import annotations

import argparse
import json

from runtime.p4_nav import DEFAULT_TIMEOUT_S, SPEED_SCALE, drive_to_shelf
from runtime.scene_zones import SHELF_APPROACH_XY


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Drive from delivery spawn to the shelf aisle")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--final-x", type=float, default=SHELF_APPROACH_XY[0])
    parser.add_argument("--final-y", type=float, default=SHELF_APPROACH_XY[1])
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=SPEED_SCALE,
        help="multiply the original 0.30/0.55 aisle speeds (default 8)",
    )
    args = parser.parse_args(argv)

    import rclpy

    rclpy.init()
    try:
        report = drive_to_shelf(
            rclpy,
            timeout_sec=args.timeout,
            final_xy=(args.final_x, args.final_y),
            speed_scale=args.speed_scale,
        )
        print(json.dumps(report, indent=2))
        if report["arrived"] and report["in_picking_zone"]:
            return 0
        return 1
    except TimeoutError as exc:
        print(json.dumps({"arrived": False, "in_picking_zone": False, "error": str(exc)}, indent=2))
        return 1
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
