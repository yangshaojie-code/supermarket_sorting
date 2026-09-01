#!/usr/bin/env python3
"""Drive MMK2 from the delivery spawn to the shelf aisle, then stop.

Intended as a one-shot setup before ``scripts/run_p3_preview.sh``. Does not
grasp, does not rewrite ``client_task_1.py``, and publishes only through
``RosRobotController``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from runtime.head_camera_kinematics import AISLE_SCAN_HEAD, AISLE_SCAN_SPINE
from runtime.ros_contract import JOINT_STATES_TOPIC, ODOM_TOPIC, SCAN_TOPIC
from runtime.ros_robot_control import RosRobotController
from runtime.ros_sensor_utils import SensorCache
from runtime.scene_zones import (
    SHELF_APPROACH_XY,
    SHELF_FACE_YAW,
    in_picking_zone,
)
from runtime.waypoint_nav import (
    WaypointFollower,
    build_shelf_route,
    min_forward_range_from_scan,
    pose_from_odom,
)

FORWARD_STOP_M = 0.40
RATE_HZ = 20.0
BASE_LIN = 0.30
BASE_ANG = 0.55
SPEED_SCALE = 8.0
HEAD_SETTLE_S = 1.5


def _wait_odom(rclpy_mod, node, sensors: SensorCache, timeout_sec: float) -> None:
    deadline = time.monotonic() + float(timeout_sec)
    while sensors.odom is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for odom")
        rclpy_mod.spin_once(node, timeout_sec=0.05)


def _aim_head(controller: RosRobotController) -> None:
    controller.command_spine(AISLE_SCAN_SPINE)
    controller.command_head(AISLE_SCAN_HEAD)


def _head_joints(sensors: SensorCache) -> dict:
    values = dict(zip(sensors.joint_names, [float(v) for v in sensors.joint_positions]))
    return {
        "slide": values.get("slide_joint"),
        "head_yaw": values.get("head_yaw_joint"),
        "head_pitch": values.get("head_pitch_joint"),
    }


def _joint_cb(sensors: SensorCache):
    def wrapped(message):
        try:
            sensors.update_joint_state(message)
        except Exception:
            pass
    return wrapped


def drive_to_shelf(
    rclpy_mod,
    timeout_sec: float = 90.0,
    final_xy=SHELF_APPROACH_XY,
    final_yaw: float = SHELF_FACE_YAW,
    speed_scale: float = SPEED_SCALE,
) -> dict:
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState, LaserScan
    from std_msgs.msg import Float64MultiArray

    scale = max(0.1, float(speed_scale))
    max_lin = BASE_LIN * scale
    max_ang = BASE_ANG * scale
    node = rclpy_mod.create_node("supermarket_drive_to_shelf")
    sensors = SensorCache()
    controller = RosRobotController(
        node,
        Twist,
        Float64MultiArray,
        sensors,
        max_linear=max_lin,
        max_angular=max_ang,
    )
    node.create_subscription(Odometry, ODOM_TOPIC, sensors.update_odom, 10)
    node.create_subscription(JointState, JOINT_STATES_TOPIC, _joint_cb(sensors), 10)
    node.create_subscription(LaserScan, SCAN_TOPIC, sensors.update_scan, 10)

    route = build_shelf_route(final_xy)
    follower = WaypointFollower(
        route,
        final_yaw=final_yaw,
        max_lin=max_lin,
        max_ang=max_ang,
        pos_tol=0.12,
    )
    period = 1.0 / RATE_HZ
    arrived = False
    blocked = False
    x = y = yaw = 0.0
    last_log = 0.0
    start = time.monotonic()
    try:
        _wait_odom(rclpy_mod, node, sensors, timeout_sec=8.0)
        for _ in range(20):
            rclpy_mod.spin_once(node, timeout_sec=0.05)
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            rclpy_mod.spin_once(node, timeout_sec=period)
            if sensors.odom is None:
                continue
            x, y, yaw = pose_from_odom(sensors.odom)
            linear, angular, arrived = follower.step(x, y, yaw)
            forward = min_forward_range_from_scan(sensors.scan)
            blocked = linear > 0.0 and forward < FORWARD_STOP_M
            if blocked:
                linear = 0.0
            _aim_head(controller)
            controller.publish_velocity(linear, angular)
            if time.monotonic() - last_log >= 2.0:
                print(
                    f"drive_to_shelf: xy=({x:.2f},{y:.2f}) yaw={yaw:.2f} "
                    f"wp={follower.idx}/{len(follower.waypoints)} fwd={forward:.2f}",
                    file=sys.stderr,
                )
                last_log = time.monotonic()
            if arrived:
                break
        controller.stop_base()
        settle_until = time.monotonic() + HEAD_SETTLE_S
        while time.monotonic() < settle_until:
            _aim_head(controller)
            rclpy_mod.spin_once(node, timeout_sec=period)
        in_zone = in_picking_zone((x, y))
        return {
            "arrived": bool(arrived),
            "in_picking_zone": bool(in_zone),
            "odom_xy": [round(x, 3), round(y, 3)],
            "yaw": round(yaw, 3),
            "head_joints": _head_joints(sensors),
            "scan_head": {"yaw": AISLE_SCAN_HEAD[0], "pitch": AISLE_SCAN_HEAD[1]},
            "goal_xy": [float(route[-1][0]), float(route[-1][1])],
            "elapsed_s": round(time.monotonic() - start, 2),
            "blocked_by_scan": bool(blocked),
            "speed_scale": scale,
            "max_lin": round(max_lin, 3),
            "max_ang": round(max_ang, 3),
            "next": "scripts/run_p3_preview.sh",
            "note": "empty P3 markers usually means the head camera is not facing a shelf",
        }
    finally:
        controller.stop_base()
        try:
            _aim_head(controller)
        except Exception:
            controller.stop_all()
        node.destroy_node()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Drive from delivery spawn to the shelf aisle")
    parser.add_argument("--timeout", type=float, default=90.0)
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
