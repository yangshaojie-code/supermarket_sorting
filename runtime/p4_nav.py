#!/usr/bin/env python3
"""P4 corridor nav: grid A* + lidar brake. No occupancy ROS stack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from runtime.head_camera_kinematics import AISLE_SCAN_HEAD, AISLE_SCAN_SPINE
from runtime.grid_planner import STOP_DIST, GridNavController
from runtime.lidar_avoid import scan_sectors_from_msg
from runtime.ros_contract import JOINT_STATES_TOPIC, ODOM_TOPIC, SCAN_TOPIC
from runtime.ros_robot_control import RosRobotController
from runtime.ros_sensor_utils import SensorCache
from runtime.scene_zones import (
    DELIVERY_APPROACH_XY,
    DELIVERY_FACE_YAW,
    SHELF_APPROACH_XY,
    SHELF_FACE_YAW,
    in_delivery_base,
    in_picking_zone,
)
from runtime.waypoint_nav import build_delivery_route, build_shelf_route, pose_from_odom

RATE_HZ = 20.0
BASE_LIN = 0.30
BASE_ANG = 0.55
SPEED_SCALE = 4.0
MAX_ANG_CAP = 1.65
HEAD_SETTLE_S = 1.0
DEFAULT_TIMEOUT_S = 150.0
LOG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "p4"


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _wait_odom(rclpy_mod, node, sensors: SensorCache, timeout_sec: float) -> None:
    deadline = time.monotonic() + float(timeout_sec)
    while sensors.odom is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for odom")
        rclpy_mod.spin_once(node, timeout_sec=0.05)


def _joint_cb(sensors: SensorCache):
    def wrapped(message):
        try:
            sensors.update_joint_state(message)
        except Exception:
            pass
    return wrapped


def _head_joints(sensors: SensorCache) -> dict:
    values = dict(zip(sensors.joint_names, [float(v) for v in sensors.joint_positions]))
    return {
        "slide": values.get("slide_joint"),
        "head_yaw": values.get("head_yaw_joint"),
        "head_pitch": values.get("head_pitch_joint"),
    }


def _goal_spec(goal: str, final_xy=None, final_yaw=None):
    goal = str(goal).strip().lower()
    if goal in {"shelf", "pick", "picking"}:
        xy = final_xy if final_xy is not None else SHELF_APPROACH_XY
        yaw = SHELF_FACE_YAW if final_yaw is None else float(final_yaw)
        return {
            "name": "shelf",
            "route": build_shelf_route(xy),
            "yaw": yaw,
            "aim_head": True,
        }
    if goal in {"delivery", "deliver", "table"}:
        xy = final_xy if final_xy is not None else DELIVERY_APPROACH_XY
        yaw = DELIVERY_FACE_YAW if final_yaw is None else float(final_yaw)
        return {
            "name": "delivery",
            "route": build_delivery_route(xy),
            "yaw": yaw,
            "aim_head": False,
        }
    raise ValueError("goal must be shelf or delivery")


def navigate(
    rclpy_mod,
    goal: str = "shelf",
    timeout_sec: float = DEFAULT_TIMEOUT_S,
    final_xy=None,
    final_yaw=None,
    speed_scale: float = SPEED_SCALE,
) -> dict:
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState, LaserScan
    from std_msgs.msg import Float64MultiArray

    spec = _goal_spec(goal, final_xy=final_xy, final_yaw=final_yaw)
    scale = max(0.1, float(speed_scale))
    max_lin = BASE_LIN * scale
    max_ang = min(MAX_ANG_CAP, BASE_ANG * scale)
    node = rclpy_mod.create_node(f"supermarket_p4_{spec['name']}")
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

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{spec['name']}_{stamp}.log"
    json_path = LOG_DIR / f"{spec['name']}_{stamp}.json"
    log_file = log_path.open("w", encoding="utf-8")
    old_err = sys.stderr
    sys.stderr = _Tee(old_err, log_file)

    nav = None
    arrived = False
    x = y = yaw = 0.0
    last_log = 0.0
    last_status = "idle"
    start = time.monotonic()
    period = 1.0 / RATE_HZ
    try:
        _wait_odom(rclpy_mod, node, sensors, timeout_sec=8.0)
        for _ in range(20):
            rclpy_mod.spin_once(node, timeout_sec=0.05)
        x, y, yaw = pose_from_odom(sensors.odom)
        goal_xy = spec["route"][-1]
        nav = GridNavController(
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            final_yaw=spec["yaw"],
            max_lin=max_lin,
            max_ang=max_ang,
        )
        print(
            f"p4_{spec['name']}: start xy=({x:.2f},{y:.2f}) yaw={yaw:.2f} "
            f"goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f}) "
            f"stop={STOP_DIST:.2f} planner=grid "
            f"max_lin={max_lin:.2f} max_ang={max_ang:.2f}",
            file=sys.stderr,
        )
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            rclpy_mod.spin_once(node, timeout_sec=period)
            if sensors.odom is None:
                continue
            x, y, yaw = pose_from_odom(sensors.odom)
            scan = sensors.scan
            ranges = None
            angle_min = -3.14159
            angle_inc = 0.0
            if scan is not None:
                ranges = scan.ranges
                angle_min = float(scan.angle_min)
                angle_inc = float(scan.angle_increment)
            linear, angular, arrived, last_status = nav.step(
                x, y, yaw, ranges, angle_min, angle_inc, dt=period, now=time.monotonic() - start,
            )
            if spec["aim_head"]:
                controller.command_spine(AISLE_SCAN_SPINE)
                controller.command_head(AISLE_SCAN_HEAD)
            controller.publish_velocity(linear, angular)
            if time.monotonic() - last_log >= 2.0:
                sectors = scan_sectors_from_msg(scan)
                fwd = sectors.forward if sectors is not None else float("inf")
                wp = 0
                nwp = 0
                if nav.follower is not None:
                    wp = nav.follower.idx
                    nwp = len(nav.follower.waypoints)
                print(
                    f"p4_{spec['name']}: xy=({x:.2f},{y:.2f}) yaw={yaw:.2f} "
                    f"wp={wp}/{nwp} fwd={fwd:.2f} lin={linear:.2f} ang={angular:.2f} "
                    f"{last_status} plan_id={nav.plan_id} detours=0 "
                    f"plan_ms={nav.last_plan_ms:.1f}",
                    file=sys.stderr,
                )
                last_log = time.monotonic()
            if arrived:
                break
        controller.stop_base()
        if spec["aim_head"]:
            settle_until = time.monotonic() + HEAD_SETTLE_S
            while time.monotonic() < settle_until:
                controller.command_spine(AISLE_SCAN_SPINE)
                controller.command_head(AISLE_SCAN_HEAD)
                rclpy_mod.spin_once(node, timeout_sec=period)
        in_zone = (
            in_picking_zone((x, y)) if spec["name"] == "shelf" else in_delivery_base((x, y))
        )
        report = {
            "arrived": bool(arrived),
            "goal": spec["name"],
            "in_zone": bool(in_zone),
            "in_picking_zone": bool(in_picking_zone((x, y))),
            "in_delivery_base": bool(in_delivery_base((x, y))),
            "odom_xy": [round(x, 3), round(y, 3)],
            "yaw": round(yaw, 3),
            "head_joints": _head_joints(sensors),
            "goal_xy": [float(spec["route"][-1][0]), float(spec["route"][-1][1])],
            "route": (
                [[round(px, 3), round(py, 3)] for px, py in nav.follower.waypoints]
                if nav is not None and nav.follower is not None
                else [[round(px, 3), round(py, 3)] for px, py in spec["route"]]
            ),
            "elapsed_s": round(time.monotonic() - start, 2),
            "status": last_status,
            "detours": 0,
            "plan_id": int(nav.plan_id) if nav is not None else 0,
            "plan_reason": nav.last_plan_reason if nav is not None else "",
            "speed_scale": scale,
            "max_lin": round(max_lin, 3),
            "max_ang": round(max_ang, 3),
            "stop_dist": STOP_DIST,
            "slow_dist": None,
            "blocked_by_scan": last_status.startswith("safety") or last_status.startswith("blocked"),
            "log_json": str(json_path),
            "log_txt": str(log_path),
        }
        text = json.dumps(report, indent=2)
        json_path.write_text(text + "\n", encoding="utf-8")
        (LOG_DIR / "last.json").write_text(text + "\n", encoding="utf-8")
        log_file.write("\n" + text + "\n")
        log_file.flush()
        print(f"p4 log: {log_path}", file=sys.stderr)
        return report
    finally:
        controller.stop_base()
        node.destroy_node()
        sys.stderr = old_err
        log_file.close()


def drive_to_shelf(rclpy_mod, timeout_sec: float = DEFAULT_TIMEOUT_S, final_xy=SHELF_APPROACH_XY,
                   final_yaw: float = SHELF_FACE_YAW, speed_scale: float = SPEED_SCALE) -> dict:
    report = navigate(
        rclpy_mod,
        goal="shelf",
        timeout_sec=timeout_sec,
        final_xy=final_xy,
        final_yaw=final_yaw,
        speed_scale=speed_scale,
    )
    report["scan_head"] = {"yaw": AISLE_SCAN_HEAD[0], "pitch": AISLE_SCAN_HEAD[1]}
    report["next"] = "scripts/run_p3_preview.sh"
    report["note"] = "empty P3 markers usually means the head camera is not facing a shelf"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P4 lidar corridor navigation")
    parser.add_argument("--goal", choices=["shelf", "delivery"], default="shelf")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--speed-scale", type=float, default=SPEED_SCALE)
    args = parser.parse_args(argv)

    import rclpy

    rclpy.init()
    try:
        report = navigate(
            rclpy,
            goal=args.goal,
            timeout_sec=args.timeout,
            speed_scale=args.speed_scale,
        )
        print(json.dumps(report, indent=2))
        if report["arrived"] and report["in_zone"]:
            return 0
        return 1
    except TimeoutError as exc:
        print(json.dumps({"arrived": False, "in_zone": False, "error": str(exc)}, indent=2))
        return 1
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
