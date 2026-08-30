#!/usr/bin/env python3
"""Read-only P1 preflight for the supermarket Client image.

Subscribes to Server feedback, validates joints, builds ``base_link <-
head_camera`` from MJCF + JointState, and smoke-tests MMK2Kdl.  Does not
publish motion unless ``--stop-base`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from runtime.head_camera_kinematics import base_to_head_camera_from_joint_state
from runtime.motion_planning import IKSolveError, MMK2KdlBackend
from runtime.ros_contract import (
    DEPTH_TOPIC,
    HEAD_CAMERA_FRAME,
    JOINT_STATES_TOPIC,
    ODOM_TOPIC,
    REQUIRED_JOINT_NAMES,
    RGB_CAMERA_INFO_TOPIC,
    RGB_TOPIC,
    SCAN_TOPIC,
    TASK_TOPIC,
    TF_TOPIC,
    topic_contract,
)
from runtime.orchestrator import MissionOrchestrator
from runtime.ros_robot_control import RosRobotController
from runtime.ros_sensor_utils import SensorCache, SensorDataError, TransformStore
from runtime.task_protocol import TaskProtocolError, parse_task_payload, unknown_kinds


def _safe_callback(callback, errors, label):
    def wrapped(message):
        try:
            callback(message)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    return wrapped


class RuntimePreflight:
    def __init__(self, rclpy_mod, create_publishers: bool):
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
        from std_msgs.msg import Float64MultiArray, String
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._rclpy = rclpy_mod
        self.errors = []
        self.camera_transform_source = None
        self.sensors = SensorCache()
        self.transforms = TransformStore()
        self.node = rclpy_mod.create_node("supermarket_runtime_preflight")
        self.controller = None
        if create_publishers:
            self.controller = RosRobotController(
                self.node, Twist, Float64MultiArray, self.sensors
            )

        latch = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.node.create_subscription(
            String, TASK_TOPIC, _safe_callback(self.sensors.update_task, self.errors, "task"), latch
        )
        self.node.create_subscription(
            Image, RGB_TOPIC, _safe_callback(self.sensors.update_rgb, self.errors, "rgb"), 10
        )
        self.node.create_subscription(
            Image, DEPTH_TOPIC, _safe_callback(self.sensors.update_depth, self.errors, "depth"), 10
        )
        self.node.create_subscription(
            CameraInfo, RGB_CAMERA_INFO_TOPIC,
            _safe_callback(self.sensors.update_camera_info, self.errors, "camera_info"), 10,
        )
        self.node.create_subscription(
            JointState, JOINT_STATES_TOPIC,
            _safe_callback(self._joint_state_callback, self.errors, "joint_states"), 10,
        )
        self.node.create_subscription(
            Odometry, ODOM_TOPIC, _safe_callback(self.sensors.update_odom, self.errors, "odom"), 10
        )
        self.node.create_subscription(
            LaserScan, SCAN_TOPIC, _safe_callback(self.sensors.update_scan, self.errors, "scan"), 10
        )
        self.node.create_subscription(
            TFMessage, TF_TOPIC, _safe_callback(self.transforms.update, self.errors, "tf"), 10
        )

    def _joint_state_callback(self, message) -> None:
        self.sensors.update_joint_state(message)
        matrix = base_to_head_camera_from_joint_state(message.name, message.position)
        frame = os.environ.get("SUPERMARKET_HEAD_CAMERA_FRAME", HEAD_CAMERA_FRAME).lstrip("/")
        self.transforms.set_transform("base_link", frame, matrix)
        self.camera_transform_source = "mjcf_joint_state"

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def wait_for_robot_state(self, timeout_sec: float = 8.0) -> None:
        required = set(REQUIRED_JOINT_NAMES)
        deadline = time.monotonic() + float(timeout_sec)
        while self.sensors.odom is None or not required.issubset(self.sensors.joint_names):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(required.difference(self.sensors.joint_names))
                raise TimeoutError(
                    f"robot state timeout: odom={self.sensors.odom is not None}, missing_joints={missing}"
                )
            self.spin_once(min(0.05, remaining))

    def wait_for_snapshot(self, timeout_sec: float = 8.0):
        deadline = time.monotonic() + float(timeout_sec)
        while True:
            try:
                return self.sensors.wait_snapshot(timeout=0.0)
            except TimeoutError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(str(exc)) from exc
                self.spin_once(min(0.05, remaining))

    def close(self, stop_robot: bool = False) -> None:
        if stop_robot and self.controller is not None:
            self.controller.stop_all()
        self.node.destroy_node()


def build_report(preflight: RuntimePreflight, snapshot=None) -> dict:
    names = preflight.sensors.joint_names
    positions = preflight.sensors.joint_positions
    camera = None
    if names:
        matrix = base_to_head_camera_from_joint_state(names, positions)
        camera = {
            "source": preflight.camera_transform_source,
            "base_from_head_camera_xyz": matrix[:3, 3].tolist(),
        }
        try:
            camera["tf_lookup_xyz"] = preflight.transforms.lookup(
                "base_link", HEAD_CAMERA_FRAME
            )[:3, 3].tolist()
        except SensorDataError as exc:
            camera["tf_lookup_error"] = str(exc)

    kdl = {"backend": "MMK2KdlBackend"}
    try:
        backend = MMK2KdlBackend()
        reference = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])
        pose = backend.forward("r", 0.10, reference)
        solution = backend.solve(pose[:3, 3], pose[:3, :3], "r", 0.10, reference)
        recovered = backend.forward("r", 0.10, solution)
        kdl["fk_ik_round_trip_ok"] = bool(np.allclose(recovered, pose, atol=1e-6))
    except (IKSolveError, ValueError, ImportError) as exc:
        kdl["error"] = str(exc)

    odom_xy = None
    if preflight.sensors.odom is not None:
        pos = preflight.sensors.odom.pose.pose.position
        odom_xy = [float(pos.x), float(pos.y)]

    rgb = None
    if snapshot is not None:
        rgb = {
            "shape": list(snapshot.rgb.shape),
            "depth_shape": list(snapshot.depth_m.shape),
            "frame": snapshot.camera_frame,
            "fx": snapshot.intrinsics.fx,
            "fy": snapshot.intrinsics.fy,
        }

    scan = None
    if preflight.sensors.scan is not None:
        ranges = list(preflight.sensors.scan.ranges)
        scan = {"n_ranges": len(ranges), "frame": getattr(preflight.sensors.scan, "header", SimpleNamespace(frame_id="")).frame_id}

    mission = None
    if preflight.sensors.task_raw is not None:
        try:
            task = parse_task_payload(preflight.sensors.task_raw)
            orch = MissionOrchestrator()
            orch.load_task(task)
            mission = {
                "run_prefix": task.run_prefix,
                "count": task.count,
                "unknown_kinds": list(unknown_kinds(task)),
                "planned_cycles": orch.planned_cycles(),
            }
        except TaskProtocolError as exc:
            preflight.errors.append(f"task: {exc}")

    return {
        "contract": topic_contract(),
        "joints": list(names),
        "odom_xy": odom_xy,
        "camera": camera,
        "kdl": kdl,
        "rgb_d": rgb,
        "task_raw": preflight.sensors.task_raw,
        "mission": mission,
        "scan": scan,
        "errors": list(preflight.errors),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P1 supermarket runtime preflight")
    parser.add_argument(
        "--stop-base",
        action="store_true",
        help="create control publishers and send a zero cmd_vel on exit",
    )
    args = parser.parse_args(argv)

    import rclpy

    rclpy.init()
    preflight = RuntimePreflight(rclpy, create_publishers=args.stop_base)
    snapshot = None
    try:
        preflight.wait_for_robot_state(timeout_sec=10.0)
        try:
            snapshot = preflight.wait_for_snapshot(timeout_sec=8.0)
        except TimeoutError as exc:
            preflight.errors.append(str(exc))
        for _ in range(20):
            preflight.spin_once(0.05)
        report = build_report(preflight, snapshot)
        print(json.dumps(report, indent=2, default=str))
        if report["kdl"].get("fk_ik_round_trip_ok") and report["camera"] and report["odom_xy"]:
            return 0
        return 1
    finally:
        preflight.close(stop_robot=args.stop_base)
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
