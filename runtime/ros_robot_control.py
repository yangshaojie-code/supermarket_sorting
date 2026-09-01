"""Bounded ROS 2 command publishers for the MMK2 supermarket client."""

import math
import time
from typing import Callable, Iterable, Optional

import numpy as np

from runtime.ros_contract import (
    CMD_VEL_TOPIC,
    HEAD_COMMAND_TOPIC,
    LEFT_ARM_COMMAND_TOPIC,
    RIGHT_ARM_COMMAND_TOPIC,
    SPINE_COMMAND_TOPIC,
)


class ControlSafetyError(RuntimeError):
    pass


def _bounded(values: Iterable[float], count: int, limit: float, label: str) -> list:
    array = np.asarray(list(values), dtype=float)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ControlSafetyError(f"{label} command must contain {count} finite values")
    if np.any(np.abs(array) > float(limit)):
        raise ControlSafetyError(f"{label} command exceeds safety limit {limit}")
    return array.tolist()


class RosRobotController:
    """Publish base and joint commands with explicit limits and stop behavior."""

    def __init__(
        self,
        node,
        twist_type,
        array_type,
        sensor_cache=None,
        max_linear: float = 0.35,
        max_angular: float = 0.65,
    ):
        self.node = node
        self.twist_type = twist_type
        self.array_type = array_type
        self.sensor_cache = sensor_cache
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.cmd_vel_pub = node.create_publisher(twist_type, CMD_VEL_TOPIC, 10)
        self.spine_pub = node.create_publisher(array_type, SPINE_COMMAND_TOPIC, 10)
        self.head_pub = node.create_publisher(array_type, HEAD_COMMAND_TOPIC, 10)
        self.left_arm_pub = node.create_publisher(array_type, LEFT_ARM_COMMAND_TOPIC, 10)
        self.right_arm_pub = node.create_publisher(array_type, RIGHT_ARM_COMMAND_TOPIC, 10)

    def publish_velocity(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        if not math.isfinite(linear_x) or not math.isfinite(angular_z):
            raise ControlSafetyError("base velocity must be finite")
        message = self.twist_type()
        message.linear.x = float(np.clip(linear_x, -self.max_linear, self.max_linear))
        message.angular.z = float(np.clip(angular_z, -self.max_angular, self.max_angular))
        self.cmd_vel_pub.publish(message)

    def stop_base(self) -> None:
        self.publish_velocity()

    def drive_for(self, linear_x: float, angular_z: float, duration: float,
                  spin_once: Optional[Callable[[float], None]] = None, rate_hz: float = 20.0) -> None:
        duration = float(duration)
        if duration < 0 or duration > 15.0:
            raise ControlSafetyError("one base command may run for at most 15 seconds")
        period = 1.0 / float(rate_hz)
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline:
                self.publish_velocity(linear_x, angular_z)
                if spin_once:
                    spin_once(min(period, max(0.0, deadline - time.monotonic())))
                else:
                    time.sleep(period)
        finally:
            self.stop_base()

    def _publish_array(self, publisher, values, count: int, limit: float, label: str) -> None:
        message = self.array_type()
        message.data = _bounded(values, count, limit, label)
        publisher.publish(message)

    def command_spine(self, position: float) -> None:
        self._publish_array(self.spine_pub, [position], 1, 1.0, "spine")

    def command_head(self, yaw_pitch) -> None:
        self._publish_array(self.head_pub, yaw_pitch, 2, math.pi, "head")

    def command_arm(self, arm: str, joints, gripper: Optional[float] = None) -> None:
        values = list(joints)
        if len(values) == 6:
            if gripper is None:
                raise ControlSafetyError("a six-joint arm command requires gripper")
            values.append(float(gripper))
        publisher = self.left_arm_pub if arm == "l" else self.right_arm_pub if arm == "r" else None
        if publisher is None:
            raise ControlSafetyError("arm must be 'l' or 'r'")
        self._publish_array(publisher, values, 7, 2 * math.pi, f"{arm} arm")

    def hold_current_joints(self) -> None:
        if self.sensor_cache is None or not self.sensor_cache.joint_names:
            return
        names = self.sensor_cache.joint_names
        positions = self.sensor_cache.joint_positions
        values = dict(zip(names, positions))
        required = ["slide_joint", "head_yaw_joint", "head_pitch_joint"]
        if all(name in values for name in required):
            self.command_spine(values["slide_joint"])
            self.command_head([values["head_yaw_joint"], values["head_pitch_joint"]])
        for side, prefix in (("l", "left"), ("r", "right")):
            arm_names = [f"{prefix}_arm_joint{i}" for i in range(1, 7)]
            gripper_name = f"{prefix}_arm_eef_gripper_joint"
            if all(name in values for name in arm_names + [gripper_name]):
                self.command_arm(side, [values[name] for name in arm_names], values[gripper_name])

    def stop_all(self) -> None:
        # Stopping the base is the critical safety action. Joint feedback can
        # be temporarily malformed during a simulator fault, so do not let a
        # failed hold command mask the original attempt error or abort cleanup.
        try:
            self.stop_base()
        finally:
            try:
                self.hold_current_joints()
            except (ControlSafetyError, ValueError, TypeError, FloatingPointError):
                pass
