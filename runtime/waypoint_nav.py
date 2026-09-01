"""Odom waypoint following for the supermarket aisle. No ROS import."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from runtime.scene_zones import ROUTE_DELIVERY_TO_SHELF, SHELF_APPROACH_XY, SHELF_FACE_YAW


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pose_from_odom(odom) -> Tuple[float, float, float]:
    pose = odom.pose.pose
    position = pose.position
    orientation = pose.orientation
    return (
        float(position.x),
        float(position.y),
        yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w),
    )


def build_shelf_route(final_xy: Optional[Sequence[float]] = None) -> List[Tuple[float, float]]:
    goal = tuple(float(v) for v in (final_xy if final_xy is not None else SHELF_APPROACH_XY))
    route: List[Tuple[float, float]] = []
    for point in (*ROUTE_DELIVERY_TO_SHELF, goal):
        xy = (float(point[0]), float(point[1]))
        if route and math.hypot(xy[0] - route[-1][0], xy[1] - route[-1][1]) < 0.05:
            continue
        route.append(xy)
    return route


def min_forward_range(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    cone_half: float = 0.40,
    range_min: float = 0.05,
    range_max: float = 8.0,
) -> float:
    """Nearest return inside a forward cone. inf means nothing valid in the cone."""
    nearest = float("inf")
    for index, raw in enumerate(ranges):
        distance = float(raw)
        if not math.isfinite(distance) or distance < range_min or distance > range_max:
            continue
        angle = wrap_to_pi(float(angle_min) + index * float(angle_increment))
        if abs(angle) <= cone_half:
            nearest = min(nearest, distance)
    return nearest


def min_forward_range_from_scan(scan, cone_half: float = 0.40) -> float:
    if scan is None:
        return float("inf")
    return min_forward_range(
        scan.ranges,
        float(scan.angle_min),
        float(scan.angle_increment),
        cone_half=cone_half,
    )


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class WaypointFollower:
    """Turn-then-drive along world-frame waypoints, then face ``final_yaw``."""

    route: Sequence[Sequence[float]]
    final_yaw: float = SHELF_FACE_YAW
    pos_tol: float = 0.08
    turn_tol: float = 0.05
    max_lin: float = 0.30
    max_ang: float = 0.55
    ang_gain: float = 2.2
    idx: int = 0
    mode: str = "turn"
    waypoints: List[Tuple[float, float]] = field(init=False)
    brake_dist: float = field(init=False)

    def __post_init__(self) -> None:
        self.waypoints = [(float(x), float(y)) for x, y in self.route]
        self.final_yaw = float(self.final_yaw)
        self.max_lin = float(self.max_lin)
        self.max_ang = float(self.max_ang)
        self.brake_dist = max(0.8, abs(self.max_lin) * 1.5)

    def step(self, x: float, y: float, yaw: float) -> Tuple[float, float, bool]:
        if self.idx < len(self.waypoints):
            target_x, target_y = self.waypoints[self.idx]
            dx = target_x - float(x)
            dy = target_y - float(y)
            dist = math.hypot(dx, dy)
            yaw_err = wrap_to_pi(math.atan2(dy, dx) - float(yaw))
            if dist <= self.pos_tol:
                self.idx += 1
                self.mode = "turn"
                return 0.0, 0.0, False
            if self.mode == "turn":
                if abs(yaw_err) < self.turn_tol:
                    self.mode = "drive"
                return 0.0, _clip(self.ang_gain * yaw_err, self.max_ang), False
            if abs(yaw_err) < 0.05 or dist < 0.25:
                angular = 0.0
            else:
                angular = _clip(self.ang_gain * yaw_err, self.max_ang)
            align = max(0.0, math.cos(yaw_err))
            linear = self.max_lin * align * min(1.0, dist / self.brake_dist)
            return linear, angular, False

        yaw_err = wrap_to_pi(self.final_yaw - float(yaw))
        if abs(yaw_err) < self.turn_tol:
            return 0.0, 0.0, True
        return 0.0, _clip(self.ang_gain * yaw_err, self.max_ang), False
