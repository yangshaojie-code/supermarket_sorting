"""Reactive lidar gating for corridor waypoints. No ROS, no occupancy map."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

from runtime.scene_zones import (
    HUG_WEST_X,
    NAV_BOUNDS,
    SOUTH_PEEL_Y,
    in_center_wall_band,
    in_north_racks,
)
from runtime.waypoint_nav import WaypointFollower, min_forward_range, wrap_to_pi


def blocked_span(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    horizon: float,
    window: float = 0.50,
    range_min: float = 0.05,
) -> float:
    """Angular width of returns closer than ``horizon`` inside ±window."""
    lo: Optional[float] = None
    hi: Optional[float] = None
    for index, raw in enumerate(ranges):
        distance = float(raw)
        if not math.isfinite(distance) or distance < range_min or distance > float(horizon):
            continue
        angle = wrap_to_pi(float(angle_min) + index * float(angle_increment))
        if abs(angle) > float(window):
            continue
        lo = angle if lo is None else min(lo, angle)
        hi = angle if hi is None else max(hi, angle)
    if lo is None or hi is None:
        return 0.0
    return hi - lo


def sector_min(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    lo: float,
    hi: float,
    range_min: float = 0.05,
    range_max: float = 8.0,
) -> float:
    nearest = float("inf")
    for index, raw in enumerate(ranges):
        distance = float(raw)
        if not math.isfinite(distance) or distance < range_min or distance > range_max:
            continue
        angle = wrap_to_pi(float(angle_min) + index * float(angle_increment))
        if lo <= angle <= hi:
            nearest = min(nearest, distance)
    return nearest


@dataclass(frozen=True)
class ScanSectors:
    forward: float
    ahead: float
    left: float
    right: float
    left_front: float
    right_front: float


def scan_sectors(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float = 0.05,
    range_max: float = 8.0,
) -> ScanSectors:
    kwargs = {
        "ranges": ranges,
        "angle_min": angle_min,
        "angle_increment": angle_increment,
        "range_min": range_min,
        "range_max": range_max,
    }
    return ScanSectors(
        # Wide cone: slow / shoulder steer. Narrow cone: only this can jam.
        forward=min_forward_range(cone_half=0.40, **kwargs),
        ahead=min_forward_range(cone_half=0.22, **kwargs),
        left=sector_min(lo=0.50, hi=1.45, **kwargs),
        right=sector_min(lo=-1.45, hi=-0.50, **kwargs),
        left_front=sector_min(lo=0.22, hi=0.85, **kwargs),
        right_front=sector_min(lo=-0.85, hi=-0.22, **kwargs),
    )


def scan_sectors_from_msg(scan) -> Optional[ScanSectors]:
    if scan is None:
        return None
    return scan_sectors(
        scan.ranges,
        float(scan.angle_min),
        float(scan.angle_increment),
    )


def clip_to_nav_bounds(x: float, y: float, bounds=NAV_BOUNDS) -> Tuple[float, float]:
    return (
        min(bounds["x"][1], max(bounds["x"][0], float(x))),
        min(bounds["y"][1], max(bounds["y"][0], float(y))),
    )


def detour_xy(x: float, y: float, yaw: float, side: str, dist: float = 0.70) -> Tuple[float, float]:
    sign = 1.0 if side == "left" else -1.0
    heading = wrap_to_pi(float(yaw) + sign * 1.05)
    px, py = clip_to_nav_bounds(
        float(x) + float(dist) * math.cos(heading),
        float(y) + float(dist) * math.sin(heading),
    )
    if in_center_wall_band(px, py) and float(x) <= 0.20:
        px = min(px, HUG_WEST_X)
        px, py = clip_to_nav_bounds(px, py)
    if in_center_wall_band(px, py) and float(x) >= 1.35:
        px = max(px, 1.52)
        px, py = clip_to_nav_bounds(px, py)
    return px, py


def left_corridor_southbound(x: float, y: float, yaw: float, goal_x: float) -> bool:
    return (
        in_west_delivery_lane(x, y, goal_x)
        and math.sin(float(yaw)) < -0.30
    )


def in_west_delivery_lane(x: float, y: float, goal_x: float) -> bool:
    return (
        float(x) < 0.15
        and SOUTH_PEEL_Y + 0.15 < float(y) < 2.40
        and float(goal_x) < 0.0
    )


def pick_detour_side(
    x: float,
    y: float,
    yaw: float,
    sectors: ScanSectors,
    goal_x: float,
) -> str:
    """Left while facing south in the west corridor drives into the divider."""
    side = clearer_side(sectors)
    if in_west_delivery_lane(x, y, goal_x):
        sine, cosine = math.sin(float(yaw)), math.cos(float(yaw))
        if sine < -0.30:
            return "right"
        if cosine < -0.30:
            return "left"
        if cosine > 0.30:
            return "right"
        return "right"
    if float(x) >= 1.35 and math.sin(float(yaw)) > 0.30 and float(goal_x) > 0.0:
        return "left"
    if in_center_wall_band(x, y) and math.cos(float(yaw)) < -0.30 and float(goal_x) < 0.0:
        return "right" if float(y) >= 0.0 else side
    return side


def clearer_side(sectors: ScanSectors) -> str:
    left = sectors.left if math.isfinite(sectors.left) else 8.0
    right = sectors.right if math.isfinite(sectors.right) else 8.0
    left += 0.15 * (sectors.left_front if math.isfinite(sectors.left_front) else 8.0)
    right += 0.15 * (sectors.right_front if math.isfinite(sectors.right_front) else 8.0)
    return "left" if left >= right else "right"


@dataclass
class CorridorFollower:
    """Follow odom waypoints, slow/stop for lidar, splice a side detour when jammed."""

    follower: WaypointFollower
    stop_dist: float = 0.0
    slow_dist: float = 0.0
    detour_m: float = 0.70
    max_detours: int = 8
    blocked_s: float = 0.45
    _blocked_t: float = field(init=False, default=0.0)
    _detour_idx: int = field(init=False, default=-1)
    _avoid_ang: float = field(init=False, default=0.55)
    _brake_s: float = field(init=False, default=0.85)
    _clear_s: float = field(init=False, default=0.30)
    _stuck_t: float = field(init=False, default=0.0)
    _reverse_left: float = field(init=False, default=0.0)
    _last_xy: Optional[Tuple[float, float]] = field(init=False, default=None)
    _jam_extra: float = field(init=False, default=0.10)
    _ahead_cone: float = field(init=False, default=0.22)
    _detour_at: Optional[Tuple[float, float]] = field(init=False, default=None)
    _jam_latch: bool = field(init=False, default=False)
    detours: int = field(init=False, default=0)
    last_status: str = field(init=False, default="idle")

    def __post_init__(self) -> None:
        max_lin = abs(self.follower.max_lin)
        # Physical clearance, not 8x-speed braking. Scaling stop_dist with
        # max_lin made the shelf/center wall look like a box at ~1 m.
        if self.stop_dist <= 0.0:
            self.stop_dist = min(0.55, max(0.40, 0.18 * max_lin))
        if self.slow_dist <= 0.0:
            self.slow_dist = min(1.05, max(0.80, self.stop_dist + 0.40))
        # Seconds of travel that must be clear ahead for full cruise.
        self._clear_s = 0.30 if max_lin > 1.0 else 0.45
        self._brake_s = 0.25 if max_lin > 1.0 else 0.35
        self._avoid_ang = min(1.05, max(0.45, 0.55 * abs(self.follower.max_ang)))

    def _clear_horizon(self) -> float:
        return self.stop_dist + abs(self.follower.max_lin) * self._clear_s

    def _occlusion_kind(self, sectors: ScanSectors, span: float) -> str:
        """clear / nudge (steer while driving) / approach (keep speed) / block (stop-turn)."""
        ahead = sectors.ahead if math.isfinite(sectors.ahead) else 8.0
        if ahead > self._clear_horizon():
            return "clear"
        left_f = sectors.left_front if math.isfinite(sectors.left_front) else 8.0
        right_f = sectors.right_front if math.isfinite(sectors.right_front) else 8.0
        side_open = left_f > 0.80 or right_f > 0.80
        small = span <= 0.32
        far = ahead > 0.90
        if side_open and (small or far):
            return "nudge"
        if ahead > self.stop_dist + self._jam_extra:
            return "approach"
        return "block"

    def _note_motion(self, x: float, y: float, dt: float, trying_forward: bool, jammed: bool = False) -> bool:
        here = (float(x), float(y))
        if self._last_xy is None:
            self._last_xy = here
            self._stuck_t = 0.0
            return False
        moved = math.hypot(here[0] - self._last_xy[0], here[1] - self._last_xy[1])
        if moved > 0.04:
            self._stuck_t = 0.0
            self._last_xy = here
            return True
        if trying_forward or jammed:
            self._stuck_t += float(dt)
        return False

    def _reverse_cmd(self, x: float, y: float, yaw: float, turn: float) -> Tuple[float, float, bool, str]:
        # Backing up while facing south in the picking aisle drives into the racks.
        if float(y) > 2.35 and math.sin(float(yaw)) < -0.25:
            self._stuck_t = 0.0
            self.last_status = "turn"
            return 0.0, turn, False, self.last_status
        self._detour_idx = -1
        self._blocked_t = 0.0
        self._stuck_t = 0.0
        self._reverse_left = 0.35
        self.last_status = "reverse"
        back = -min(0.45, max(0.20, 0.22 * abs(self.follower.max_lin)))
        return back, turn, False, self.last_status

    def _cap_linear_for_range(self, linear: float, forward: float) -> float:
        """Keep speed low enough to stop before stop_dist."""
        if linear <= 0.04 or not math.isfinite(forward):
            return linear
        horizon = max(0.0, float(forward) - self.stop_dist)
        return min(linear, horizon / self._brake_s)

    def _side_steer(self, x: float, y: float, yaw: float, sectors: ScanSectors) -> float:
        """Nudge away from a nearby shoulder. Does not stop or splice a detour."""
        left_f = sectors.left_front if math.isfinite(sectors.left_front) else 8.0
        right_f = sectors.right_front if math.isfinite(sectors.right_front) else 8.0
        dodge = 0.0
        if left_f < 0.70 and left_f + 0.12 < right_f:
            dodge = -0.35
        elif right_f < 0.70 and right_f + 0.12 < left_f:
            dodge = 0.35
        if dodge == 0.0 or self.follower.idx >= len(self.follower.waypoints):
            return 0.0
        _dist, route_err, _steer_err, _corner = self.follower._aim_error(x, y, yaw)
        if abs(route_err) > 0.30:
            return 0.0
        return dodge

    def step(
        self,
        x: float,
        y: float,
        yaw: float,
        ranges: Optional[Sequence[float]] = None,
        angle_min: float = -math.pi,
        angle_increment: float = 0.0,
        dt: float = 0.05,
    ) -> Tuple[float, float, bool, str]:
        linear, angular, done = self.follower.step(x, y, yaw)
        if done:
            self.last_status = "arrived"
            return 0.0, 0.0, True, self.last_status
        if self._reverse_left > 0.0:
            self._reverse_left = max(0.0, self._reverse_left - float(dt))
            back = -min(0.45, max(0.20, 0.22 * abs(self.follower.max_lin)))
            self.last_status = "reverse"
            return back, angular, False, self.last_status
        if ranges is None:
            self._note_motion(x, y, dt, trying_forward=(linear > 0.04))
            self.last_status = "no_scan"
            return linear, angular, False, self.last_status
        sectors = scan_sectors(ranges, angle_min, angle_increment)
        near_goal = False
        if self.follower.waypoints:
            gx, gy = self.follower.waypoints[-1]
            near_goal = (
                self.follower.idx >= len(self.follower.waypoints) - 1
                and math.hypot(gx - float(x), gy - float(y)) < 0.90
            )
        goal_x = self.follower.waypoints[-1][0] if self.follower.waypoints else x
        span = blocked_span(
            ranges, angle_min, angle_increment, horizon=self._clear_horizon()
        )
        kind = self._occlusion_kind(sectors, span)
        raw_jam = kind == "block"
        if raw_jam:
            self._jam_latch = True
        translated = self._note_motion(
            x, y, dt, trying_forward=(linear > 0.04), jammed=raw_jam or self._jam_latch
        )
        if not raw_jam and kind in ("clear", "nudge"):
            if not in_west_delivery_lane(x, y, goal_x) or translated:
                self._jam_latch = False
        jammed = self._jam_latch if in_west_delivery_lane(x, y, goal_x) else raw_jam
        if (
            self._stuck_t >= 0.90
            and not near_goal
            and kind in ("approach", "block")
        ):
            jammed = True
            self._jam_latch = True
        if jammed:
            self._blocked_t += float(dt)
            side = pick_detour_side(x, y, yaw, sectors, goal_x)
            turn = (1.0 if side == "left" else -1.0) * self._avoid_ang
            if in_west_delivery_lane(x, y, goal_x):
                face = math.pi if float(x) > HUG_WEST_X + 0.04 else -math.pi / 2.0
                face_err = wrap_to_pi(face - float(yaw))
                turn = max(-self._avoid_ang, min(self._avoid_ang, 1.6 * face_err))
            frozen = self.follower.idx == self._detour_idx
            turning = self.follower.mode == "turn"
            moved_since = (
                self._detour_at is None
                or math.hypot(float(x) - self._detour_at[0], float(y) - self._detour_at[1])
                >= 0.50
            )
            immobile = self._stuck_t >= 1.50
            can_detour = (
                self._blocked_t >= self.blocked_s
                and self.detours < self.max_detours
                and not frozen
                and (moved_since or immobile)
            )
            if can_detour:
                dist = self.detour_m
                if left_corridor_southbound(x, y, yaw, goal_x):
                    dist = max(self.detour_m, 1.00)
                if in_west_delivery_lane(x, y, goal_x) and float(x) <= HUG_WEST_X - 0.20:
                    point = clip_to_nav_bounds(float(x), float(y) - max(dist, 0.90))
                    south_err = wrap_to_pi(-math.pi / 2.0 - float(yaw))
                    turn = (1.0 if south_err > 0.0 else -1.0) * self._avoid_ang
                else:
                    point = detour_xy(x, y, yaw, side, dist)
                    if in_center_wall_band(*point) and left_corridor_southbound(x, y, yaw, goal_x):
                        side = "right"
                        turn = -self._avoid_ang
                        point = detour_xy(x, y, yaw, side, dist)
                if in_center_wall_band(*point):
                    return self._reverse_cmd(x, y, yaw, turn)
                self.follower.insert_ahead(point)
                self.follower.rebase_west_south(x, y)
                self.follower.mode = "turn"
                self.follower._have_yaw_err = False
                self._stuck_t = 0.0
                self.detours += 1
                self._detour_idx = self.follower.idx
                self._detour_at = (float(x), float(y))
                self._blocked_t = 0.0
                self.last_status = f"detour_{side}"
                return 0.0, turn, False, self.last_status
            if frozen and self._blocked_t < 2.50:
                self.last_status = "turn" if turning else f"blocked_{side}"
                return 0.0, angular if turning else turn, False, self.last_status
            if turning and self._stuck_t < 2.00:
                self.last_status = "turn"
                return 0.0, angular, False, self.last_status
            # Reverse only after detours are spent. A stuck-induced jam used
            # to reverse immediately (stuck_t already 0.90) and never splice.
            if self.detours >= self.max_detours or (
                frozen and self._blocked_t >= 2.50
            ):
                return self._reverse_cmd(x, y, yaw, turn)
            self.last_status = f"blocked_{side}"
            return 0.0, turn, False, self.last_status
        self._blocked_t = 0.0
        dodge = 0.0
        if kind == "nudge":
            dodge = self._side_steer(x, y, yaw, sectors)
        elif kind == "clear":
            dodge = self._side_steer(x, y, yaw, sectors)
            # Only scrape-avoid when a shoulder is really close.
            left_f = sectors.left_front if math.isfinite(sectors.left_front) else 8.0
            right_f = sectors.right_front if math.isfinite(sectors.right_front) else 8.0
            if min(left_f, right_f) > 0.50:
                dodge = 0.0
        if kind == "approach" and linear > 0.04:
            # Keep speed until the last ~0.25 m above stop, then halt and turn.
            if math.isfinite(sectors.ahead) and sectors.ahead < self.stop_dist + 0.25:
                linear = self._cap_linear_for_range(linear, sectors.ahead)
            if in_north_racks(x, y) and math.sin(float(yaw)) > 0.12:
                back = -min(0.45, max(0.20, 0.22 * abs(self.follower.max_lin)))
                self.last_status = "reverse"
                return back, angular, False, self.last_status
        if dodge != 0.0 and abs(linear) > 0.04:
            limit = abs(self.follower.max_ang)
            angular = max(-limit, min(limit, angular + dodge))
            self.last_status = "nudge"
            return linear, angular, False, self.last_status
        if abs(linear) <= 0.04:
            self.last_status = "turn" if abs(angular) > 0.05 else "hold"
            return linear, angular, False, self.last_status
        self.last_status = "cruise" if kind == "clear" else kind
        return linear, angular, False, self.last_status
