"""Odom waypoint following for the supermarket aisle. No ROS import."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from runtime.scene_zones import (
    DELIVERY_APPROACH_XY,
    EAST_STUB_X_MIN,
    HUG_WEST_X,
    NAV_BOUNDS,
    ROUTE_DELIVERY_TO_SHELF,
    SHELF_APPROACH_XY,
    SHELF_CORNER_XY,
    SHELF_CORNER_Y,
    SHELF_FACE_YAW,
    SOUTH_PEEL_Y,
    WEST_LANE_Y,
    in_center_wall_band,
    in_east_shelf_stub,
    in_north_racks,
    in_picking_zone,
    in_south_east_stub,
    near_divider_nw_corner,
)


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


def _segment_crosses_divider(x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if driving straight would cut through the center wall slab."""
    if in_center_wall_band(x0, y0) or in_center_wall_band(x1, y1):
        return True
    left, right = 0.20, 1.45
    y_lo, y_hi = -3.08, 1.90
    if (y0 > y_hi and y1 > y_hi) or (y0 < y_lo and y1 < y_lo):
        return False
    if (x0 < left and x1 < left) or (x0 > right and x1 > right):
        return False
    return (x0 - left) * (x1 - left) <= 0.0 or (x0 - right) * (x1 - right) <= 0.0


def prune_passed_waypoints(
    route: Sequence[Sequence[float]],
    x: float,
    y: float,
    margin: float = 0.35,
) -> List[Tuple[float, float]]:
    """Drop corridor points that would send the robot backwards."""
    points = [(float(px), float(py)) for px, py in route]
    if points and points[-1][0] < 0.0 and points[-1][1] < 0.0:
        if in_center_wall_band(x, y) or _segment_crosses_divider(x, y, points[0][0], points[0][1]):
            return build_delivery_route(points[-1], start_xy=(x, y))
    if len(points) >= 2 and in_picking_zone((x, y)) and in_picking_zone(points[-1]):
        if not _segment_crosses_divider(x, y, points[-1][0], points[-1][1]):
            return [points[-1]]
    dists = [math.hypot(px - x, py - y) for px, py in points]
    nearest = min(range(len(points)), key=lambda i: dists[i])
    if nearest > 0 and dists[nearest] + margin < dists[0]:
        nx, ny = points[nearest]
        if not _segment_crosses_divider(x, y, nx, ny):
            points = points[nearest:]
    while len(points) >= 2:
        d0 = math.hypot(points[0][0] - x, points[0][1] - y)
        d1 = math.hypot(points[1][0] - x, points[1][1] - y)
        if d0 < 0.22 or d1 + margin < d0:
            if _segment_crosses_divider(x, y, points[1][0], points[1][1]) and not _segment_crosses_divider(
                x, y, points[0][0], points[0][1]
            ):
                break
            points.pop(0)
        else:
            break
    return points


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
    """Follow odom waypoints. Steer while driving; only spin in place for large heading error."""

    route: Sequence[Sequence[float]]
    final_yaw: float = SHELF_FACE_YAW
    pos_tol: float = 0.08
    turn_tol: float = 0.05
    in_place_yaw: float = 1.05
    drive_yaw: float = 0.35
    lookahead: float = 0.0
    max_lin: float = 0.30
    max_ang: float = 0.55
    ang_gain: float = 2.2
    idx: int = 0
    mode: str = "turn"
    lock_idx: int = -1
    waypoints: List[Tuple[float, float]] = field(init=False)
    brake_dist: float = field(init=False)
    _prev_yaw_err: float = field(init=False, default=0.0)
    _have_yaw_err: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.waypoints = [(float(x), float(y)) for x, y in self.route]
        self.final_yaw = float(self.final_yaw)
        self.max_lin = float(self.max_lin)
        self.max_ang = float(self.max_ang)
        self.brake_dist = max(0.70, min(1.35, 0.50 * abs(self.max_lin)))
        # Small heading error: steer while driving. Large error: stop and turn.
        if abs(self.max_lin) > 1.0:
            self.in_place_yaw = min(self.in_place_yaw, 0.50)
            self.drive_yaw = min(self.drive_yaw, 0.30)
            self.ang_gain = min(self.ang_gain, 1.45)
        if self.lookahead <= 0.0:
            self.lookahead = min(1.40, max(0.80, 0.55 * abs(self.max_lin)))

    def _outgoing_cross_track(self, x: float, y: float) -> float:
        if self.idx + 1 >= len(self.waypoints):
            return 0.0
        tx, ty = self.waypoints[self.idx]
        nx, ny = self.waypoints[self.idx + 1]
        vx, vy = nx - tx, ny - ty
        vlen = math.hypot(vx, vy)
        if vlen <= 1e-6:
            return math.hypot(float(x) - tx, float(y) - ty)
        return abs((float(x) - tx) * vy - (float(y) - ty) * vx) / vlen

    def _along_track_passed(self, x: float, y: float) -> bool:
        if self.idx + 1 >= len(self.waypoints):
            return False
        tx, ty = self.waypoints[self.idx]
        nx, ny = self.waypoints[self.idx + 1]
        vx, vy = nx - tx, ny - ty
        vlen = math.hypot(vx, vy)
        if vlen <= 1e-6:
            return False
        along = ((float(x) - tx) * vx + (float(y) - ty) * vy) / vlen
        if self._blocks_early_southbound(x, y) or self._blocks_early_westbound(x, y):
            return False
        if _segment_crosses_divider(x, y, nx, ny):
            return False
        if self._cuts_back_east(x, y, nx, ny):
            return False
        return along > max(0.12, self.pos_tol) and self._outgoing_cross_track(x, y) <= 0.20

    def _cuts_back_east(self, x: float, y: float, nx: float, ny: float) -> bool:
        """True if the next point would drive back through the west-corridor boxes."""
        if not self.waypoints or self.waypoints[-1][0] >= 0.0:
            return False
        if float(x) >= 0.15:
            return False
        if not (SOUTH_PEEL_Y - 0.20 < float(y) < 2.45):
            return False
        return float(nx) > float(x) + 0.25

    def _blocks_early_southbound(self, x: float, y: float) -> bool:
        """South from east of the hug rail clips the divider's north-west tip."""
        if self.idx + 1 >= len(self.waypoints):
            return False
        _nx, ny = self.waypoints[self.idx + 1]
        if ny >= 1.70:
            return False
        return float(x) > HUG_WEST_X + 0.08

    def _blocks_early_westbound(self, x: float, y: float) -> bool:
        """West from south of the yellow lane drives into the divider."""
        if self.idx + 1 >= len(self.waypoints):
            return False
        nx, ny = self.waypoints[self.idx + 1]
        if nx >= float(x) - 0.05 or ny < 1.70:
            return False
        return float(y) < WEST_LANE_Y - 0.08

    def _skip_near_waypoints(self, x: float, y: float) -> None:
        """Drop short hops and corners already passed along the route."""
        while self.idx + 1 < len(self.waypoints):
            tx, ty = self.waypoints[self.idx]
            dist = math.hypot(tx - float(x), ty - float(y))
            if dist <= max(0.40, self.pos_tol + 0.20):
                if self.idx == self.lock_idx and dist > self.pos_tol:
                    break
                if self._blocks_early_southbound(x, y) or self._blocks_early_westbound(x, y):
                    break
                nx, ny = self.waypoints[self.idx + 1]
                if _segment_crosses_divider(x, y, nx, ny):
                    break
                if self._cuts_back_east(x, y, nx, ny):
                    self.rebase_west_south(x, y)
                    break
                self.idx += 1
                continue
            if self._along_track_passed(x, y):
                self.idx += 1
                continue
            break
        if self.lock_idx >= 0 and self.idx > self.lock_idx:
            self.lock_idx = -1

    def _passed_current(self, x: float, y: float, yaw: float) -> bool:
        if self._blocks_early_southbound(x, y) or self._blocks_early_westbound(x, y):
            return False
        tx, ty = self.waypoints[self.idx]
        dx = tx - float(x)
        dy = ty - float(y)
        dist = math.hypot(dx, dy)
        if dist <= self.pos_tol:
            return True
        if self._along_track_passed(x, y):
            return True
        ahead = dx * math.cos(yaw) + dy * math.sin(yaw)
        return dist < max(0.30, self.pos_tol + 0.10) and ahead < -0.08

    def _aim_target(self, x: float, y: float) -> Tuple[float, float]:
        """Look-ahead point on the remaining polyline so corners become arcs."""
        tx, ty = self.waypoints[self.idx]
        dist = math.hypot(tx - float(x), ty - float(y))
        look = max(0.40, float(self.lookahead))
        if dist > look:
            scale = look / dist
            ax = float(x) + scale * (tx - float(x))
            ay = float(y) + scale * (ty - float(y))
        elif self._blocks_early_southbound(x, y) or self._blocks_early_westbound(x, y):
            ax, ay = tx, ty
        elif dist > max(0.22, self.pos_tol + 0.10) and not (
            self.idx + 1 < len(self.waypoints) and self.waypoints[self.idx + 1][1] < 1.70
            and float(x) <= HUG_WEST_X + 0.08
        ):
            ax, ay = tx, ty
        elif self.idx + 1 < len(self.waypoints):
            nx, ny = self.waypoints[self.idx + 1]
            if not _segment_crosses_divider(x, y, nx, ny):
                rest = look - dist
                along = math.hypot(nx - tx, ny - ty)
                if along > 1e-6:
                    scale = min(1.0, rest / along)
                    ax = tx + scale * (nx - tx)
                    ay = ty + scale * (ny - ty)
                else:
                    ax, ay = tx, ty
            else:
                ax, ay = tx, ty
        else:
            ax, ay = tx, ty
        return ax, ay

    def _path_yaw_and_cross(self, x: float, y: float) -> Tuple[float, float]:
        """Segment heading and left-positive cross-track to the current route leg."""
        tx, ty = self.waypoints[self.idx]
        if self.idx > 0:
            ox, oy = self.waypoints[self.idx - 1]
            path_yaw = math.atan2(ty - oy, tx - ox)
        else:
            path_yaw = math.atan2(ty - float(y), tx - float(x))
            return path_yaw, 0.0
        ux, uy = math.cos(path_yaw), math.sin(path_yaw)
        along = ux * (float(x) - ox) + uy * (float(y) - oy)
        seg_len = math.hypot(tx - ox, ty - oy)
        if along > seg_len - 0.05 and math.hypot(tx - float(x), ty - float(y)) > 0.08:
            # Past this waypoint: aim back at it instead of holding the old heading.
            path_yaw = math.atan2(ty - float(y), tx - float(x))
            return path_yaw, 0.0
        cross = ux * (float(y) - oy) - uy * (float(x) - ox)
        return path_yaw, cross

    def _aim_error(self, x: float, y: float, yaw: float) -> Tuple[float, float, float, bool]:
        """Route heading error, plus a small Stanley steer for cross-track.

        ``route_err`` is path_yaw − yaw (realign with the prescribed route).
        ``steer_err`` adds a bounded cross-track offset used only while driving.
        """
        tx, ty = self.waypoints[self.idx]
        dist = math.hypot(tx - float(x), ty - float(y))
        path_yaw, cross = self._path_yaw_and_cross(x, y)
        route_err = wrap_to_pi(path_yaw - float(yaw))
        steer = _clip(math.atan(1.2 * cross), 0.12 if abs(self.max_lin) > 1.0 else 0.22)
        steer_err = wrap_to_pi(path_yaw - steer - float(yaw))
        corner = False
        if self.idx + 1 < len(self.waypoints) and dist < max(1.60, self.brake_dist + 0.4):
            nx, ny = self.waypoints[self.idx + 1]
            out_yaw = math.atan2(ny - ty, nx - tx)
            in_yaw = math.atan2(ty - float(y), tx - float(x))
            if abs(wrap_to_pi(out_yaw - in_yaw)) > 0.55:
                corner = True
        return dist, route_err, steer_err, corner

    def _linear_for_heading(self, yaw_err: float, dist: float, corner: bool) -> float:
        """Forward speed only when already close to the route heading."""
        err = abs(float(yaw_err))
        if err > self.in_place_yaw or math.cos(float(yaw_err)) <= 0.20:
            return 0.0
        brake = max(self.brake_dist, 1.60) if corner else self.brake_dist
        dist_scale = min(1.0, float(dist) / brake)
        heading = max(0.0, math.cos(float(yaw_err))) ** 2
        if abs(self.max_lin) > 1.0:
            heading *= max(0.50, 1.0 - err / 0.55)
        return self.max_lin * heading * dist_scale

    def _heading_command(self, yaw_err: float) -> float:
        if abs(yaw_err) < self.turn_tol:
            self._prev_yaw_err = float(yaw_err)
            self._have_yaw_err = True
            return 0.0
        if self._have_yaw_err:
            d_err = wrap_to_pi(float(yaw_err) - self._prev_yaw_err)
        else:
            d_err = 0.0
            self._have_yaw_err = True
        self._prev_yaw_err = float(yaw_err)
        kd = 0.40 if abs(self.max_lin) > 1.0 else 0.15
        return _clip(self.ang_gain * float(yaw_err) + kd * d_err, self.max_ang)

    def _corridor_guard(
        self, x: float, y: float, yaw: float, linear: float, angular: float
    ) -> Tuple[float, float]:
        if in_east_shelf_stub(x, y):
            west_err = wrap_to_pi(math.pi - float(yaw))
            if math.sin(float(yaw)) > 0.25:
                return -min(0.55, 0.28 * self.max_lin), _clip(self.ang_gain * west_err, self.max_ang)
            return 0.0, _clip(self.ang_gain * west_err, self.max_ang)
        if in_south_east_stub(x, y):
            west_err = wrap_to_pi(math.pi - float(yaw))
            if math.sin(float(yaw)) < -0.25:
                return -min(0.55, 0.28 * self.max_lin), _clip(self.ang_gain * west_err, self.max_ang)
            return 0.0, _clip(self.ang_gain * west_err, self.max_ang)
        if in_north_racks(x, y):
            south_err = wrap_to_pi(-math.pi / 2.0 - float(yaw))
            if math.sin(float(yaw)) > 0.12:
                return -min(0.50, 0.28 * self.max_lin), _clip(self.ang_gain * south_err, self.max_ang)
            if abs(south_err) > 0.25:
                return 0.0, _clip(self.ang_gain * south_err, self.max_ang)
            return min(self.max_lin, 0.45), _clip(self.ang_gain * south_err, self.max_ang)
        goal_x = self.waypoints[-1][0] if self.waypoints else x
        if (
            goal_x < 0.0
            and 0.12 < float(x) < 0.55
            and 1.70 < float(y) < WEST_LANE_Y + 0.25
        ):
            west_err = wrap_to_pi(math.pi - float(yaw))
            if abs(west_err) > 0.70:
                return 0.0, _clip(self.ang_gain * west_err, self.max_ang)
            cruise = min(self.max_lin, 1.60 if abs(west_err) < 0.35 else 1.00)
            return cruise, _clip(self.ang_gain * west_err, self.max_ang)
        if near_divider_nw_corner(x, y) and goal_x < 0.0 and float(x) > HUG_WEST_X + 0.10:
            if self.lock_idx >= 0:
                west_err = wrap_to_pi(math.pi - float(yaw))
                return 0.0, _clip(self.ang_gain * west_err, self.max_ang)
            west_err = wrap_to_pi(math.pi - float(yaw))
            south_of_gap = float(y) < 1.88
            facing_south = math.sin(float(yaw)) < -0.20
            if south_of_gap and facing_south:
                return -min(0.50, 0.28 * self.max_lin), _clip(self.ang_gain * west_err, self.max_ang)
            if south_of_gap and abs(west_err) > 0.50:
                return 0.0, _clip(self.ang_gain * west_err, self.max_ang)
            if abs(west_err) > 0.25:
                return 0.0, _clip(self.ang_gain * west_err, self.max_ang)
            return min(self.max_lin, 0.70), _clip(self.ang_gain * west_err, self.max_ang)
        if (
            goal_x < 0.0
            and float(x) < 0.15
            and SOUTH_PEEL_Y + 0.20 < float(y) < WEST_LANE_Y - 0.10
        ):
            if float(x) > HUG_WEST_X + 0.04:
                west_err = wrap_to_pi(math.pi - float(yaw))
                if abs(west_err) > 0.28:
                    return 0.0, _clip(self.ang_gain * west_err, self.max_ang)
                return max(0.45, min(0.70, linear)), _clip(self.ang_gain * west_err, self.max_ang)
            on_rail = HUG_WEST_X - 0.18 <= float(x) <= HUG_WEST_X + 0.04
            target_x = self.waypoints[self.idx][0] if self.idx < len(self.waypoints) else float(x)
            if on_rail and abs(target_x - HUG_WEST_X) < 0.20:
                south_err = wrap_to_pi(-math.pi / 2.0 - float(yaw))
                angular = self._heading_command(south_err)
                if abs(south_err) > 0.28:
                    return 0.0, angular
                if linear > 0.0:
                    linear = min(linear, 1.20)
                return linear, angular
        if in_center_wall_band(x, y):
            # Do not westbound through the slab. Near the north tip, climb onto
            # the yellow aisle; further south, leave toward the east corridor.
            if float(y) >= SHELF_CORNER_Y - 0.25:
                face = math.pi / 2.0
            else:
                face = 0.0
            face_err = wrap_to_pi(face - float(yaw))
            if abs(face_err) > 0.35:
                return 0.0, _clip(self.ang_gain * face_err, self.max_ang)
            return min(self.max_lin, max(0.50, 0.30 * self.max_lin)), _clip(
                self.ang_gain * face_err, self.max_ang
            )
        return linear, angular

    def step(self, x: float, y: float, yaw: float) -> Tuple[float, float, bool]:
        if self.idx < len(self.waypoints):
            self._skip_near_waypoints(x, y)
            if self.idx >= len(self.waypoints):
                yaw_err = wrap_to_pi(self.final_yaw - float(yaw))
                if abs(yaw_err) < self.turn_tol:
                    return 0.0, 0.0, True
                return 0.0, _clip(self.ang_gain * yaw_err, self.max_ang), False
            if self._passed_current(x, y, yaw):
                self.idx += 1
                if self.idx >= len(self.waypoints):
                    yaw_err = wrap_to_pi(self.final_yaw - float(yaw))
                    if abs(yaw_err) < self.turn_tol:
                        return 0.0, 0.0, True
                    return 0.0, _clip(self.ang_gain * yaw_err, self.max_ang), False
            dist, route_err, steer_err, corner = self._aim_error(x, y, yaw)
            if self.mode == "turn":
                if abs(route_err) <= self.drive_yaw:
                    self.mode = "drive"
                    self._have_yaw_err = False
            elif abs(route_err) > self.in_place_yaw:
                self.mode = "turn"
                self._have_yaw_err = False
            if self.mode == "turn":
                # Stop and face the route heading, not a Stanley inboard bias.
                linear = 0.0
                angular = self._heading_command(route_err)
            else:
                linear = self._linear_for_heading(route_err, dist, corner)
                angular = self._heading_command(steer_err)
            linear, angular = self._corridor_guard(x, y, yaw, linear, angular)
            return linear, angular, False

        yaw_err = wrap_to_pi(self.final_yaw - float(yaw))
        if abs(yaw_err) < self.turn_tol:
            return 0.0, 0.0, True
        return 0.0, _clip(self.ang_gain * yaw_err, self.max_ang), False

    def insert_ahead(self, xy: Sequence[float]) -> None:
        """Splice a detour in front of the current waypoint."""
        point = (float(xy[0]), float(xy[1]))
        if self.idx >= len(self.waypoints):
            return
        if self.idx < len(self.waypoints):
            current = self.waypoints[self.idx]
            if math.hypot(point[0] - current[0], point[1] - current[1]) < 0.08:
                return
        self.waypoints.insert(self.idx, point)
        self.lock_idx = self.idx

    def rebase_west_south(self, x: float, y: float) -> None:
        """After going west around boxes, stay on that x and go south.

        The canned peel is (HUG_WEST_X, SOUTH_PEEL_Y). Skipping back to it
        from x≈-1 drives east through the same boxes we just went around.
        """
        if not self.waypoints or self.waypoints[-1][0] >= 0.0:
            return
        if float(x) >= 0.15:
            return
        if float(y) <= SOUTH_PEEL_Y + 0.12:
            return
        goal = self.waypoints[-1]
        lane_x = min(float(x), HUG_WEST_X)
        if self.idx < len(self.waypoints):
            lane_x = min(lane_x, self.waypoints[self.idx][0])
        lane_x = min(max(lane_x, NAV_BOUNDS["x"][0] + 0.15), HUG_WEST_X)
        south = (lane_x, SOUTH_PEEL_Y)
        head = self.waypoints[: self.idx + 1] if self.idx < len(self.waypoints) else []
        rebuilt: List[Tuple[float, float]] = []
        for point in (*head, south, goal):
            if rebuilt and math.hypot(point[0] - rebuilt[-1][0], point[1] - rebuilt[-1][1]) < 0.08:
                continue
            rebuilt.append(point)
        if not rebuilt:
            rebuilt = [south, goal]
        self.waypoints = rebuilt
        self.idx = min(self.idx, max(0, len(self.waypoints) - 1))


def build_delivery_route(
    final_xy: Optional[Sequence[float]] = None,
    start_xy: Optional[Sequence[float]] = None,
) -> List[Tuple[float, float]]:
    """Hug the divider south, then peel to the table. Not via the outer box lane."""
    goal = tuple(float(v) for v in (final_xy if final_xy is not None else DELIVERY_APPROACH_XY))
    hug_north = (HUG_WEST_X, WEST_LANE_Y)
    hug_south = (HUG_WEST_X, SOUTH_PEEL_Y)
    east_lane = (SHELF_CORNER_XY[0], WEST_LANE_Y)
    sx = sy = None
    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])

    planned: List[Tuple[float, float]]
    if sx is not None and (in_center_wall_band(sx, sy) or (sx >= EAST_STUB_X_MIN and sy < WEST_LANE_Y - 0.20)):
        planned = []
        if in_center_wall_band(sx, sy):
            planned.append((east_lane[0], sy))
        planned.extend((east_lane, hug_north, hug_south, goal))
    elif sx is not None and sx <= 0.15:
        if sy >= WEST_LANE_Y - 0.20:
            planned = [hug_north, hug_south, goal]
        elif sy > SOUTH_PEEL_Y + 0.20:
            planned = [hug_south, goal]
        else:
            planned = [goal]
    else:
        planned = [hug_north, hug_south, goal]

    route: List[Tuple[float, float]] = []
    for point in planned:
        xy = (float(point[0]), float(point[1]))
        if route and math.hypot(xy[0] - route[-1][0], xy[1] - route[-1][1]) < 0.05:
            continue
        if sx is not None and math.hypot(xy[0] - sx, xy[1] - sy) < 0.18:
            continue
        route.append(xy)
    if not route:
        route.append((float(goal[0]), float(goal[1])))
    elif route[-1] != (float(goal[0]), float(goal[1])):
        route.append((float(goal[0]), float(goal[1])))
    return route
