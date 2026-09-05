"""Static supermarket grid + lidar occupancy + A*. No ROS."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from runtime.lidar_avoid import (
    blocked_span,
    clear_horizon_m,
    occlusion_kind,
    pick_detour_side,
    safety_gate,
    scan_sectors,
)
from runtime.scene_zones import (
    CENTER_WALL_X,
    CENTER_WALL_Y,
    DELIVERY_APPROACH_XY,
    HUG_WEST_X,
    NAV_BOUNDS,
    NORTH_RACK_Y,
    ROUTE_DELIVERY_TO_SHELF,
    ROUTE_SHELF_TO_DELIVERY,
    SHELF_APPROACH_XY,
    SHELF_CORNER_XY,
    SOUTH_CROSS_Y,
    SOUTH_PEEL_Y,
    WEST_LANE_Y,
    in_center_wall_band,
    in_north_racks,
    in_south_east_stub,
)
from runtime.waypoint_nav import WaypointFollower, wrap_to_pi

GRID_RES = 0.10
STATIC_INFLATE_M = 0.30
INFLATE_M = 0.22
SELF_RANGE = 0.25
HIT_MIN = 0.25
HIT_MAX = 8.0
LETHAL_SCORE = 3
SCORE_MAX = 5
UNKNOWN_COST = 2.5
FREE_COST = 1.0
EDGE_COST = 0.15
CORRIDOR_COST = 0.35
TURN_COST = 0.15
CORRIDOR_WIDTH = 0.50
STOP_DIST = 0.43
MAP_READY_HITS = 8
WALL_ECHO_M = 0.16
PLAN_HOLD_S = 1.50
REPLAN_MIN_S = 0.50
FAIL_BACKOFF_S = 2.00
PATH_SKIP_M = 0.30
OFF_TRACK_M = 0.35
RELAXED_INFLATE_M = 0.18
PROGRESS_WINDOW_S = 1.50
PROGRESS_EPS_M = 0.05
YELLOW_SOUTH_Y = WEST_LANE_Y - 0.18
PEEK_S = 0.90


def _world_bounds() -> Tuple[float, float, float, float]:
    return (
        float(NAV_BOUNDS["x"][0]),
        float(NAV_BOUNDS["x"][1]),
        float(NAV_BOUNDS["y"][0]),
        float(NAV_BOUNDS["y"][1]),
    )


def _point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _polyline_distance(px: float, py: float, line: Sequence[Tuple[float, float]]) -> float:
    if not line:
        return 0.0
    if len(line) == 1:
        return math.hypot(px - line[0][0], py - line[0][1])
    nearest = float("inf")
    for index in range(len(line) - 1):
        nearest = min(nearest, _point_to_segment(px, py, *line[index], *line[index + 1]))
    return nearest


def preferred_corridor(goal_xy: Sequence[float]) -> List[Tuple[float, float]]:
    goal = (float(goal_xy[0]), float(goal_xy[1]))
    east = (SHELF_CORNER_XY[0], WEST_LANE_Y)
    line = [(east[0], -3.17), east]
    if goal[0] < 0.0:
        line.extend(ROUTE_SHELF_TO_DELIVERY)
    else:
        line.extend(ROUTE_DELIVERY_TO_SHELF)
        line.append(goal)
    cleaned: List[Tuple[float, float]] = []
    for point in line:
        if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) < 0.05:
            continue
        cleaned.append((float(point[0]), float(point[1])))
    return cleaned


def early_southbound(x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if the segment leaves the yellow lane south before the hug rail."""
    if min(float(x0), float(x1)) <= HUG_WEST_X + 0.08:
        return False
    return (float(y1) - float(y0)) < -0.04 and float(y1) < YELLOW_SOUTH_Y


def illegal_shortcut(x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if the segment cuts the slab or drops south before the hug rail."""
    if in_center_wall_band(x0, y0) or in_center_wall_band(x1, y1):
        return True
    if early_southbound(x0, y0, x1, y1):
        return True
    left, right = CENTER_WALL_X
    y_lo, y_hi = CENTER_WALL_Y
    if not ((y0 > y_hi and y1 > y_hi) or (y0 < y_lo and y1 < y_lo)):
        if not ((x0 < left and x1 < left) or (x0 > right and x1 > right)):
            if (x0 - left) * (x1 - left) <= 0.0 or (x0 - right) * (x1 - right) <= 0.0:
                return True
    westbound = (x1 - x0) < -0.15
    if westbound and min(y0, y1) < SOUTH_CROSS_Y + 0.15 and max(x0, x1) > left - 0.05:
        if min(x0, x1) < right + 0.05:
            return True
    samples = max(2, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.04)))
    for step in range(samples + 1):
        t = step / samples
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if early_southbound(x0, y0, x, y):
            return True
    return False


class OccupancyGrid:
    """0.10 m grid over NAV_BOUNDS. Static solids + accumulated lidar hits."""

    def __init__(self, resolution: float = GRID_RES, inflate: float = INFLATE_M) -> None:
        self.res = float(resolution)
        self.inflate = float(inflate)
        x0, x1, y0, y1 = _world_bounds()
        self.origin_x = x0
        self.origin_y = y0
        self.cols = int(math.ceil((x1 - x0) / self.res))
        self.rows = int(math.ceil((y1 - y0) / self.res))
        n = self.cols * self.rows
        self.static = bytearray(n)
        self.observed = bytearray(n)
        self.score = [0] * n
        self.last_hit = [0.0] * n
        self.confirmed = bytearray(n)
        self.lethal = bytearray(n)
        self.dynamic = bytearray(n)
        self._dynamic_inflate = self.inflate
        self._inflate_disk = self._disk_offsets(STATIC_INFLATE_M)
        self._echo_disk = self._disk_offsets(WALL_ECHO_M)
        self._paint_static()
        self.rebuild_masks()

    def _disk_offsets(self, radius_m: float) -> List[Tuple[int, int]]:
        span = max(1, int(math.ceil(float(radius_m) / self.res)))
        limit = float(radius_m) * float(radius_m) + 1e-6
        cells: List[Tuple[int, int]] = []
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                if (dr * dr + dc * dc) * self.res * self.res <= limit:
                    cells.append((dc, dr))
        return cells

    def _index(self, col: int, row: int) -> int:
        return row * self.cols + col

    def in_grid(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor((float(x) - self.origin_x) / self.res)),
            int(math.floor((float(y) - self.origin_y) / self.res)),
        )

    def cell_center(self, col: int, row: int) -> Tuple[float, float]:
        return (
            self.origin_x + (col + 0.5) * self.res,
            self.origin_y + (row + 0.5) * self.res,
        )

    def _paint_static(self) -> None:
        solids: List[int] = []
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = self.cell_center(col, row)
                south_gap = (
                    float(y) <= SOUTH_CROSS_Y + 0.10
                    and 0.05 <= float(x) <= 1.55
                )
                if (
                    in_center_wall_band(x, y)
                    or in_south_east_stub(x, y)
                    or in_north_racks(x, y)
                    or south_gap
                ):
                    idx = self._index(col, row)
                    self.static[idx] = 1
                    if in_center_wall_band(x, y) or in_south_east_stub(x, y):
                        solids.append(idx)
        for idx in solids:
            row, col = idx // self.cols, idx % self.cols
            for dc, dr in self._inflate_disk:
                ncol, nrow = col + dc, row + dr
                if not self.in_grid(ncol, nrow):
                    continue
                nx, ny = self.cell_center(ncol, nrow)
                if ny >= WEST_LANE_Y - 0.06 and ny <= WEST_LANE_Y + CORRIDOR_WIDTH:
                    continue
                if abs(nx - HUG_WEST_X) <= CORRIDOR_WIDTH * 0.55 and ny <= WEST_LANE_Y + 0.10:
                    continue
                if in_north_racks(nx, ny):
                    continue
                self.static[self._index(ncol, nrow)] = 1
        self._carve_static_corridor()
        for goal in (SHELF_APPROACH_XY, DELIVERY_APPROACH_XY):
            col, row = self.world_to_cell(*goal)
            if self.in_grid(col, row):
                self.static[self._index(col, row)] = 0

    def _carve_static_corridor(self) -> None:
        half = 0.5 * CORRIDOR_WIDTH
        for line in (preferred_corridor(DELIVERY_APPROACH_XY), preferred_corridor(SHELF_APPROACH_XY)):
            for index in range(len(line) - 1):
                ax, ay = line[index]
                bx, by = line[index + 1]
                length = max(self.res, math.hypot(bx - ax, by - ay))
                steps = int(math.ceil(length / (0.5 * self.res)))
                for step in range(steps + 1):
                    t = step / steps
                    px, py = ax + t * (bx - ax), ay + t * (by - ay)
                    lo_c, lo_r = self.world_to_cell(px - half, py - half)
                    hi_c, hi_r = self.world_to_cell(px + half, py + half)
                    for row in range(lo_r, hi_r + 1):
                        for col in range(lo_c, hi_c + 1):
                            if not self.in_grid(col, row):
                                continue
                            x, y = self.cell_center(col, row)
                            if (
                                in_center_wall_band(x, y)
                                or in_north_racks(x, y)
                                or in_south_east_stub(x, y)
                            ):
                                continue
                            south_gap = y <= SOUTH_CROSS_Y + 0.10 and 0.05 <= x <= 1.55
                            if south_gap:
                                continue
                            if math.hypot(x - px, y - py) <= half:
                                self.static[self._index(col, row)] = 0

    def clear_footprint(self, x: float, y: float, radius: float = SELF_RANGE) -> None:
        lo_c, lo_r = self.world_to_cell(x - radius, y - radius)
        hi_c, hi_r = self.world_to_cell(x + radius, y + radius)
        for row in range(lo_r, hi_r + 1):
            for col in range(lo_c, hi_c + 1):
                if not self.in_grid(col, row):
                    continue
                cx, cy = self.cell_center(col, row)
                if math.hypot(cx - x, cy - y) > radius:
                    continue
                idx = self._index(col, row)
                if self.static[idx]:
                    continue
                self.score[idx] = 0
                self.confirmed[idx] = 0
                self.observed[idx] = 1

    def integrate_scan(
        self,
        x: float,
        y: float,
        yaw: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        now: float,
    ) -> None:
        self.clear_footprint(x, y)
        increment = float(angle_increment) if angle_increment else (2.0 * math.pi / max(1, len(ranges)))
        hit_cells = set()
        free_cells = set()
        for index, raw in enumerate(ranges):
            distance = float(raw)
            if not math.isfinite(distance) or distance < HIT_MIN:
                continue
            # Max-range returns mean "no hit", not an 8 m box.
            if distance >= HIT_MAX - 1e-3:
                continue
            angle = wrap_to_pi(float(angle_min) + index * increment + float(yaw))
            hx = float(x) + distance * math.cos(angle)
            hy = float(y) + distance * math.sin(angle)
            self._trace_ray(x, y, hx, hy, hit_cells, free_cells)
        for idx in free_cells:
            if idx in hit_cells or self.static[idx]:
                continue
            self.score[idx] = max(0, self.score[idx] - 1)
            self.observed[idx] = 1
        for idx in hit_cells:
            if self.static[idx]:
                continue
            col, row = idx % self.cols, idx // self.cols
            if self._near_static(col, row):
                self.observed[idx] = 1
                continue
            self.score[idx] = min(SCORE_MAX, self.score[idx] + 2)
            self.last_hit[idx] = float(now)
            self.observed[idx] = 1
            if self.score[idx] >= LETHAL_SCORE:
                self.confirmed[idx] = 1
        self._expire(now)
        self.rebuild_masks()

    def _trace_ray(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        hit_cells: set,
        free_cells: set,
    ) -> None:
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            return
        steps = max(1, int(math.ceil(length / (0.6 * self.res))))
        last = None
        for step in range(steps + 1):
            t = step / steps
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            col, row = self.world_to_cell(px, py)
            if not self.in_grid(col, row) or (col, row) == last:
                continue
            last = (col, row)
            idx = self._index(col, row)
            if self.static[idx]:
                continue
            if step < steps:
                free_cells.add(idx)
            else:
                hit_cells.add(idx)

    def _near_static(self, col: int, row: int) -> bool:
        for dc, dr in self._echo_disk:
            ncol, nrow = col + dc, row + dr
            if self.in_grid(ncol, nrow) and self.static[self._index(ncol, nrow)]:
                return True
        return False

    def rebuild_masks(self, dynamic_inflate: Optional[float] = None) -> None:
        if dynamic_inflate is not None:
            self._dynamic_inflate = max(0.0, float(dynamic_inflate))
        dynamic_disk = self._disk_offsets(self._dynamic_inflate)
        n = self.cols * self.rows
        self.lethal = bytearray(self.static)
        self.dynamic = bytearray(n)
        for idx in range(n):
            if self.static[idx]:
                continue
            if not (self.confirmed[idx] or self.score[idx] >= LETHAL_SCORE):
                continue
            row, col = idx // self.cols, idx % self.cols
            for dc, dr in dynamic_disk:
                ncol, nrow = col + dc, row + dr
                if not self.in_grid(ncol, nrow):
                    continue
                nidx = self._index(ncol, nrow)
                self.lethal[nidx] = 1
                if not self.static[nidx]:
                    self.dynamic[nidx] = 1

    def _expire(self, now: float) -> None:
        for idx, score in enumerate(self.score):
            if self.static[idx] or score <= 0:
                continue
            age = float(now) - self.last_hit[idx]
            if not self.confirmed[idx] and age > 5.0:
                self.score[idx] = 0
            elif self.confirmed[idx] and age > 30.0 and score < LETHAL_SCORE:
                self.confirmed[idx] = 0

    def mark_circle(self, x: float, y: float, radius: float, now: float) -> None:
        """Test helper: confirm a box without a scan."""
        lo_c, lo_r = self.world_to_cell(x - radius, y - radius)
        hi_c, hi_r = self.world_to_cell(x + radius, y + radius)
        for row in range(lo_r, hi_r + 1):
            for col in range(lo_c, hi_c + 1):
                if not self.in_grid(col, row):
                    continue
                cx, cy = self.cell_center(col, row)
                if math.hypot(cx - x, cy - y) > radius:
                    continue
                idx = self._index(col, row)
                if self.static[idx]:
                    continue
                self.score[idx] = SCORE_MAX
                self.confirmed[idx] = 1
                self.observed[idx] = 1
                self.last_hit[idx] = float(now)
        self.rebuild_masks()

    def is_static(self, x: float, y: float) -> bool:
        col, row = self.world_to_cell(x, y)
        if not self.in_grid(col, row):
            return True
        return bool(self.static[self._index(col, row)])

    def is_lethal(self, col: int, row: int, inflate_dynamic: bool = True) -> bool:
        if not self.in_grid(col, row):
            return True
        idx = self._index(col, row)
        if inflate_dynamic:
            return bool(self.lethal[idx])
        if self.static[idx]:
            return True
        return bool(self.confirmed[idx] or self.score[idx] >= LETHAL_SCORE)

    def is_dynamic(self, col: int, row: int) -> bool:
        if not self.in_grid(col, row):
            return False
        return bool(self.dynamic[self._index(col, row)])

    def confirmed_count(self) -> int:
        return int(sum(self.confirmed))

    def is_observed_free(self, col: int, row: int) -> bool:
        if not self.in_grid(col, row):
            return False
        idx = self._index(col, row)
        return bool(self.observed[idx]) and not self.lethal[idx]


class GlobalGridPlanner:
    def __init__(self, grid: Optional[OccupancyGrid] = None) -> None:
        self.grid = grid if grid is not None else OccupancyGrid()

    def integrate_scan(self, *args, **kwargs) -> None:
        self.grid.integrate_scan(*args, **kwargs)

    def plan(
        self,
        x: float,
        y: float,
        goal_xy: Sequence[float],
        dynamic_inflate: Optional[float] = None,
    ) -> Optional[List[Tuple[float, float]]]:
        # Normal plans restore the 0.22 m dynamic footprint.  A recovery
        # plan may use the tighter 0.18 m footprint; that choice stays
        # active for later scan mask rebuilds.
        self.grid.rebuild_masks(
            self.grid.inflate if dynamic_inflate is None else dynamic_inflate
        )
        start = self._nearest_open(float(x), float(y), 0.40)
        goal = self._nearest_open(float(goal_xy[0]), float(goal_xy[1]), 0.80)
        if start is None or goal is None:
            return None
        corridor = preferred_corridor(goal_xy)
        cells = self._astar(start, goal, goal_xy, corridor)
        if not cells:
            return None
        points = [self.grid.cell_center(col, row) for col, row in cells]
        points.append((float(goal_xy[0]), float(goal_xy[1])))
        waypoints = plan_to_waypoints(points)
        if not self._path_legal(waypoints):
            waypoints = plan_to_waypoints(points, spacing=0.25, min_spacing=0.18)
        if not self._path_legal(waypoints):
            waypoints = plan_to_waypoints(points, spacing=0.20, min_spacing=0.12)
        if not self._path_legal(waypoints):
            return None
        return waypoints

    def _nearest_open(self, x: float, y: float, radius: float) -> Optional[Tuple[int, int]]:
        col0, row0 = self.grid.world_to_cell(x, y)
        if self.grid.in_grid(col0, row0) and not self.grid.is_lethal(col0, row0):
            return (col0, row0)
        best = None
        best_d = radius + 1.0
        span = max(1, int(math.ceil(radius / self.grid.res)))
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                col, row = col0 + dc, row0 + dr
                if not self.grid.in_grid(col, row) or self.grid.is_lethal(col, row):
                    continue
                cx, cy = self.grid.cell_center(col, row)
                dist = math.hypot(cx - x, cy - y)
                if dist < best_d:
                    best = (col, row)
                    best_d = dist
        return best

    def _cell_cost(self, col: int, row: int, corridor: Sequence[Tuple[float, float]]) -> float:
        idx = self.grid._index(col, row)
        base = FREE_COST if self.grid.observed[idx] else UNKNOWN_COST
        if self.grid.dynamic[idx] or self._touches_dynamic(col, row):
            base += EDGE_COST
        x, y = self.grid.cell_center(col, row)
        base += CORRIDOR_COST * _polyline_distance(x, y, corridor)
        return base

    def _touches_dynamic(self, col: int, row: int) -> bool:
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ncol, nrow = col + dc, row + dr
            if self.grid.in_grid(ncol, nrow) and self.grid.dynamic[self.grid._index(ncol, nrow)]:
                return True
        return False

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        goal_xy: Sequence[float],
        corridor: Sequence[Tuple[float, float]],
    ) -> Optional[List[Tuple[int, int]]]:
        moves = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        )
        gx, gy = self.grid.cell_center(*goal)

        def heuristic(col: int, row: int) -> float:
            x, y = self.grid.cell_center(col, row)
            return math.hypot(gx - x, gy - y)

        start_cost = 0.0
        open_heap = [(heuristic(*start), 0.0, start, None)]
        best = {start: 0.0}
        came: dict = {}
        seen = 0
        while open_heap and seen < 8000:
            _f, cost, node, parent = heapq.heappop(open_heap)
            if node in came:
                continue
            came[node] = parent
            seen += 1
            if node == goal:
                return self._reconstruct(came, node)
            col, row = node
            incoming = None
            if parent is not None:
                incoming = (col - parent[0], row - parent[1])
            for dc, dr, step in moves:
                ncol, nrow = col + dc, row + dr
                if self.grid.is_lethal(ncol, nrow):
                    continue
                if dc != 0 and dr != 0:
                    if self.grid.is_lethal(col + dc, row) or self.grid.is_lethal(col, row + dr):
                        continue
                nx, ny = self.grid.cell_center(ncol, nrow)
                if illegal_shortcut(*self.grid.cell_center(col, row), nx, ny):
                    continue
                turn = TURN_COST if incoming is not None and incoming != (dc, dr) else 0.0
                nxt = (ncol, nrow)
                new_cost = cost + step * self._cell_cost(ncol, nrow, corridor) + turn
                if new_cost >= best.get(nxt, 1e18):
                    continue
                best[nxt] = new_cost
                heapq.heappush(open_heap, (new_cost + heuristic(ncol, nrow), new_cost, nxt, node))
        return None

    def _reconstruct(self, came: dict, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [node]
        while came[node] is not None:
            node = came[node]
            path.append(node)
        path.reverse()
        return path

    def _path_legal(self, points: Sequence[Tuple[float, float]]) -> bool:
        return not self.path_hits_lethal(points)

    def path_blocked_ahead(
        self,
        x: float,
        y: float,
        waypoints: Sequence[Tuple[float, float]],
        horizon: float = 1.50,
        skip: float = PATH_SKIP_M,
    ) -> bool:
        """True if a confirmed box (not a static wall) sits on the next path."""
        if not waypoints:
            return False
        remaining = [(float(x), float(y)), *[(float(px), float(py)) for px, py in waypoints]]
        walked = 0.0
        for index in range(len(remaining) - 1):
            ax, ay = remaining[index]
            bx, by = remaining[index + 1]
            length = math.hypot(bx - ax, by - ay)
            if length < 1e-6:
                continue
            steps = max(1, int(math.ceil(length / self.grid.res)))
            for step in range(1, steps + 1):
                t = step / steps
                px, py = ax + t * (bx - ax), ay + t * (by - ay)
                walked += length / steps
                if walked < skip:
                    continue
                col, row = self.grid.world_to_cell(px, py)
                if self.grid.is_dynamic(col, row):
                    return True
                if walked >= horizon:
                    return False
        return False

    def path_hits_lethal(
        self,
        points: Sequence[Tuple[float, float]],
        skip_start: float = 0.0,
    ) -> bool:
        """Check every segment, including beyond the local safety horizon."""
        if len(points) < 2:
            return False
        walked = 0.0
        for ax, ay, bx, by in (
            (float(a[0]), float(a[1]), float(b[0]), float(b[1]))
            for a, b in zip(points, points[1:])
        ):
            if illegal_shortcut(ax, ay, bx, by):
                return True
            length = math.hypot(bx - ax, by - ay)
            if length < 1e-6:
                continue
            steps = max(1, int(math.ceil(length / (0.5 * self.grid.res))))
            for step in range(1, steps + 1):
                t = step / steps
                walked += length / steps
                if walked <= float(skip_start):
                    continue
                px = ax + t * (bx - ax)
                py = ay + t * (by - ay)
                col, row = self.grid.world_to_cell(px, py)
                if self.grid.is_lethal(col, row):
                    return True
        return False


def _nearest_index(points: Sequence[Tuple[float, float]], x: float, y: float) -> int:
    best_i = 0
    best_d = float("inf")
    for index, point in enumerate(points):
        dist = math.hypot(point[0] - x, point[1] - y)
        if dist < best_d:
            best_i = index
            best_d = dist
    return best_i


def plan_to_waypoints(
    points: Sequence[Tuple[float, float]],
    spacing: float = 0.45,
    min_spacing: float = 0.25,
) -> List[Tuple[float, float]]:
    if not points:
        return []
    simplified = [points[0]]
    for index in range(1, len(points) - 1):
        ax, ay = simplified[-1]
        bx, by = points[index]
        cx, cy = points[index + 1]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        if dot < 0.985 or illegal_shortcut(ax, ay, cx, cy):
            simplified.append(points[index])
    simplified.append(points[-1])
    sampled = [simplified[0]]
    for index in range(len(simplified) - 1):
        ax, ay = simplified[index]
        bx, by = simplified[index + 1]
        if illegal_shortcut(ax, ay, bx, by):
            i0 = _nearest_index(points, ax, ay)
            i1 = _nearest_index(points, bx, by)
            if i1 < i0:
                i0, i1 = i1, i0
            for pt in points[i0 + 1 : i1 + 1]:
                if math.hypot(pt[0] - sampled[-1][0], pt[1] - sampled[-1][1]) >= 0.05:
                    sampled.append((float(pt[0]), float(pt[1])))
            continue
        length = math.hypot(bx - ax, by - ay)
        if length < min_spacing and index < len(simplified) - 2:
            continue
        steps = max(1, int(math.floor(length / spacing)))
        for step in range(1, steps + 1):
            t = min(1.0, (step * spacing) / length)
            px, py = ax + t * (bx - ax), ay + t * (by - ay)
            if math.hypot(px - sampled[-1][0], py - sampled[-1][1]) >= min_spacing:
                sampled.append((px, py))
        if math.hypot(bx - sampled[-1][0], by - sampled[-1][1]) >= min_spacing:
            sampled.append((bx, by))
    if math.hypot(sampled[-1][0] - points[-1][0], sampled[-1][1] - points[-1][1]) > 0.08:
        sampled.append((float(points[-1][0]), float(points[-1][1])))
    cleaned: List[Tuple[float, float]] = []
    for point in sampled:
        if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) < 0.08:
            continue
        cleaned.append(point)
    return cleaned


@dataclass
class GridNavController:
    """Yellow-lane first pass, then A* once the lidar map is enough."""

    goal_xy: Tuple[float, float]
    final_yaw: float
    max_lin: float
    max_ang: float
    planner: GlobalGridPlanner = field(default_factory=GlobalGridPlanner)
    follower: Optional[WaypointFollower] = None
    plan_id: int = 0
    last_status: str = "idle"
    last_plan_s: float = -100.0
    last_eval_s: float = -100.0
    last_plan_reason: str = ""
    last_plan_ms: float = 0.0
    stop_dist: float = STOP_DIST
    _last_xy: Optional[Tuple[float, float]] = None
    _moved_t: float = 0.0
    _off_t: float = 0.0
    _face_hits: int = 0
    _fail_plans: int = 0
    _recover_left: float = 0.0
    _recover_yaw: float = 0.0
    _reverse_left: float = 0.0
    _scan_count: int = 0
    _started: bool = False
    _mode: str = "explore"
    _last_confirmed: int = 0

    def integrate(
        self,
        x: float,
        y: float,
        yaw: float,
        ranges: Optional[Sequence[float]],
        angle_min: float,
        angle_increment: float,
        now: float,
    ) -> None:
        if ranges is None:
            return
        self._scan_count += 1
        self.planner.integrate_scan(x, y, yaw, ranges, angle_min, angle_increment, now)

    def _set_route(self, route: Sequence[Tuple[float, float]], x: float, y: float) -> None:
        trimmed = [(float(px), float(py)) for px, py in route]
        while len(trimmed) >= 2:
            d0 = math.hypot(trimmed[0][0] - x, trimmed[0][1] - y)
            d1 = math.hypot(trimmed[1][0] - x, trimmed[1][1] - y)
            if d0 < 0.22 or d1 + 0.35 < d0:
                trimmed.pop(0)
            else:
                break
        if len(trimmed) < 1:
            trimmed = list(route)
        self.follower = WaypointFollower(
            list(trimmed),
            final_yaw=self.final_yaw,
            max_lin=self.max_lin,
            max_ang=self.max_ang,
            pos_tol=0.12,
            global_mode=True,
        )
        self.follower.mode = "turn"
        self.plan_id += 1
        self._off_t = 0.0
        self._moved_t = 0.0
        self._last_xy = (float(x), float(y))
        self._face_hits = 0

    def _fallback_route(self, x: float, y: float) -> Optional[List[Tuple[float, float]]]:
        from runtime.waypoint_nav import build_delivery_route, build_shelf_route, prune_passed_waypoints

        if self.goal_xy[0] < 0.0:
            seed = prune_passed_waypoints(build_delivery_route(self.goal_xy, start_xy=(x, y)), x, y)
        else:
            seed = prune_passed_waypoints(build_shelf_route(self.goal_xy), x, y)
        # A fallback is allowed only when the complete seed route is clear.
        # Checking a short local horizon used to emit the two-point
        # ``(-0.40,-2.70) -> delivery`` line straight through a known box.
        self.planner.grid.rebuild_masks(self.planner.grid.inflate)
        if self.planner.path_hits_lethal([(float(x), float(y)), *seed]):
            return None
        return seed

    def _corridor_seed(self, x: float, y: float) -> List[Tuple[float, float]]:
        from runtime.waypoint_nav import build_delivery_route, build_shelf_route, prune_passed_waypoints

        if self.goal_xy[0] < 0.0:
            seed = build_delivery_route(self.goal_xy, start_xy=(x, y))
        else:
            seed = build_shelf_route(self.goal_xy)
        return prune_passed_waypoints(seed, x, y) or list(seed)

    def _map_ready(self) -> bool:
        return self.planner.grid.confirmed_count() >= MAP_READY_HITS

    def _path_trusted(self, x: float, y: float, route: Sequence[Tuple[float, float]]) -> bool:
        if not route:
            return False
        prev = (float(x), float(y))
        for px, py in route:
            if illegal_shortcut(prev[0], prev[1], float(px), float(py)):
                return False
            prev = (float(px), float(py))
        # A* can chamfer hug by dropping south before x reaches the rail.
        if float(y) >= WEST_LANE_Y - 0.30:
            for px, py in route:
                if float(px) > HUG_WEST_X + 0.10 and float(py) < WEST_LANE_Y - 0.22:
                    return False
        return True

    def _shelf_wall_ahead(self, x: float, y: float, yaw: float) -> bool:
        """Racks on the yellow-lane corner are a wall, not a box."""
        if self.goal_xy[0] < 0.0 and float(x) >= 1.35 and float(y) >= 2.05 and math.sin(float(yaw)) > 0.12:
            return True
        return float(y) > 2.35 and math.sin(float(yaw)) > 0.12

    def _react(
        self,
        linear: float,
        angular: float,
        x: float,
        y: float,
        yaw: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> Tuple[float, float, str]:
        sectors = scan_sectors(ranges, angle_min, angle_increment)
        horizon = clear_horizon_m(self.stop_dist, self.max_lin)
        span = blocked_span(ranges, angle_min, angle_increment, horizon=horizon)
        kind = occlusion_kind(sectors, span, self.stop_dist, horizon)
        if self._shelf_wall_ahead(x, y, yaw):
            if abs(linear) > 0.04:
                return linear, angular, "cruise"
            return linear, angular, "turn" if abs(angular) > 0.05 else "hold"
        if kind == "block":
            side = pick_detour_side(x, y, yaw, sectors, self.goal_xy[0])
            turn = (1.0 if side == "left" else -1.0) * min(1.05, abs(self.max_ang))
            return 0.0, turn, "face_turn"
        if kind == "nudge":
            left_f = sectors.left_front if math.isfinite(sectors.left_front) else 8.0
            right_f = sectors.right_front if math.isfinite(sectors.right_front) else 8.0
            dodge = 0.0
            if left_f < 0.70 and left_f + 0.12 < right_f:
                dodge = -0.35
            elif right_f < 0.70 and right_f + 0.12 < left_f:
                dodge = 0.35
            if dodge != 0.0 and abs(linear) > 0.04:
                limit = abs(self.max_ang)
                return linear, max(-limit, min(limit, angular + dodge)), "nudge"
        if kind == "approach" and linear > 0.04:
            ahead = sectors.ahead if math.isfinite(sectors.ahead) else 8.0
            horizon_m = max(0.0, ahead - self.stop_dist)
            linear = min(linear, horizon_m / 0.25)
            return linear, angular, "approach"
        if abs(linear) <= 0.04:
            return linear, angular, "turn" if abs(angular) > 0.05 else "hold"
        return linear, angular, "cruise" if kind == "clear" else kind

    def _cross_track(self, x: float, y: float) -> float:
        if self.follower is None or not self.follower.waypoints:
            return 0.0
        remain = list(self.follower.waypoints[self.follower.idx :])
        if not remain:
            return 0.0
        if self.follower.idx > 0:
            remain.insert(0, self.follower.waypoints[self.follower.idx - 1])
        else:
            remain.insert(0, (float(x), float(y)))
        nearest = float("inf")
        for index in range(len(remain) - 1):
            nearest = min(nearest, _point_to_segment(x, y, *remain[index], *remain[index + 1]))
        return nearest

    def _same_route(self, route: Sequence[Tuple[float, float]]) -> bool:
        if self.follower is None or not route:
            return False
        current = self.follower.waypoints[self.follower.idx :]
        if not current:
            return False
        count = min(3, len(route), len(current))
        for index in range(count):
            if math.hypot(route[index][0] - current[index][0], route[index][1] - current[index][1]) > 0.22:
                return False
        return True

    def _side_clearance(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
    ) -> Tuple[float, float]:
        sectors = scan_sectors(ranges, angle_min, angle_increment)
        left = min(
            sectors.left if math.isfinite(sectors.left) else 8.0,
            sectors.left_front if math.isfinite(sectors.left_front) else 8.0,
        )
        right = min(
            sectors.right if math.isfinite(sectors.right) else 8.0,
            sectors.right_front if math.isfinite(sectors.right_front) else 8.0,
        )
        return left, right

    def _begin_peek(
        self,
        yaw: float,
        ranges: Optional[Sequence[float]],
        angle_min: float,
        angle_increment: float,
    ) -> bool:
        """Turn toward a clear side so the next plan can see a gap. No detour points."""
        if ranges is None:
            return False
        left, right = self._side_clearance(ranges, angle_min, angle_increment)
        need = self.stop_dist + 0.12
        if left < need and right < need:
            return False
        if left >= right + 0.10:
            delta = 0.55
        elif right >= left + 0.10:
            delta = -0.55
        else:
            delta = 0.55
        self._recover_left = PEEK_S
        self._recover_yaw = wrap_to_pi(float(yaw) + delta)
        self.last_status = "recover_scan"
        return True

    def _route_still_faces_hit(
        self,
        x: float,
        y: float,
        yaw: float,
        route: Sequence[Tuple[float, float]],
        ahead: float,
    ) -> bool:
        if not route or not math.isfinite(ahead) or ahead > self.stop_dist + 0.08:
            return False
        return self.planner.path_blocked_ahead(x, y, route, horizon=0.55, skip=0.12)

    def _need_replan(self, x: float, y: float, yaw: float, ahead: float, now: float, turning: bool) -> str:
        if not self._started or self.follower is None:
            return "start"
        holding = (now - self.last_plan_s) < PLAN_HOLD_S
        remain = self.follower.waypoints[self.follower.idx :]
        blocked = self.planner.path_blocked_ahead(x, y, remain, horizon=1.50)
        if self._mode != "global":
            if not self._map_ready():
                return ""
            if holding:
                return ""
            if blocked:
                return "path_blocked"
            if math.isfinite(ahead) and ahead <= self.stop_dist and self._face_hits >= 3:
                return "face"
            return ""
        if math.isfinite(ahead) and ahead <= self.stop_dist and self._face_hits >= 3:
            if holding:
                return ""
            return "face"
        if holding or turning:
            return ""
        if blocked:
            return "path_blocked"
        if self._off_t >= 0.50:
            return "off_track"
        confirmed = self.planner.grid.confirmed_count()
        if self._moved_t >= PROGRESS_WINDOW_S and confirmed > self._last_confirmed:
            return "stuck"
        if now - self.last_eval_s >= 8.0 and self.planner.path_blocked_ahead(x, y, remain, horizon=2.50):
            return "stale"
        return ""

    def _note_progress(self, x: float, y: float, dt: float, turning: bool = False) -> None:
        """Measure displacement over a window, rather than one control tick."""
        here = (float(x), float(y))
        if self._last_xy is None:
            self._last_xy = here
            self._moved_t = 0.0
            return
        if turning:
            self._moved_t = 0.0
            self._off_t = 0.0
            self._last_xy = here
            return
        moved = math.hypot(here[0] - self._last_xy[0], here[1] - self._last_xy[1])
        if moved >= PROGRESS_EPS_M:
            self._moved_t = 0.0
            self._last_xy = here
        else:
            self._moved_t += float(dt)
        if self.follower is None or self.follower.idx >= len(self.follower.waypoints):
            self._off_t = 0.0
            return
        if self._cross_track(x, y) > OFF_TRACK_M:
            self._off_t += float(dt)
        else:
            self._off_t = 0.0

    def step(
        self,
        x: float,
        y: float,
        yaw: float,
        ranges: Optional[Sequence[float]],
        angle_min: float,
        angle_increment: float,
        dt: float,
        now: float,
    ) -> Tuple[float, float, bool, str]:
        self.integrate(x, y, yaw, ranges, angle_min, angle_increment, now)
        ahead = float("inf")
        if ranges is not None:
            sectors = scan_sectors(ranges, angle_min, angle_increment)
            ahead = sectors.ahead if math.isfinite(sectors.ahead) else float("inf")
            if ahead <= self.stop_dist:
                self._face_hits += 1
            else:
                self._face_hits = 0
        turning = bool(self.follower is not None and self.follower.mode == "turn")
        holding = self.last_plan_s >= 0.0 and now - self.last_plan_s < PLAN_HOLD_S
        self._note_progress(x, y, dt, turning=(turning or holding))

        if self._reverse_left > 0.0:
            self._reverse_left = max(0.0, self._reverse_left - dt)
            self.last_status = "recover_reverse"
            return -min(0.35, 0.20 * abs(self.max_lin)), 0.0, False, self.last_status
        if self._recover_left > 0.0:
            self._recover_left = max(0.0, self._recover_left - dt)
            err = wrap_to_pi(self._recover_yaw - yaw)
            self.last_status = "recover_scan"
            return 0.0, max(-self.max_ang, min(self.max_ang, 1.6 * err)), False, self.last_status

        reason = self._need_replan(x, y, yaw, ahead, now, turning)
        min_gap = FAIL_BACKOFF_S if self._fail_plans >= 3 else REPLAN_MIN_S
        can_search = self.last_plan_s < 0.0 or now - self.last_plan_s >= min_gap
        if reason == "start":
            seed = self._corridor_seed(x, y)
            self._set_route(seed, x, y)
            self._started = True
            self._mode = "explore"
            self.last_plan_s = now
            self.last_eval_s = now
            self.last_plan_reason = "corridor"
            self.last_status = "replan corridor"
            return 0.0, 0.0, False, self.last_status
        if reason and can_search:
            import time as _time

            t0 = _time.perf_counter()
            route = self.planner.plan(x, y, self.goal_xy)
            if route is None:
                route = self.planner.plan(
                    x, y, self.goal_xy, dynamic_inflate=RELAXED_INFLATE_M
                )
            self.last_plan_ms = 1000.0 * (_time.perf_counter() - t0)
            self.last_plan_s = now
            self.last_eval_s = now
            if route and not self._path_trusted(x, y, route):
                route = None
            if route and reason == "face" and (
                self._route_still_faces_hit(x, y, yaw, route, ahead) or self._same_route(route)
            ):
                route = None
            if route:
                self._set_route(route, x, y)
                self._started = True
                self._mode = "global"
                self._fail_plans = 0
                self._last_confirmed = self.planner.grid.confirmed_count()
                self.last_plan_reason = reason
                self.last_status = f"replan {reason}"
                return 0.0, 0.0, False, self.last_status
            if reason == "face" and self._begin_peek(yaw, ranges, angle_min, angle_increment):
                err = wrap_to_pi(self._recover_yaw - yaw)
                return 0.0, max(-self.max_ang, min(self.max_ang, 1.6 * err)), False, self.last_status
            self._fail_plans += 1
            self._mode = "explore"
            fallback = self._fallback_route(x, y) or self._corridor_seed(x, y)
            if fallback:
                self._set_route(fallback, x, y)
                self._started = True
                self._fail_plans = 0
                self.last_plan_reason = "fallback"
                self.last_status = "replan fallback"
                return 0.0, 0.0, False, self.last_status
            if self._fail_plans >= 3 and self._reverse_left <= 0.0:
                self._reverse_left = 0.35
            self.last_status = "blocked_no_path"
            return 0.0, 0.0, False, self.last_status

        if self.follower is None:
            self.last_status = "hold"
            return 0.0, 0.0, False, self.last_status
        linear, angular, done = self.follower.step(x, y, yaw)
        if ranges is not None:
            linear, angular, reacted = self._react(
                linear, angular, x, y, yaw, ranges, angle_min, angle_increment
            )
            linear, angular, gate = safety_gate(
                linear, angular, ranges, angle_min, angle_increment, stop_dist=self.stop_dist
            )
            if gate == "safety_stop":
                if reacted != "face_turn":
                    linear, angular, reacted = self._react(
                        0.0, angular, x, y, yaw, ranges, angle_min, angle_increment
                    )
                self.last_status = reacted if reacted == "face_turn" else gate
                return linear, angular, False, self.last_status
            if gate != "ok":
                self.last_status = gate
                return linear, angular, False, self.last_status
            self.last_eval_s = now
            if done:
                self.last_status = "arrived"
                return 0.0, 0.0, True, self.last_status
            self.last_status = reacted
            return linear, angular, False, self.last_status
        self.last_eval_s = now
        if done:
            self.last_status = "arrived"
            return 0.0, 0.0, True, self.last_status
        self.last_status = "turn" if abs(linear) <= 0.04 else "cruise"
        return linear, angular, False, self.last_status
