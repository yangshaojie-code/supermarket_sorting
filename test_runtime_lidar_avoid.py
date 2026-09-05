import math
import unittest

from runtime.lidar_avoid import (
    CorridorFollower,
    blocked_span,
    clearer_side,
    clip_to_nav_bounds,
    detour_xy,
    pick_detour_side,
    scan_sectors,
)
from runtime.scene_zones import (
    DELIVERY_APPROACH_XY,
    DELIVERY_FACE_YAW,
    HUG_WEST_X,
    NAV_BOUNDS,
    SOUTH_CROSS_Y,
    SOUTH_PEEL_Y,
    SHELF_FACE_YAW,
    WEST_LANE_Y,
    in_delivery_base,
    in_picking_zone,
)
from runtime.waypoint_nav import (
    WaypointFollower,
    build_delivery_route,
    build_shelf_route,
    prune_passed_waypoints,
    wrap_to_pi,
)


def _scan(n=72, hit_lo=-0.40, hit_hi=0.40, hit=0.22, clear=8.0):
    increment = 2.0 * math.pi / n
    angle_min = -math.pi
    ranges = []
    for index in range(n):
        angle = wrap_to_pi(angle_min + index * increment)
        ranges.append(hit if hit_lo <= angle <= hit_hi else clear)
    return ranges, angle_min, increment


def _ray_circle(ox, oy, dx, dy, cx, cy, radius):
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return float("inf")
    root = math.sqrt(disc)
    hits = [t for t in ((-b - root) / 2.0, (-b + root) / 2.0) if t > 0.08]
    return min(hits) if hits else float("inf")


def _lidar_around(x, y, yaw, obstacles, n=72, max_range=8.0):
    increment = 2.0 * math.pi / n
    angle_min = -math.pi
    ranges = []
    for index in range(n):
        angle = yaw + angle_min + index * increment
        dx, dy = math.cos(angle), math.sin(angle)
        nearest = max_range
        for cx, cy, radius in obstacles:
            hit = _ray_circle(x, y, dx, dy, cx, cy, radius)
            nearest = min(nearest, hit)
        ranges.append(nearest if nearest < max_range else max_range)
    return ranges, angle_min, increment


class LidarAvoidTests(unittest.TestCase):
    def test_forward_sector_sees_only_the_front_hit(self):
        ranges, amin, inc = _scan()
        sectors = scan_sectors(ranges, amin, inc)
        self.assertLess(sectors.forward, 0.3)
        self.assertGreater(sectors.left, 7.0)
        self.assertGreater(sectors.right, 7.0)

    def test_side_box_steers_instead_of_detouring(self):
        corridor = CorridorFollower(
            WaypointFollower([(2.0, 0.0)], final_yaw=0.0, max_lin=2.4, max_ang=1.2),
        )
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit_lo=0.40, hit_hi=0.95, hit=0.45)
        linear, angular, done, status = corridor.step(
            0.0, 0.0, 0.0, ranges, amin, inc, dt=0.05,
        )
        self.assertFalse(done)
        self.assertGreater(linear, 0.4)
        self.assertLess(angular, 0.0)
        self.assertEqual(corridor.detours, 0)
        self.assertFalse(status.startswith("detour"))
        self.assertFalse(status.startswith("blocked"))

    def test_blocked_span_is_narrow_for_a_side_hit(self):
        ranges, amin, inc = _scan(hit_lo=0.40, hit_hi=0.90, hit=0.50)
        self.assertLess(blocked_span(ranges, amin, inc, horizon=1.2), 0.35)
        ranges, amin, inc = _scan(hit=0.50)
        self.assertGreater(blocked_span(ranges, amin, inc, horizon=1.2), 0.50)

    def test_clear_horizon_keeps_full_cruise(self):
        corridor = CorridorFollower(
            WaypointFollower([(4.0, 0.0)], final_yaw=0.0, max_lin=2.4, max_ang=1.2),
        )
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit=2.50)
        linear, _angular, done, status = corridor.step(
            0.0, 0.0, 0.0, ranges, amin, inc, dt=0.05,
        )
        self.assertFalse(done)
        self.assertGreater(linear, 2.0)
        self.assertEqual(status, "cruise")

    def test_wide_block_far_away_still_approaches_fast(self):
        corridor = CorridorFollower(
            WaypointFollower([(4.0, 0.0)], final_yaw=0.0, max_lin=2.4, max_ang=1.2),
        )
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit=0.90)
        linear, _angular, done, status = corridor.step(
            0.0, 0.0, 0.0, ranges, amin, inc, dt=0.05,
        )
        self.assertFalse(done)
        self.assertGreater(linear, 1.8)
        self.assertIn(status, ("cruise", "approach"))

    def test_wide_block_at_stop_distance_turns_in_place(self):
        corridor = CorridorFollower(
            WaypointFollower([(4.0, 0.0)], final_yaw=0.0, max_lin=2.4, max_ang=1.2),
            blocked_s=0.2,
        )
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit=0.48)
        linear, _angular, done, status = corridor.step(
            0.0, 0.0, 0.0, ranges, amin, inc, dt=0.05,
        )
        self.assertFalse(done)
        self.assertEqual(linear, 0.0)
        self.assertTrue(
            status.startswith("blocked") or status.startswith("detour") or status == "turn"
        )

    def test_clearer_side_prefers_the_open_half(self):
        ranges, amin, inc = _scan(hit_lo=-0.2, hit_hi=1.2, hit=0.3)
        sectors = scan_sectors(ranges, amin, inc)
        self.assertEqual(clearer_side(sectors), "right")

    def test_detour_stays_inside_nav_bounds(self):
        x, y = clip_to_nav_bounds(9.0, 9.0)
        self.assertLessEqual(x, NAV_BOUNDS["x"][1])
        self.assertLessEqual(y, NAV_BOUNDS["y"][1])
        left = detour_xy(0.0, 0.0, 0.0, "left", dist=0.7)
        self.assertGreater(left[1], 0.3)

    def test_left_corridor_detour_goes_west_not_into_the_wall(self):
        ranges, amin, inc = _scan()
        sectors = scan_sectors(ranges, amin, inc)
        side = pick_detour_side(-0.20, 0.50, -math.pi / 2.0, sectors, -1.94)
        self.assertEqual(side, "right")
        point = detour_xy(-0.20, 0.50, -math.pi / 2.0, side, dist=0.90)
        self.assertLess(point[0], -0.20)
        left_into_wall = detour_xy(-0.10, 0.60, -math.pi / 2.0, "left", dist=0.70)
        self.assertLessEqual(left_into_wall[0], HUG_WEST_X)

    def test_blocked_scan_inserts_a_detour_and_stops_forward(self):
        follower = CorridorFollower(
            WaypointFollower([(2.0, 0.0)], final_yaw=0.0, max_lin=0.30, max_ang=0.55),
            blocked_s=0.2,
        )
        follower.follower.mode = "drive"
        ranges, amin, inc = _scan()
        linear = 1.0
        status = "idle"
        for _ in range(8):
            linear, angular, done, status = follower.step(
                0.0, 0.0, 0.0, ranges, amin, inc, dt=0.1,
            )
            self.assertFalse(done)
            self.assertEqual(linear, 0.0)
            if status.startswith("detour"):
                break
        self.assertTrue(status.startswith("detour"))
        self.assertEqual(follower.detours, 1)
        self.assertGreater(len(follower.follower.waypoints), 1)
        follower.follower.mode = "drive"
        for _ in range(12):
            linear, angular, done, status = follower.step(
                0.0, 0.0, 0.0, ranges, amin, inc, dt=0.1,
            )
            self.assertFalse(done)
            self.assertLessEqual(linear, 0.0)
        self.assertEqual(follower.detours, 1)

    def test_stuck_against_a_hit_reverses_instead_of_standing_still(self):
        follower = CorridorFollower(
            WaypointFollower([(2.0, 0.0)], final_yaw=0.0, max_lin=0.30, max_ang=0.55),
            blocked_s=0.2,
        )
        follower.follower.mode = "drive"
        ranges, amin, inc = _scan()
        status = "idle"
        linear = 0.0
        for _ in range(40):
            linear, _angular, done, status = follower.step(
                0.0, 0.0, 0.0, ranges, amin, inc, dt=0.1,
            )
            self.assertFalse(done)
            if status == "reverse":
                break
        self.assertEqual(status, "reverse")
        self.assertLess(linear, 0.0)

    def test_turning_in_front_of_the_shelf_does_not_reverse(self):
        corridor = CorridorFollower(
            WaypointFollower(
                build_delivery_route(),
                final_yaw=DELIVERY_FACE_YAW,
                max_lin=2.4,
                max_ang=1.2,
            ),
            blocked_s=0.2,
        )
        ranges, amin, inc = _scan(hit=0.83)
        statuses = []
        for _ in range(25):
            _linear, _angular, done, status = corridor.step(
                0.82, 2.38, 1.35, ranges, amin, inc, dt=0.1,
            )
            self.assertFalse(done)
            statuses.append(status)
        self.assertNotIn("reverse", statuses)
        self.assertIn("turn", statuses)

    def test_creep_in_front_of_a_box_stops_instead_of_crawling(self):
        corridor = CorridorFollower(
            WaypointFollower(
                [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
                final_yaw=-math.pi / 2.0,
                max_lin=2.4,
                max_ang=1.2,
            ),
            blocked_s=0.2,
        )
        corridor.follower.idx = 1
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit=0.50)
        linear = 1.0
        status = "idle"
        for _ in range(12):
            linear, _angular, done, status = corridor.step(
                HUG_WEST_X, 1.20, -math.pi / 2.0, ranges, amin, inc, dt=0.1,
            )
            self.assertFalse(done)
        self.assertLessEqual(linear, 0.0)
        self.assertGreaterEqual(corridor.detours, 1)
        self.assertTrue(
            status == "turn"
            or status.startswith("blocked")
            or status.startswith("detour")
        )

    def test_west_corridor_box_detours_instead_of_creeping(self):
        """delivery_20260904_085809 crawled at fwd≈0.55 with 0 detours."""
        corridor = CorridorFollower(
            WaypointFollower(
                [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
                final_yaw=-math.pi / 2.0,
                max_lin=2.4,
                max_ang=1.2,
            ),
            blocked_s=0.2,
        )
        corridor.follower.idx = 1
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit=0.55)
        status = "idle"
        linear = 1.0
        for _ in range(20):
            linear, _angular, done, status = corridor.step(
                -0.38, -1.00, -math.pi / 2.0, ranges, amin, inc, dt=0.1,
            )
            self.assertFalse(done)
            if status.startswith("detour"):
                break
        self.assertTrue(status.startswith("detour"))
        self.assertEqual(linear, 0.0)
        self.assertEqual(corridor.detours, 1)
        detour = corridor.follower.waypoints[corridor.follower.idx]
        self.assertLess(detour[0], -0.38)
        self.assertNotIn((HUG_WEST_X, SOUTH_PEEL_Y), corridor.follower.waypoints)

    def test_west_facing_detour_turns_south_not_north(self):
        ranges, amin, inc = _scan()
        sectors = scan_sectors(ranges, amin, inc)
        side = pick_detour_side(-0.90, -1.00, math.pi, sectors, -1.94)
        self.assertEqual(side, "left")

    def test_same_pose_does_not_stack_detours(self):
        corridor = CorridorFollower(
            WaypointFollower(
                [(HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
                final_yaw=-math.pi / 2.0,
                max_lin=2.4,
                max_ang=1.2,
            ),
            blocked_s=0.2,
        )
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit=0.50)
        for _ in range(30):
            corridor.step(-0.38, -1.00, -math.pi / 2.0, ranges, amin, inc, dt=0.1)
        self.assertEqual(corridor.detours, 1)

    def test_jam_cone_clearing_while_spinning_still_reverses(self):
        """delivery_20260904_091630: fwd flickered 0.5↔1.6 and never left blocked."""
        corridor = CorridorFollower(
            WaypointFollower(
                [(HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
                final_yaw=-math.pi / 2.0,
                max_lin=2.4,
                max_ang=1.2,
            ),
            blocked_s=0.2,
        )
        corridor.follower.mode = "drive"
        corridor.follower.idx = 0
        statuses = []
        for index in range(45):
            ranges, amin, inc = _scan(hit=0.50 if index % 2 == 0 else 1.60)
            _linear, _angular, done, status = corridor.step(
                -0.79, 0.64, -math.pi / 2.0, ranges, amin, inc, dt=0.1,
            )
            self.assertFalse(done)
            statuses.append(status)
            if status == "reverse":
                break
        self.assertIn("reverse", statuses)
        self.assertLessEqual(corridor.detours, 2)

    def test_east_divider_on_cone_edge_does_not_cap_cruise(self):
        corridor = CorridorFollower(
            WaypointFollower(
                [(1.92, 2.32), (0.852, 2.32)],
                final_yaw=math.pi / 2.0,
                max_lin=2.4,
                max_ang=1.2,
            ),
        )
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit_lo=-0.45, hit_hi=-0.35, hit=1.45)
        linear, _angular, done, status = corridor.step(
            1.92, -3.17, math.pi / 2.0, ranges, amin, inc, dt=0.05,
        )
        self.assertFalse(done)
        self.assertEqual(status, "cruise")
        self.assertGreater(linear, 2.0)

    def test_eight_times_stop_and_slow_stay_aisle_scale(self):
        corridor = CorridorFollower(
            WaypointFollower([(2.0, 0.0)], final_yaw=0.0, max_lin=2.4, max_ang=1.2),
        )
        self.assertLessEqual(corridor.stop_dist, 0.55)
        self.assertLess(corridor.slow_dist, 1.30)

    def test_eight_times_speed_caps_before_a_nearby_box(self):
        corridor = CorridorFollower(
            WaypointFollower([(2.0, 0.0)], final_yaw=0.0, max_lin=2.4, max_ang=1.2),
        )
        corridor.follower.mode = "drive"
        ranges, amin, inc = _scan(hit=0.85)
        linear, _angular, done, _status = corridor.step(
            0.0, 0.0, 0.0, ranges, amin, inc, dt=0.05,
        )
        self.assertFalse(done)
        self.assertGreater(linear, 1.8)

    def test_delivery_route_hugs_the_divider_not_the_outer_box_lane(self):
        route = build_delivery_route()
        self.assertEqual(route[0], (HUG_WEST_X, WEST_LANE_Y))
        self.assertEqual(route[1], (HUG_WEST_X, SOUTH_PEEL_Y))
        self.assertEqual(route[-1], DELIVERY_APPROACH_XY)
        self.assertTrue(in_delivery_base(route[-1]))
        self.assertGreater(route[0][0], -1.0)
        self.assertNotIn((1.92, SOUTH_CROSS_Y), route)
        self.assertNotIn((-1.94, WEST_LANE_Y), route)

    def test_delivery_from_east_spawn_goes_north_first(self):
        route = build_delivery_route(start_xy=(1.92, -3.17))
        self.assertEqual(route[0], (1.92, WEST_LANE_Y))
        self.assertEqual(route[1], (HUG_WEST_X, WEST_LANE_Y))
        self.assertEqual(route[-1], DELIVERY_APPROACH_XY)
        self.assertNotIn((1.92, SOUTH_CROSS_Y), route)

    def test_prune_delivery_from_the_south_gap_does_not_return_to_the_shelf(self):
        route = build_delivery_route(start_xy=(0.96, -2.59))
        self.assertGreaterEqual(route[0][0], 1.35)
        self.assertEqual(route[-1], DELIVERY_APPROACH_XY)
        self.assertNotIn((1.92, SOUTH_CROSS_Y), route)
        pruned = prune_passed_waypoints(build_delivery_route(), 0.96, -2.59)
        self.assertGreaterEqual(pruned[0][0], 1.35)
        self.assertEqual(pruned[-1], DELIVERY_APPROACH_XY)

    def test_center_wall_band_turns_south_instead_of_driving_west(self):
        follower = WaypointFollower(
            build_delivery_route(),
            final_yaw=DELIVERY_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        follower.idx = 0
        linear, angular, done = follower.step(1.05, -2.80, math.pi)
        self.assertFalse(done)
        self.assertEqual(linear, 0.0)
        self.assertNotEqual(angular, 0.0)

    def test_unicycle_reaches_delivery_from_the_shelf(self):
        follower = WaypointFollower(
            build_delivery_route(),
            final_yaw=DELIVERY_FACE_YAW,
            pos_tol=0.12,
        )
        x, y, yaw = 0.865, 2.375, 1.428
        dt = 0.05
        done = False
        max_east_y = y
        west_of_aisle = False
        outer_lane_while_north = False
        max_x_south_of_gap = -9.0
        for _ in range(4000):
            linear, angular, done = follower.step(x, y, yaw)
            if done:
                break
            yaw = wrap_to_pi(yaw + angular * dt)
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
            if x > 1.35:
                max_east_y = max(max_east_y, y)
            if x < 0.20 and y > 1.70:
                west_of_aisle = True
            if y > 0.0 and x < -1.20:
                outer_lane_while_north = True
            if y < 1.85 and y > SOUTH_PEEL_Y:
                max_x_south_of_gap = max(max_x_south_of_gap, x)
        self.assertTrue(done)
        self.assertTrue(in_delivery_base((x, y)))
        self.assertLess(max_east_y, 2.50)
        self.assertTrue(west_of_aisle)
        self.assertFalse(outer_lane_while_north)
        self.assertLess(max_x_south_of_gap, 0.05)

    def test_unicycle_goes_around_a_box_in_the_east_corridor(self):
        corridor = CorridorFollower(
            WaypointFollower(build_shelf_route(), final_yaw=SHELF_FACE_YAW, pos_tol=0.14),
            blocked_s=0.15,
            detour_m=0.85,
            max_detours=10,
        )
        x, y, yaw = 1.92, -3.17, math.pi / 2.0
        dt = 0.05
        done = False
        obstacle = (1.92, 0.20, 0.40)
        for _ in range(8000):
            ranges, amin, inc = _lidar_around(x, y, yaw, [obstacle])
            linear, angular, done, _status = corridor.step(
                x, y, yaw, ranges, amin, inc, dt=dt,
            )
            if done:
                break
            yaw = wrap_to_pi(yaw + angular * dt)
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
        self.assertTrue(done)
        self.assertTrue(in_picking_zone((x, y)))
        self.assertGreaterEqual(corridor.detours, 1)


if __name__ == "__main__":
    unittest.main()
