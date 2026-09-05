import math
import unittest

from runtime.grid_planner import (
    OccupancyGrid,
    GlobalGridPlanner,
    GridNavController,
    illegal_shortcut,
    plan_to_waypoints,
)
from runtime.lidar_avoid import safety_gate
from runtime.scene_zones import (
    DELIVERY_APPROACH_XY,
    HUG_WEST_X,
    SHELF_APPROACH_XY,
    WEST_LANE_Y,
    in_center_wall_band,
    in_delivery_base,
    in_north_racks,
    in_south_east_stub,
)
from runtime.waypoint_nav import wrap_to_pi


def _scan(n=72, hit_lo=-0.22, hit_hi=0.22, hit=0.30, clear=8.0):
    increment = 2.0 * math.pi / n
    angle_min = -math.pi
    ranges = []
    for index in range(n):
        angle = wrap_to_pi(angle_min + index * increment)
        ranges.append(hit if hit_lo <= angle <= hit_hi else clear)
    return ranges, angle_min, increment


class GridPlannerTests(unittest.TestCase):
    def test_static_layer_marks_solids_and_keeps_the_yellow_lane(self):
        grid = OccupancyGrid()
        self.assertTrue(grid.is_static(0.80, 0.00))
        self.assertTrue(in_center_wall_band(0.80, 0.00))
        self.assertTrue(grid.is_static(0.80, 2.60))
        self.assertTrue(in_north_racks(0.80, 2.60))
        self.assertTrue(grid.is_static(1.80, -3.60))
        self.assertTrue(in_south_east_stub(1.80, -3.60))
        self.assertFalse(grid.is_static(1.92, WEST_LANE_Y))
        self.assertFalse(grid.is_static(HUG_WEST_X, 0.50))
        self.assertFalse(grid.is_static(*SHELF_APPROACH_XY))
        self.assertFalse(grid.is_static(*DELIVERY_APPROACH_XY))

    def test_empty_map_plans_delivery_without_crossing_the_divider(self):
        planner = GlobalGridPlanner()
        route = planner.plan(1.92, -3.17, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(route)
        self.assertGreater(len(route), 2)
        self.assertLess(math.hypot(route[-1][0] - DELIVERY_APPROACH_XY[0], route[-1][1] - DELIVERY_APPROACH_XY[1]), 0.20)
        west_of_aisle = False
        for px, py in route:
            self.assertFalse(in_center_wall_band(px, py))
            if px < 0.10 and py > 1.70:
                west_of_aisle = True
        self.assertTrue(west_of_aisle)
        for index in range(len(route) - 1):
            self.assertFalse(illegal_shortcut(*route[index], *route[index + 1]))

    def test_empty_map_plans_shelf_along_the_yellow_lane(self):
        planner = GlobalGridPlanner()
        route = planner.plan(1.92, -3.17, SHELF_APPROACH_XY)
        self.assertIsNotNone(route)
        self.assertGreater(max(py for _px, py in route), 2.10)
        self.assertLess(min(abs(px - SHELF_APPROACH_XY[0]) for px, py in route[-3:]), 0.40)

    def test_a_box_on_the_hug_rail_does_not_force_the_outer_wall(self):
        planner = GlobalGridPlanner()
        planner.grid.mark_circle(-0.40, -0.20, 0.18, now=1.0)
        route = planner.plan(-0.40, 0.40, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(route)
        self.assertFalse(any(px < -1.70 for px, _py in route[:8]))

    def test_two_boxes_in_the_west_corridor_leave_the_hug_rail(self):
        planner = GlobalGridPlanner()
        planner.grid.mark_circle(-0.40, -0.80, 0.22, now=1.0)
        planner.grid.mark_circle(-1.20, -0.80, 0.22, now=1.0)
        route = planner.plan(-0.44, -0.84, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(route)
        self.assertTrue(any(px < -1.30 for px, _py in route) or any(py < -1.40 for _px, py in route))
        self.assertLess(route[-1][1], -2.40)

    def test_unknown_cells_do_not_make_the_start_unsolvable(self):
        planner = GlobalGridPlanner()
        route = planner.plan(1.92, -3.17, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(route)

    def test_plan_to_waypoints_keeps_spacing(self):
        points = [(0.0, 0.0), (0.10, 0.0), (0.20, 0.0), (1.20, 0.0)]
        waypoints = plan_to_waypoints(points)
        self.assertGreaterEqual(waypoints[0][0], -0.01)
        for index in range(len(waypoints) - 1):
            self.assertGreaterEqual(
                math.hypot(waypoints[index + 1][0] - waypoints[index][0], waypoints[index + 1][1] - waypoints[index][1]),
                0.08,
            )

    def test_safety_gate_stops_and_does_not_steer(self):
        ranges, amin, inc = _scan(hit=0.30)
        linear, angular, status = safety_gate(1.2, 0.4, ranges, amin, inc, stop_dist=0.43)
        self.assertEqual(linear, 0.0)
        self.assertEqual(angular, 0.4)
        self.assertEqual(status, "safety_stop")

    def test_controller_replans_instead_of_detouring(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        ranges, amin, inc = _scan(hit=8.0)
        linear, _angular, done, status = nav.step(
            1.92, -3.17, math.pi / 2.0, ranges, amin, inc, dt=0.05, now=0.6,
        )
        self.assertFalse(done)
        self.assertTrue(status.startswith("replan"))
        self.assertEqual(nav.plan_id, 1)
        self.assertEqual(linear, 0.0)
        self.assertIsNotNone(nav.follower)
        first = list(nav.follower.waypoints)
        nav.planner.grid.mark_circle(1.70, 0.20, 0.16, now=1.0)
        reroute = nav.planner.plan(1.92, -3.17, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(reroute)
        self.assertTrue(
            any(abs(px - 1.70) > 0.25 for px, py in reroute if abs(py - 0.20) < 0.35)
            or reroute != first
        )

    def test_delivery_route_ends_in_the_delivery_base(self):
        planner = GlobalGridPlanner()
        route = planner.plan(0.86, 2.32, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(route)
        self.assertTrue(in_delivery_base((route[-1][0], min(route[-1][1], -2.70))) or route[-1][1] <= -2.80)

    def test_divider_corner_is_not_a_static_wall(self):
        grid = OccupancyGrid()
        self.assertFalse(grid.is_static(-0.10, 2.10))
        self.assertFalse(grid.is_static(HUG_WEST_X, 2.00))
        self.assertTrue(in_center_wall_band(0.40, 1.60) or grid.is_static(0.40, 1.60))

    def test_rack_echo_does_not_block_the_yellow_lane(self):
        planner = GlobalGridPlanner()
        ranges, amin, inc = _scan(n=72, hit_lo=-0.30, hit_hi=0.30, hit=0.20, clear=8.0)
        for tick in range(6):
            planner.integrate_scan(0.90, 2.32, math.pi / 2.0, ranges, amin, inc, now=float(tick))
        col, row = planner.grid.world_to_cell(0.90, 2.32)
        self.assertFalse(planner.grid.is_dynamic(col, row))
        self.assertFalse(planner.grid.is_lethal(col, row))
        route = planner.plan(0.92, 2.39, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(route)
        self.assertFalse(planner.path_blocked_ahead(0.92, 2.39, route))

    def test_controller_holds_route_while_turning(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        ranges, amin, inc = _scan(hit=8.0)
        _lin, _ang, _done, status = nav.step(
            0.92, 2.39, 1.42, ranges, amin, inc, dt=0.05, now=0.6,
        )
        self.assertTrue(status.startswith("replan"))
        plan_id = nav.plan_id
        for step in range(24):
            nav.step(
                0.92,
                2.39,
                1.42 + 0.06 * step,
                ranges,
                amin,
                inc,
                dt=0.05,
                now=0.65 + 0.05 * step,
            )
        self.assertEqual(nav.plan_id, plan_id)

    def test_empty_plan_stays_under_budget(self):
        import time

        planner = GlobalGridPlanner()
        t0 = time.perf_counter()
        route = planner.plan(0.92, 2.39, DELIVERY_APPROACH_XY)
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        self.assertIsNotNone(route)
        self.assertLess(elapsed_ms, 120.0)

    def test_delivery_stays_on_yellow_until_the_hug_rail(self):
        planner = GlobalGridPlanner()
        route = planner.plan(0.92, 2.39, DELIVERY_APPROACH_XY)
        self.assertIsNotNone(route)
        self.assertTrue(illegal_shortcut(0.20, 2.30, -0.20, 1.70))
        for px, py in route:
            if px > HUG_WEST_X + 0.10:
                self.assertGreaterEqual(py, WEST_LANE_Y - 0.22)

    def test_face_stop_turns_toward_the_open_side(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        clear, amin, inc = _scan(hit=8.0)
        nav.step(-1.69, 0.40, -math.pi / 2.0, clear, amin, inc, dt=0.05, now=0.6)
        blocked, amin, inc = _scan(hit=0.30)
        angular = 0.0
        status = ""
        for step in range(8):
            _lin, angular, _done, status = nav.step(
                -1.69, 0.10, -1.06, blocked, amin, inc, dt=0.05, now=2.0 + 0.05 * step,
            )
            if abs(angular) > 0.05:
                break
        self.assertGreater(abs(angular), 0.05)
        self.assertIn(status, ("recover_scan", "face_turn"))

    def test_face_stop_does_not_spam_replans(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        clear, amin, inc = _scan(hit=8.0)
        nav.step(0.92, 2.39, 1.42, clear, amin, inc, dt=0.05, now=0.6)
        plan_id = nav.plan_id
        blocked, amin, inc = _scan(hit=0.30)
        for step in range(40):
            nav.step(-0.02, 1.66, -1.93, blocked, amin, inc, dt=0.05, now=2.0 + 0.05 * step)
        self.assertLess(nav.plan_id - plan_id, 4)

    def test_first_pass_from_the_shelf_stays_on_the_yellow_lane(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        ranges, amin, inc = _scan(hit=8.0)
        _lin, _ang, _done, status = nav.step(
            0.92, 2.39, math.pi, ranges, amin, inc, dt=0.05, now=0.6,
        )
        self.assertEqual(status, "replan corridor")
        self.assertEqual(nav._mode, "explore")
        self.assertIsNotNone(nav.follower)
        for px, py in nav.follower.waypoints:
            if px > HUG_WEST_X + 0.10:
                self.assertGreaterEqual(py, WEST_LANE_Y - 0.22)

    def test_empty_map_does_not_replan_while_the_lane_is_clear(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        ranges, amin, inc = _scan(hit=8.0)
        nav.step(0.92, 2.39, math.pi, ranges, amin, inc, dt=0.05, now=0.6)
        plan_id = nav.plan_id
        for step in range(8):
            nav.step(
                0.70 - 0.04 * step,
                2.36,
                math.pi,
                ranges,
                amin,
                inc,
                dt=0.05,
                now=2.2 + 0.05 * step,
            )
        self.assertEqual(nav.plan_id, plan_id)
        self.assertEqual(nav._mode, "explore")

    def test_small_far_hit_steers_while_moving(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        clear, amin, inc = _scan(hit=8.0)
        nav.step(-0.40, 1.20, -math.pi / 2.0, clear, amin, inc, dt=0.05, now=0.6)
        far, amin, inc = _scan(n=72, hit_lo=-0.10, hit_hi=0.10, hit=1.20, clear=8.0)
        linear, _angular, _done, status = nav.step(
            -0.40, 1.00, -math.pi / 2.0, far, amin, inc, dt=0.05, now=2.2,
        )
        self.assertGreater(linear, 0.04)
        self.assertIn(status, ("nudge", "approach", "cruise"))

    def test_wide_near_hit_stops_and_turns(self):
        nav = GridNavController(
            goal_xy=DELIVERY_APPROACH_XY,
            final_yaw=-math.pi / 2.0,
            max_lin=1.2,
            max_ang=1.2,
        )
        clear, amin, inc = _scan(hit=8.0)
        nav.step(-0.40, 1.20, -math.pi / 2.0, clear, amin, inc, dt=0.05, now=0.6)
        near, amin, inc = _scan(n=72, hit_lo=-0.40, hit_hi=0.40, hit=0.35, clear=8.0)
        linear, angular, _done, status = nav.step(
            -0.40, 0.90, -math.pi / 2.0, near, amin, inc, dt=0.05, now=2.2,
        )
        self.assertEqual(linear, 0.0)
        self.assertGreater(abs(angular), 0.05)
        self.assertIn(status, ("face_turn", "safety_stop", "recover_scan"))


if __name__ == "__main__":
    unittest.main()
