import math
import unittest
from types import SimpleNamespace

from runtime.scene_zones import (
    DELIVERY_APPROACH_XY,
    DELIVERY_FACE_YAW,
    HUG_WEST_X,
    SHELF_APPROACH_XY,
    SHELF_CORNER_XY,
    SHELF_FACE_YAW,
    SOUTH_PEEL_Y,
    WEST_LANE_Y,
    in_picking_zone,
)
from runtime.waypoint_nav import (
    WaypointFollower,
    build_shelf_route,
    min_forward_range,
    pose_from_odom,
    prune_passed_waypoints,
    wrap_to_pi,
    yaw_from_quaternion,
)


def fake_odom(x, y, yaw):
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(yaw / 2.0),
                    w=math.cos(yaw / 2.0),
                ),
            )
        )
    )


class WaypointNavTests(unittest.TestCase):
    def test_wrap_and_yaw_round_trip(self):
        self.assertAlmostEqual(wrap_to_pi(math.pi + 0.2), -math.pi + 0.2, places=6)
        yaw = math.pi / 2.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        self.assertAlmostEqual(yaw_from_quaternion(0.0, 0.0, qz, qw), yaw, places=6)
        x, y, recovered = pose_from_odom(fake_odom(-1.94, -3.17, yaw))
        self.assertAlmostEqual(x, -1.94)
        self.assertAlmostEqual(y, -3.17)
        self.assertAlmostEqual(recovered, yaw, places=6)

    def test_default_route_ends_in_picking_zone(self):
        route = build_shelf_route()
        self.assertGreaterEqual(len(route), 3)
        self.assertEqual(route[0], (SHELF_CORNER_XY[0], WEST_LANE_Y))
        self.assertEqual(route[1], (SHELF_APPROACH_XY[0], WEST_LANE_Y))
        self.assertEqual(route[-1], SHELF_APPROACH_XY)
        self.assertTrue(in_picking_zone(route[-1]))

    def test_prune_skips_aisle_entry_when_already_in_the_picking_zone(self):
        route = build_shelf_route()
        pruned = prune_passed_waypoints(route, 1.12, 2.08)
        self.assertEqual(pruned[0], SHELF_APPROACH_XY)
        self.assertEqual(pruned[-1], SHELF_APPROACH_XY)

    def test_prune_from_aisle_south_edge_still_goes_to_the_shelf(self):
        route = build_shelf_route()
        pruned = prune_passed_waypoints(route, 1.43, 1.72)
        self.assertEqual(pruned[-1], SHELF_APPROACH_XY)
        self.assertGreaterEqual(pruned[0][0], 1.35)
        self.assertGreaterEqual(pruned[0][1], WEST_LANE_Y - 0.15)

    def test_prune_keeps_aisle_entry_from_the_east_spawn(self):
        route = build_shelf_route()
        pruned = prune_passed_waypoints(route, 1.92, -3.17)
        self.assertEqual(pruned[0], (SHELF_CORNER_XY[0], WEST_LANE_Y))
        self.assertEqual(pruned[-1], SHELF_APPROACH_XY)

    def test_eight_times_brake_covers_the_corner(self):
        follower = WaypointFollower(
            [(1.92, 2.00)],
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
        )
        self.assertGreaterEqual(follower.brake_dist, 1.0)
        self.assertLessEqual(follower.brake_dist, 1.40)

    def test_insert_ahead_splices_a_detour_before_the_current_goal(self):
        follower = WaypointFollower([(1.92, 2.475)], final_yaw=SHELF_FACE_YAW)
        follower.insert_ahead((1.4, -1.0))
        self.assertEqual(follower.waypoints[0], (1.4, -1.0))
        self.assertEqual(follower.idx, 0)
        self.assertEqual(follower.lock_idx, 0)

    def test_west_detour_does_not_aim_back_at_the_hug_rail(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=DELIVERY_FACE_YAW,
            pos_tol=0.12,
        )
        follower.insert_ahead((-1.05, -1.30))
        follower.rebase_west_south(-0.82, -1.01)
        for px, _py in follower.waypoints[follower.idx : -1]:
            self.assertLessEqual(px, -0.70)
        self.assertEqual(follower.waypoints[-1], DELIVERY_APPROACH_XY)
        self.assertNotIn((HUG_WEST_X, SOUTH_PEEL_Y), follower.waypoints[follower.idx :])

    def test_skip_from_west_lane_does_not_return_east_through_boxes(self):
        follower = WaypointFollower(
            [
                (-1.00, -1.35),
                (-0.95, -1.50),
                (HUG_WEST_X, SOUTH_PEEL_Y),
                DELIVERY_APPROACH_XY,
            ],
            final_yaw=DELIVERY_FACE_YAW,
            pos_tol=0.12,
        )
        follower._skip_near_waypoints(-1.08, -1.71)
        tx, _ty = follower.waypoints[follower.idx]
        self.assertLess(tx, -0.70)

    def test_turns_in_place_when_facing_the_wrong_way(self):
        follower = WaypointFollower([(1.92, 2.475)], final_yaw=SHELF_FACE_YAW)
        linear, angular, done = follower.step(-1.94, -3.17, -math.pi / 2.0)
        self.assertFalse(done)
        self.assertEqual(linear, 0.0)
        self.assertGreater(angular, 0.0)

    def test_drives_forward_when_already_facing_the_waypoint(self):
        follower = WaypointFollower([(1.92, 2.475)], final_yaw=SHELF_FACE_YAW)
        follower.mode = "drive"
        linear, angular, done = follower.step(1.92, -3.17, math.pi / 2.0)
        self.assertFalse(done)
        self.assertGreater(linear, 0.0)
        self.assertEqual(angular, 0.0)

    def test_steers_while_driving_when_heading_is_only_off_a_bit(self):
        follower = WaypointFollower([(1.92, 2.475)], final_yaw=SHELF_FACE_YAW)
        follower.mode = "drive"
        linear, angular, done = follower.step(1.92, -3.17, math.pi / 2.0 - 0.35)
        self.assertFalse(done)
        self.assertGreater(linear, 0.0)
        self.assertGreater(angular, 0.0)

    def test_eight_times_speed_steers_while_driving_when_heading_is_only_off_a_bit(self):
        follower = WaypointFollower(
            [(1.92, 2.475)],
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        linear, angular, done = follower.step(1.92, -3.17, math.pi / 2.0 - 0.28)
        self.assertFalse(done)
        self.assertGreater(linear, 0.0)
        self.assertGreater(angular, 0.0)

    def test_southbound_off_the_rail_steers_back_onto_the_route(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.idx = 1
        follower.mode = "drive"
        _linear, angular, done = follower.step(-0.70, 1.20, -math.pi / 2.0)
        self.assertFalse(done)
        self.assertGreater(angular, 0.0)

    def test_southbound_small_heading_error_slows_instead_of_full_cruise(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.idx = 1
        follower.mode = "drive"
        linear, angular, done = follower.step(HUG_WEST_X, 1.20, -math.pi / 2.0 + 0.25)
        self.assertFalse(done)
        self.assertGreater(linear, 0.0)
        self.assertLess(linear, 1.6)
        self.assertLess(angular, 0.0)

    def test_southbound_large_heading_error_stops_to_realign_with_the_route(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.idx = 1
        follower.mode = "drive"
        linear, angular, done = follower.step(HUG_WEST_X, 1.20, -2.38)
        self.assertFalse(done)
        self.assertEqual(linear, 0.0)
        self.assertGreater(angular, 0.0)
        self.assertEqual(follower.mode, "turn")

    def test_eight_times_speed_turns_in_place_when_heading_is_halfway_off(self):
        follower = WaypointFollower(
            [(-1.94, 2.20)],
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        linear, angular, done = follower.step(0.96, 2.40, -2.24)
        self.assertFalse(done)
        self.assertEqual(linear, 0.0)
        self.assertNotEqual(angular, 0.0)

    def test_heading_hysteresis_keeps_turning_until_aligned(self):
        follower = WaypointFollower(
            [(-1.94, 2.20)],
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "turn"
        linear, angular, done = follower.step(0.96, 2.40, -2.50)
        self.assertFalse(done)
        self.assertEqual(linear, 0.0)
        self.assertEqual(follower.mode, "turn")
        self.assertNotEqual(angular, 0.0)

    def test_north_of_the_aisle_reverses_away_from_the_racks(self):
        follower = WaypointFollower(
            [(-1.94, 2.20), (-1.94, -2.90)],
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        linear, _angular, done = follower.step(0.25, 2.80, 2.36)
        self.assertFalse(done)
        self.assertLess(linear, 0.0)

    def test_skips_a_short_hop_and_turns_toward_the_useful_waypoint(self):
        follower = WaypointFollower(
            [(0.852, 2.00), (-1.94, 2.20), (-1.94, -2.90)],
            final_yaw=SHELF_FACE_YAW,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        linear, angular, done = follower.step(0.82, 2.38, 1.35)
        self.assertFalse(done)
        self.assertGreaterEqual(follower.idx, 1)
        self.assertNotEqual(angular, 0.0)

    def test_hug_corner_is_aimed_before_the_south_leg(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            max_lin=2.4,
            max_ang=1.2,
        )
        _ax, ay = follower._aim_target(0.20, WEST_LANE_Y)
        self.assertGreater(ay, WEST_LANE_Y - 0.05)

    def test_does_not_southbound_before_reaching_the_hug_rail(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        linear, _angular, done = follower.step(-0.06, 2.13, math.pi)
        self.assertFalse(done)
        self.assertEqual(follower.idx, 0)
        _ax, ay = follower._aim_target(-0.06, 2.13)
        self.assertGreater(ay, 2.00)
        self.assertGreaterEqual(linear, 0.0)

    def test_stays_on_hug_corner_until_west_of_the_rail(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        linear, _angular, done = follower.step(-0.25, 2.27, math.pi)
        self.assertFalse(done)
        self.assertEqual(follower.idx, 0)
        self.assertGreaterEqual(linear, 0.0)

    def test_scraping_the_divider_tip_reverses_off_the_wall(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.idx = 1
        follower.mode = "drive"
        linear, _angular, done = follower.step(0.07, 1.48, -math.pi / 2.0)
        self.assertFalse(done)
        self.assertLess(linear, 0.0)

    def test_southbound_just_east_of_the_rail_faces_west_not_reverse(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.idx = 1
        follower.mode = "drive"
        linear, angular, done = follower.step(-0.29, 1.87, -1.30)
        self.assertFalse(done)
        self.assertGreaterEqual(linear, 0.0)
        self.assertLess(angular, 0.0)

    def test_overrunning_the_hug_corner_goes_south_instead_of_spinning(self):
        follower = WaypointFollower(
            [(HUG_WEST_X, WEST_LANE_Y), (HUG_WEST_X, SOUTH_PEEL_Y), DELIVERY_APPROACH_XY],
            final_yaw=-math.pi / 2.0,
            pos_tol=0.12,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        follower.step(HUG_WEST_X - 0.01, 1.48, -math.pi / 2.0)
        self.assertGreaterEqual(follower.idx, 1)
        linear, _angular, done = follower.step(HUG_WEST_X - 0.01, 1.48, -math.pi / 2.0)
        self.assertFalse(done)
        self.assertGreater(linear, 0.0)

    def test_eight_times_speed_commands_full_cruise(self):
        follower = WaypointFollower(
            [(1.92, 2.475)],
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=4.4,
        )
        follower.mode = "drive"
        linear, angular, done = follower.step(1.92, -3.17, math.pi / 2.0)
        self.assertFalse(done)
        self.assertGreater(linear, 2.0)
        self.assertEqual(angular, 0.0)

    def test_overshooting_the_east_corner_turns_west(self):
        follower = WaypointFollower(
            build_shelf_route(),
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
            pos_tol=0.12,
        )
        follower.mode = "drive"
        follower.step(1.90, WEST_LANE_Y + 0.04, math.pi / 2.0)
        linear, angular, done = follower.step(1.90, WEST_LANE_Y + 0.04, math.pi / 2.0)
        self.assertFalse(done)
        self.assertGreaterEqual(follower.idx, 1)
        self.assertGreater(angular, 0.0)

    def test_does_not_westbound_south_of_the_yellow_lane(self):
        follower = WaypointFollower(
            build_shelf_route(),
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
            pos_tol=0.12,
        )
        follower.mode = "drive"
        linear, angular, done = follower.step(1.88, 1.68, math.pi / 2.0)
        self.assertFalse(done)
        self.assertEqual(follower.idx, 0)
        self.assertGreater(linear, 0.0)
        self.assertLess(abs(angular), 0.40)

    def test_east_stub_reverses_instead_of_driving_into_shelves(self):
        follower = WaypointFollower(
            build_shelf_route(),
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
        )
        follower.mode = "drive"
        follower.idx = 1
        linear, _angular, done = follower.step(1.87, 2.70, math.pi / 2.0)
        self.assertFalse(done)
        self.assertLess(linear, 0.0)

    def test_unicycle_from_east_spawn_turns_before_the_north_racks(self):
        follower = WaypointFollower(
            build_shelf_route(),
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=1.2,
            pos_tol=0.12,
        )
        x, y, yaw = 1.92, -3.17, math.pi / 2.0
        dt = 0.05
        max_east_y = y
        done = False
        for _ in range(4000):
            linear, angular, done = follower.step(x, y, yaw)
            if done:
                break
            yaw = wrap_to_pi(yaw + angular * dt)
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
            if x > 1.35:
                max_east_y = max(max_east_y, y)
        self.assertTrue(done)
        self.assertLess(max_east_y, 2.50)
        self.assertTrue(in_picking_zone((x, y)))

    def test_unicycle_reaches_aisle_from_delivery_spawn(self):
        follower = WaypointFollower(build_shelf_route(), final_yaw=SHELF_FACE_YAW)
        x, y, yaw = -1.94, -3.17, math.pi / 2.0
        dt = 0.05
        done = False
        for _ in range(4000):
            linear, angular, done = follower.step(x, y, yaw)
            if done:
                break
            yaw = wrap_to_pi(yaw + angular * dt)
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
        self.assertTrue(done)
        self.assertTrue(in_picking_zone((x, y)))
        self.assertAlmostEqual(wrap_to_pi(yaw - SHELF_FACE_YAW), 0.0, delta=0.08)

    def test_unicycle_still_arrives_at_eight_times_speed(self):
        follower = WaypointFollower(
            build_shelf_route(),
            final_yaw=SHELF_FACE_YAW,
            max_lin=2.4,
            max_ang=4.4,
            pos_tol=0.12,
        )
        x, y, yaw = -1.94, -3.17, math.pi / 2.0
        dt = 0.05
        done = False
        for _ in range(2000):
            linear, angular, done = follower.step(x, y, yaw)
            if done:
                break
            yaw = wrap_to_pi(yaw + angular * dt)
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
        self.assertTrue(done)
        self.assertTrue(in_picking_zone((x, y)))

    def test_forward_cone_ignores_side_hits(self):
        ranges = [8.0] * 36
        ranges[0] = 0.25
        ranges[9] = 0.10
        nearest = min_forward_range(ranges, angle_min=0.0, angle_increment=math.pi / 18.0, cone_half=0.40)
        self.assertAlmostEqual(nearest, 0.25)


if __name__ == "__main__":
    unittest.main()
