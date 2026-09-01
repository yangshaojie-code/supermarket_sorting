import math
import unittest
from types import SimpleNamespace

from runtime.scene_zones import (
    SHELF_APPROACH_XY,
    SHELF_FACE_YAW,
    in_picking_zone,
)
from runtime.waypoint_nav import (
    WaypointFollower,
    build_shelf_route,
    min_forward_range,
    pose_from_odom,
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
        self.assertGreaterEqual(len(route), 2)
        self.assertEqual(route[-1], SHELF_APPROACH_XY)
        self.assertTrue(in_picking_zone(route[-1]))

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
