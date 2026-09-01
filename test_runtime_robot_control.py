import unittest

import numpy as np

from runtime.ros_robot_control import ControlSafetyError, RosRobotController


class FakeTwist:
    def __init__(self):
        self.linear = type("Linear", (), {"x": 0.0})()
        self.angular = type("Angular", (), {"z": 0.0})()


class FakeArray:
    def __init__(self):
        self.data = []


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, value):
        self.messages.append(value)


class FakeNode:
    def __init__(self):
        self.publishers = []

    def create_publisher(self, _kind, _topic, _depth):
        publisher = Publisher()
        self.publishers.append(publisher)
        return publisher


class RobotControlTests(unittest.TestCase):
    def test_velocity_is_clamped_and_stop_is_zero(self):
        node = FakeNode()
        controller = RosRobotController(node, FakeTwist, FakeArray)
        controller.publish_velocity(3.0, -3.0)
        self.assertAlmostEqual(node.publishers[0].messages[-1].linear.x, 0.35)
        self.assertAlmostEqual(node.publishers[0].messages[-1].angular.z, -0.65)
        controller.stop_base()
        self.assertEqual(node.publishers[0].messages[-1].linear.x, 0.0)
        self.assertEqual(node.publishers[0].messages[-1].angular.z, 0.0)

    def test_custom_base_limits_are_used_when_set(self):
        node = FakeNode()
        controller = RosRobotController(
            node, FakeTwist, FakeArray, max_linear=2.4, max_angular=4.4
        )
        controller.publish_velocity(9.0, -9.0)
        self.assertAlmostEqual(node.publishers[0].messages[-1].linear.x, 2.4)
        self.assertAlmostEqual(node.publishers[0].messages[-1].angular.z, -4.4)

    def test_arm_requires_seven_values_including_gripper(self):
        controller = RosRobotController(FakeNode(), FakeTwist, FakeArray)
        with self.assertRaises(ControlSafetyError):
            controller.command_arm("r", [0.0] * 6)

    def test_stop_all_still_stops_base_with_invalid_joint_feedback(self):
        node = FakeNode()
        sensors = type("Sensors", (), {
            "joint_names": ("slide_joint", "head_yaw_joint", "head_pitch_joint"),
            "joint_positions": np.array([np.nan, 0.0, 0.0]),
        })()
        controller = RosRobotController(node, FakeTwist, FakeArray, sensors)
        controller.stop_all()
        self.assertEqual(node.publishers[0].messages[-1].linear.x, 0.0)
        self.assertEqual(node.publishers[0].messages[-1].angular.z, 0.0)


if __name__ == "__main__":
    unittest.main()
