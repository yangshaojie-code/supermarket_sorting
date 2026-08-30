import unittest

import numpy as np

from runtime.head_camera_kinematics import (
    base_to_head_camera,
    base_to_head_camera_from_joint_state,
)


class HeadCameraKinematicsTests(unittest.TestCase):
    def test_transform_is_rigid_and_slide_uses_official_negative_z_axis(self):
        at_zero = base_to_head_camera(0.0, 0.0, 0.0)
        at_raised_slide = base_to_head_camera(0.20, 0.0, 0.0)
        rotation = at_zero[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=10)
        np.testing.assert_allclose(
            at_zero[:3, 3] - at_raised_slide[:3, 3],
            np.array([0.0, 0.0, 0.20]),
            atol=1e-10,
        )

    def test_joint_state_adapter_matches_direct_kinematics(self):
        names = ("unused", "head_pitch_joint", "slide_joint", "head_yaw_joint")
        positions = (3.0, -0.35, 0.25, 0.15)
        from_joint_state = base_to_head_camera_from_joint_state(names, positions)
        direct = base_to_head_camera(0.25, 0.15, -0.35)
        np.testing.assert_allclose(from_joint_state, direct, atol=1e-12)

    def test_missing_or_invalid_camera_joint_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing camera joints"):
            base_to_head_camera_from_joint_state(("slide_joint",), (0.0,))
        with self.assertRaises(ValueError):
            base_to_head_camera(0.0, 0.51, 0.0)


if __name__ == "__main__":
    unittest.main()
