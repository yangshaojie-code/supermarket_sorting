import unittest

import numpy as np

from runtime.motion_planning import IKSolveError, MMK2KdlBackend


class MMK2KdlBackendTests(unittest.TestCase):
    """Numerical regression tests against the supermarket baseline KDL model."""

    def setUp(self):
        self.backend = MMK2KdlBackend()
        self.slide = 0.10
        self.reference = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])

    def test_single_arm_fk_ik_round_trip(self):
        for arm in ("l", "r"):
            target = self.backend.forward(arm, self.slide, self.reference)
            solution = self.backend.solve(
                target[:3, 3], target[:3, :3], arm, self.slide, self.reference
            )
            recovered = self.backend.forward(arm, self.slide, solution)
            np.testing.assert_allclose(recovered, target, atol=1e-6)

    def test_bimanual_fk_ik_round_trip(self):
        left = self.backend.forward("l", self.slide, self.reference)
        right = self.backend.forward("r", self.slide, self.reference)
        slide, left_solution, right_solution = self.backend.solve_bimanual(
            left[:3, 3],
            left[:3, :3],
            right[:3, 3],
            right[:3, :3],
            self.reference,
            self.reference,
            self.slide,
            slide_pos=self.slide,
        )
        self.assertAlmostEqual(slide, self.slide)
        np.testing.assert_allclose(
            self.backend.forward("l", slide, left_solution), left, atol=1e-6
        )
        np.testing.assert_allclose(
            self.backend.forward("r", slide, right_solution), right, atol=1e-6
        )

    def test_rejects_out_of_limit_slide_and_references(self):
        with self.assertRaises(ValueError):
            self.backend.forward("l", 0.88, self.reference)
        with self.assertRaises(ValueError):
            self.backend.forward("l", self.slide, np.full(6, np.nan))

    def test_unreachable_target_fails_closed(self):
        with self.assertRaises(IKSolveError):
            self.backend.solve(
                np.array([10.0, 10.0, 10.0]),
                np.eye(3),
                "l",
                self.slide,
                self.reference,
            )


if __name__ == "__main__":
    unittest.main()
