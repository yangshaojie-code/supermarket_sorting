"""MMK2 inverse-kinematics backend wrapping the official NumPy KDL solver.

Uses ``kinematics.mmk2_kdl.MMK2Kdl`` already shipped with the supermarket
baseline.  Fail-closed: missing or out-of-limit solutions raise ``IKSolveError``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

ARM_JOINT_COUNT = 6
SPINE_LIMITS = (-0.04, 0.87)


class IKSolveError(ValueError):
    """Raised when no finite, in-limit kinematic solution exists."""


def _pose(position, rotation) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("target position/rotation must have shapes (3,) and (3, 3)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        raise ValueError("target pose must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.0:
        raise ValueError("target rotation must be a proper orthonormal matrix")
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _arm_name(arm: str) -> str:
    if arm not in ("l", "r"):
        raise ValueError("arm must be 'l' or 'r'")
    return "left" if arm == "l" else "right"


class MMK2KdlBackend:
    """Adapter around the official pure-NumPy ``MMK2Kdl`` solver."""

    def __init__(self, solver=None, slide_samples: int = 25):
        if solver is None:
            from kinematics.mmk2_kdl import MMK2Kdl

            solver = MMK2Kdl(iteration=0)
        self._solver = solver
        self.slide_samples = max(2, int(slide_samples))

    def forward(self, arm: str, slide_pos: float, joints) -> np.ndarray:
        joints = self._validate_arm_joints(joints)
        slide = self._validate_slide(slide_pos)
        left, right = self._solver.forward_kinematics(
            np.r_[slide, joints], index=_arm_name(arm)
        )
        return np.asarray(left if arm == "l" else right, dtype=float)

    def solve(self, target_pos, target_rmat, arm, slide_pos, q_ref) -> np.ndarray:
        transform = _pose(target_pos, target_rmat)
        q_ref = self._validate_arm_joints(q_ref)
        slide = self._validate_slide(slide_pos)
        reference = np.r_[slide, q_ref]
        kwargs = {
            "T_left": transform if arm == "l" else None,
            "T_right": transform if arm == "r" else None,
            "ref_pos": reference,
            "target_height": slide,
        }
        _arm_name(arm)
        solutions = self._solver.inverse_kinematics(**kwargs)
        result = self._best_solution(solutions, reference)
        return result[1:]

    def solve_with_slide_search(
        self, target_pos, target_rmat, arm: str, slide_reference: float, q_ref
    ) -> Tuple[float, np.ndarray]:
        """Solve one arm while deterministically searching valid slide heights."""
        target = _pose(target_pos, target_rmat)
        q_ref = self._validate_arm_joints(q_ref)
        reference = np.r_[self._validate_slide(slide_reference), q_ref]
        solutions = self._search(target if arm == "l" else None,
                                 target if arm == "r" else None, reference)
        result = self._best_solution(solutions, reference)
        return float(result[0]), result[1:]

    def solve_bimanual(
        self,
        left_pos,
        left_rmat,
        right_pos,
        right_rmat,
        q_left_ref,
        q_right_ref,
        slide_reference: float,
        slide_pos: Optional[float] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Solve both arms together, optionally searching the slide height."""
        left = _pose(left_pos, left_rmat)
        right = _pose(right_pos, right_rmat)
        q_left_ref = self._validate_arm_joints(q_left_ref)
        q_right_ref = self._validate_arm_joints(q_right_ref)
        reference = np.r_[self._validate_slide(slide_reference), q_left_ref, q_right_ref]
        if slide_pos is None:
            solutions = self._search(left, right, reference)
        else:
            slide = self._validate_slide(slide_pos)
            solutions = self._solver.inverse_kinematics(
                T_left=left, T_right=right, ref_pos=reference, target_height=slide
            )
        result = self._best_solution(solutions, reference)
        return float(result[0]), result[1:7], result[7:13]

    def _search(self, left, right, reference) -> list:
        current = float(reference[0])
        grid = np.linspace(SPINE_LIMITS[0], SPINE_LIMITS[1], self.slide_samples)
        heights = sorted({current, *map(float, grid)}, key=lambda value: abs(value - current))
        candidates = []
        for height in heights:
            values = self._solver.inverse_kinematics(
                T_left=left, T_right=right, ref_pos=reference, target_height=height
            )
            if values:
                candidates.extend(values)
        return candidates

    @staticmethod
    def _validate_slide(value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or not SPINE_LIMITS[0] <= value <= SPINE_LIMITS[1]:
            raise ValueError(f"slide position must be within {SPINE_LIMITS}")
        return value

    def _validate_arm_joints(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (ARM_JOINT_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError("arm joint reference must contain six finite values")
        limits = np.asarray(self._solver.left_arm.dh.joints_limit, dtype=float)
        if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
            raise ValueError("arm joint reference exceeds official joint limits")
        return values

    @staticmethod
    def _best_solution(solutions, reference) -> np.ndarray:
        if solutions is None:
            raise IKSolveError("official MMK2Kdl found no solution")
        valid = [np.asarray(value, dtype=float) for value in solutions]
        valid = [value for value in valid if value.shape == reference.shape and np.all(np.isfinite(value))]
        if not valid:
            raise IKSolveError("official MMK2Kdl returned no finite solution")
        return min(valid, key=lambda value: float(np.sum(np.abs(value - reference))))
