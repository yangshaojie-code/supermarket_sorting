"""Dynamic MMK2 head-camera extrinsics derived from the official MJCF.

The supermarket Server publishes only ``odom -> base_link``.  RGB-D points
arrive in the optical ``head_camera`` frame, so the client must build the
missing ``base_link <- head_camera`` transform from real-time slide and head
joints.  Constants match the official MMK2 MJCF used by both competitions.
``headeye`` contributes the final 180-degree X rotation that converts MuJoCo's
camera convention to the RGB-D optical convention.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

SLIDE_LIMITS = (-0.04, 0.87)
HEAD_YAW_LIMITS = (-0.50, 0.50)
HEAD_PITCH_LIMITS = (-1.18, 0.16)

# Yellow-aisle look. Camera mount already pitches down ~19 deg, so joint pitch
# near 0 looks at the shelf face ~1 m ahead. Official grasp pitch -0.6 is only
# for close creep and hits the floor from the aisle.
AISLE_SCAN_SPINE = 0.11
AISLE_SCAN_HEAD = (0.0, -0.12)
# Look at L1 first, then pitch down for L2/L3. Do not stop the scan just
# because an upper-layer marker decoded — product_032 sits on D-L2-C3.
AISLE_SCAN_POSES = (
    (0.0, -0.12),
    (0.0, -0.28),
    (0.0, -0.50),
    (0.0, 0.00),
    (-0.30, -0.12),
    (0.30, -0.12),
)


def _translation(values) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(values, dtype=float)
    return matrix


def _axis_rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return matrix


def _euler_xyz(values) -> np.ndarray:
    x, y, z = values
    return _axis_rotation((1, 0, 0), x) @ _axis_rotation((0, 1, 0), y) @ _axis_rotation((0, 0, 1), z)


def _quat_wxyz(values) -> np.ndarray:
    w, x, y, z = np.asarray(values, dtype=float)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("MJCF quaternion cannot be zero")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return matrix


def _finite_in_range(name: str, value: float, limits) -> float:
    value = float(value)
    if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
        raise ValueError(f"{name} must be finite and within {limits}")
    return value


def base_to_head_camera(slide: float, head_yaw: float, head_pitch: float) -> np.ndarray:
    """Return the matrix that maps an optical-frame point into ``base_link``."""
    slide = _finite_in_range("slide_joint", slide, SLIDE_LIMITS)
    head_yaw = _finite_in_range("head_yaw_joint", head_yaw, HEAD_YAW_LIMITS)
    head_pitch = _finite_in_range("head_pitch_joint", head_pitch, HEAD_PITCH_LIMITS)

    # base_link is a site at -0.02371 m in agv_link, so agv_link is +0.02371 m
    # along X from the ROS base frame.  slide_joint's local axis is [0, 0, -1].
    base_to_slide = _translation((0.02371, 0.0, 1.311 - slide))
    slide_to_yaw = _translation((0.18375, 0.0, 0.023)) @ _euler_xyz((0.0, 0.0, 1.5708)) @ _axis_rotation((0, 0, 1), head_yaw)
    yaw_to_pitch = _translation((0.00099952, 0.000031059, 0.058)) @ _quat_wxyz((0.5, -0.5, 0.5, -0.5)) @ _axis_rotation((0, 0, -1), head_pitch)
    pitch_to_camera = _translation((0.0755, -0.1855, 0.0)) @ _quat_wxyz((0.0, 0.70711, 0.0, -0.70711)) @ _translation((-0.035, 0.0, 0.0)) @ _euler_xyz((-0.33, 0.0, 0.0))
    camera_to_optical = _euler_xyz((3.1416, 0.0, 0.0))
    return base_to_slide @ slide_to_yaw @ yaw_to_pitch @ pitch_to_camera @ camera_to_optical


def base_to_head_camera_from_joint_state(names: Iterable[str], positions: Iterable[float]) -> np.ndarray:
    values = dict(zip((str(name) for name in names), (float(value) for value in positions)))
    required = ("slide_joint", "head_yaw_joint", "head_pitch_joint")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"joint state missing camera joints: {missing}")
    return base_to_head_camera(*(values[name] for name in required))
