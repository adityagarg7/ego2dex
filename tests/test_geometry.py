"""SO(3)/SE(3) helpers round-trip and stay valid."""

from __future__ import annotations

import numpy as np

from ego2dex.utils.geometry import (
    axis_angle_to_rotmat,
    invert_se3,
    make_se3,
    quat_to_rotmat,
    rotmat_to_axis_angle,
    rotmat_to_quat,
    transform_points,
)


def test_axis_angle_roundtrip():
    rng = np.random.RandomState(0)
    for _ in range(20):
        aa = rng.randn(3) * 0.5
        R = axis_angle_to_rotmat(aa)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        aa2 = rotmat_to_axis_angle(R)
        assert np.allclose(aa, aa2, atol=1e-7)


def test_axis_angle_batched():
    aa = np.random.RandomState(1).randn(15, 3) * 0.2
    R = axis_angle_to_rotmat(aa)
    assert R.shape == (15, 3, 3)


def test_quat_roundtrip():
    rng = np.random.RandomState(2)
    for _ in range(20):
        R = axis_angle_to_rotmat(rng.randn(3))
        q = rotmat_to_quat(R)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert q[0] >= 0  # canonical scalar-first, non-negative w
        assert np.allclose(quat_to_rotmat(q), R, atol=1e-9)


def test_se3_inverse_and_transform():
    R = axis_angle_to_rotmat(np.array([0.2, -0.1, 0.3]))
    T = make_se3(R, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(invert_se3(T) @ T, np.eye(4), atol=1e-9)
    pts = np.random.RandomState(4).randn(10, 3)
    back = transform_points(invert_se3(T), transform_points(T, pts))
    assert np.allclose(back, pts, atol=1e-9)
