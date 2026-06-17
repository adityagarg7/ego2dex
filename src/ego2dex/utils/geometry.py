"""Pure-numpy SE(3) / SO(3) helpers.

Conventions (fixed across ego2dex; see ``schema/core.py::CameraParams``):
  * Quaternions are **Hamilton, scalar-first** ``[w, x, y, z]``.
  * Rotation matrices are right-handed, column-vector convention ``p' = R @ p``.
  * SE(3) is a 4x4 homogeneous matrix ``[[R, t], [0, 0, 0, 1]]``.
  * Axis-angle ("rotation vector") follows OpenCV/Rodrigues, as used by MANO
    (``global_orient`` and the 15x3 / 45-vector ``pose``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Axis-angle (rotation vector) <-> rotation matrix  (Rodrigues)
# --------------------------------------------------------------------------- #
def axis_angle_to_rotmat(aa: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert axis-angle vector(s) of shape ``(..., 3)`` to rotmat ``(..., 3, 3)``."""
    aa = np.asarray(aa, dtype=np.float64)
    if aa.shape[-1] != 3:
        raise ValueError(f"axis_angle expects last dim 3, got {aa.shape}")
    batch = aa.reshape(-1, 3)
    theta = np.linalg.norm(batch, axis=1, keepdims=True)  # (N,1)
    out = np.empty((batch.shape[0], 3, 3), dtype=np.float64)
    for i in range(batch.shape[0]):
        t = theta[i, 0]
        if t < EPS:
            out[i] = np.eye(3)
            continue
        k = batch[i] / t
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        out[i] = np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)
    return out.reshape(*aa.shape[:-1], 3, 3)


def rotmat_to_axis_angle(R: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert rotmat ``(..., 3, 3)`` to axis-angle ``(..., 3)``."""
    R = np.asarray(R, dtype=np.float64)
    batch = R.reshape(-1, 3, 3)
    out = np.empty((batch.shape[0], 3), dtype=np.float64)
    for i in range(batch.shape[0]):
        Ri = batch[i]
        cos = np.clip((np.trace(Ri) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos)
        if theta < EPS:
            out[i] = np.zeros(3)
            continue
        if np.pi - theta < 1e-4:  # near 180 deg: use symmetric part
            A = (Ri + np.eye(3)) / 2.0
            axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
            # fix signs from off-diagonal terms
            if axis[0] > EPS:
                axis[1] = np.sign(A[0, 1]) * axis[1]
                axis[2] = np.sign(A[0, 2]) * axis[2]
            out[i] = axis / (np.linalg.norm(axis) + EPS) * theta
            continue
        axis = np.array([Ri[2, 1] - Ri[1, 2], Ri[0, 2] - Ri[2, 0], Ri[1, 0] - Ri[0, 1]]) / (
            2.0 * np.sin(theta)
        )
        out[i] = axis * theta
    return out.reshape(*R.shape[:-2], 3)


# --------------------------------------------------------------------------- #
# Quaternion (Hamilton, scalar-first)  <-> rotation matrix
# --------------------------------------------------------------------------- #
def normalize_quat(q: NDArray[np.floating]) -> NDArray[np.float64]:
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + EPS)


def quat_to_rotmat(q: NDArray[np.floating]) -> NDArray[np.float64]:
    """Hamilton scalar-first quaternion ``[w,x,y,z]`` -> 3x3 rotation matrix."""
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(R: NDArray[np.floating]) -> NDArray[np.float64]:
    """3x3 rotation matrix -> Hamilton scalar-first quaternion ``[w,x,y,z]``."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:  # canonical: non-negative scalar part
        q = -q
    return normalize_quat(q)


# --------------------------------------------------------------------------- #
# SE(3)
# --------------------------------------------------------------------------- #
def make_se3(
    R: NDArray[np.floating] | None = None, t: NDArray[np.floating] | None = None
) -> NDArray[np.float64]:
    """Build a 4x4 SE(3) from rotation ``R`` (3x3) and translation ``t`` (3,)."""
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_se3(T: NDArray[np.floating]) -> NDArray[np.float64]:
    """Inverse of an SE(3) matrix (exploits ``R^-1 = R^T``)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: NDArray[np.floating], pts: NDArray[np.floating]) -> NDArray[np.float64]:
    """Apply SE(3) ``T`` to points ``(..., 3)``; returns ``(..., 3)``."""
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)  # (N,4)
    out = (homog @ T.T)[:, :3]
    return out.reshape(pts.shape)
