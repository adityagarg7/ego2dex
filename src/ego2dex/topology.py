"""Hand keypoint topology + MANO constants (the #1 interop surface).

Every fact here is load-bearing. The most common bug when stitching hand
estimators, MANO fitters, and retargeters together is a silent finger-order
mismatch (MANO's native order is index, middle, **pinky, ring**, thumb -- pinky
before ring!) or a 45-vs-48 pose-dimension confusion. We pin the constants,
expose explicit remaps, and assert them in ``tests/test_topology.py``.

References (see docs/sota_survey.md):
  * Standard 21-kpt order == MediaPipe == OpenPose == FreiHAND (index-compatible).
  * MANO: 778 verts, 1538 faces, 16 joints, pose 45 (15x3) + 3 global = 48,
    10 betas. Tip vertices are appended to the 16 joints to make 21 keypoints.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #


class HandConvention(str, Enum):
    """Keypoint orderings ego2dex knows how to remap between.

    ``STANDARD21``, ``MEDIAPIPE`` and ``OPENPOSE`` are index-compatible (same
    21-point order). ``MANO_NATIVE`` is the raw MANO joint order with 5 tips
    appended -- a *different* order that must always be remapped explicitly.
    """

    STANDARD21 = "standard21"
    MEDIAPIPE = "mediapipe"
    OPENPOSE = "openpose"
    MANO_NATIVE = "mano_native"


# --------------------------------------------------------------------------- #
# Standard 21-keypoint hand (index 0 = wrist; then thumb..pinky, base->tip)
# --------------------------------------------------------------------------- #
NUM_HAND_KEYPOINTS: int = 21

STANDARD21_NAMES: tuple[str, ...] = (
    "WRIST",  # 0
    "THUMB_CMC",  # 1
    "THUMB_MCP",  # 2
    "THUMB_IP",  # 3
    "THUMB_TIP",  # 4
    "INDEX_MCP",  # 5
    "INDEX_PIP",  # 6
    "INDEX_DIP",  # 7
    "INDEX_TIP",  # 8
    "MIDDLE_MCP",  # 9
    "MIDDLE_PIP",  # 10
    "MIDDLE_DIP",  # 11
    "MIDDLE_TIP",  # 12
    "RING_MCP",  # 13
    "RING_PIP",  # 14
    "RING_DIP",  # 15
    "RING_TIP",  # 16
    "PINKY_MCP",  # 17
    "PINKY_PIP",  # 18
    "PINKY_DIP",  # 19
    "PINKY_TIP",  # 20
)

# Per-finger keypoint indices in the standard-21 order.
FINGERS: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)  # standard-21 tips
MCP_INDICES: tuple[int, ...] = (5, 9, 13, 17)  # finger knuckles (no thumb CMC)

# Skeleton edges (0-based) -- the MediaPipe HAND_CONNECTIONS set. Wrist links to
# each finger base, then each finger chains base->tip. 20 edges.
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
)

# COCO skeleton is 1-based.
COCO_HAND_SKELETON: tuple[tuple[int, int], ...] = tuple((a + 1, b + 1) for a, b in HAND_EDGES)

# NOTE on MediaPipe z: `hand_landmarks` z is wrist-relative (roughly normalized
# by hand size); `hand_world_landmarks` z is metric with the hand center as
# origin. Always check which one a stage emitted before using z.

# --------------------------------------------------------------------------- #
# MANO constants
# --------------------------------------------------------------------------- #
MANO_NUM_VERTICES: int = 778
MANO_NUM_FACES: int = 1538
MANO_NUM_JOINTS: int = 16  # 1 wrist + 3 per finger
MANO_POSE_DIM: int = 45  # 15 joints x 3 axis-angle (NO global)
MANO_GLOBAL_ORIENT_DIM: int = 3
MANO_FULL_POSE_DIM: int = MANO_POSE_DIM + MANO_GLOBAL_ORIENT_DIM  # 48
MANO_NUM_BETAS: int = 10

# MANO's native finger order for the 15 articulated joints (pinky BEFORE ring!).
MANO_FINGER_ORDER: tuple[str, ...] = ("index", "middle", "pinky", "ring", "thumb")

# Fingertip vertex indices used to lift MANO's 16 joints to 21 keypoints.
# Two common conventions exist; pick ONE per model. Order: (thumb, index,
# middle, ring, pinky) -- the order in which tips are appended (joints 16..20).
MANO_TIP_VERTEX_IDS: dict[str, dict[str, list[int]]] = {
    # hassony2/manopth
    "manopth": {
        "right": [745, 317, 444, 556, 673],
        "left": [745, 317, 445, 556, 673],  # differs only at middle
    },
    # otaheri/MANO (and smplx-style)
    "otaheri": {
        "right": [744, 320, 443, 554, 671],
        "left": [744, 320, 443, 554, 671],
    },
}

# Default tip set ego2dex hard-codes (manopth right hand).
DEFAULT_MANO_TIP_VERTEX_IDS: list[int] = MANO_TIP_VERTEX_IDS["manopth"]["right"]

# manopth MANO(16 joints + 5 appended tips = 21, native order) -> standard-21.
# Read as: standard21[i] = mano_native_21[MANOPTH_TO_STANDARD21[i]].
MANOPTH_TO_STANDARD21: tuple[int, ...] = (
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
)

# MANO native 21-keypoint names (16 joints in MANO order + 5 appended tips).
MANO_NATIVE21_NAMES: tuple[str, ...] = (
    "WRIST",  # 0
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",  # 1-3
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",  # 4-6
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",  # 7-9
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",  # 10-12
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",  # 13-15
    "THUMB_TIP",
    "INDEX_TIP",
    "MIDDLE_TIP",
    "RING_TIP",
    "PINKY_TIP",  # 16-20 appended
)


def _invert_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for dst, src in enumerate(perm):
        inv[src] = dst
    return tuple(inv)


# standard-21 -> MANO native 21 (inverse of MANOPTH_TO_STANDARD21).
STANDARD21_TO_MANOPTH: tuple[int, ...] = _invert_permutation(MANOPTH_TO_STANDARD21)

# Identity permutation for the index-compatible conventions.
_IDENTITY21: tuple[int, ...] = tuple(range(NUM_HAND_KEYPOINTS))

# The set of conventions that share the standard-21 order.
_STANDARD_COMPATIBLE = {
    HandConvention.STANDARD21,
    HandConvention.MEDIAPIPE,
    HandConvention.OPENPOSE,
}


# --------------------------------------------------------------------------- #
# Remapping
# --------------------------------------------------------------------------- #
def remap_keypoints(src: HandConvention | str, dst: HandConvention | str) -> tuple[int, ...]:
    """Return a permutation ``perm`` s.t. ``dst_kpts = src_kpts[perm]``.

    Only the standard-21 family and MANO_NATIVE are supported (the conventions
    ego2dex actually emits). Raises ``ValueError`` for unknown pairs so callers
    fail loudly instead of silently scrambling fingers.
    """
    src = HandConvention(src)
    dst = HandConvention(dst)
    if src == dst:
        return _IDENTITY21
    if src in _STANDARD_COMPATIBLE and dst in _STANDARD_COMPATIBLE:
        return _IDENTITY21
    if src == HandConvention.MANO_NATIVE and dst in _STANDARD_COMPATIBLE:
        return MANOPTH_TO_STANDARD21
    if src in _STANDARD_COMPATIBLE and dst == HandConvention.MANO_NATIVE:
        return STANDARD21_TO_MANOPTH
    raise ValueError(f"No keypoint remap defined for {src.value} -> {dst.value}")


def apply_remap(
    keypoints: NDArray[np.floating],
    src: HandConvention | str,
    dst: HandConvention | str,
) -> NDArray[np.floating]:
    """Reorder a ``(21, D)`` keypoint array from ``src`` to ``dst`` convention."""
    kpts = np.asarray(keypoints)
    if kpts.shape[0] != NUM_HAND_KEYPOINTS:
        raise ValueError(f"Expected 21 keypoints, got shape {kpts.shape}")
    perm = remap_keypoints(src, dst)
    return kpts[list(perm)]


def mano_native_to_standard21(keypoints: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convenience: reorder MANO-native 21 keypoints to the standard-21 order."""
    return apply_remap(keypoints, HandConvention.MANO_NATIVE, HandConvention.STANDARD21)


def split_full_pose(full_pose: NDArray[np.floating]) -> tuple[NDArray, NDArray]:
    """Split a 48-d MANO full pose into ``(global_orient[3], pose[45])``.

    Asserts the 48 vs 45 distinction (the other classic interop bug).
    """
    fp = np.asarray(full_pose).reshape(-1)
    if fp.shape[0] != MANO_FULL_POSE_DIM:
        raise ValueError(
            f"MANO full pose must be {MANO_FULL_POSE_DIM} (3 global + 45 pose), "
            f"got {fp.shape[0]}. (Did you pass the 45-d articulated pose only?)"
        )
    return fp[:MANO_GLOBAL_ORIENT_DIM], fp[MANO_GLOBAL_ORIENT_DIM:]


def wrist_relative(
    keypoints_3d: NDArray[np.floating], wrist_index: int = 0
) -> NDArray[np.floating]:
    """Translate 3D keypoints so the wrist sits at the origin.

    This is the canonical input frame for ``dex_retargeting`` (21 wrist-relative
    keypoints). See ``retarget/base.py``.
    """
    kpts = np.asarray(keypoints_3d, dtype=np.float64)
    return kpts - kpts[wrist_index : wrist_index + 1]
