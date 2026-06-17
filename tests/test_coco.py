"""COCO export + keypoint/box round-trips."""

from __future__ import annotations

import numpy as np

from ego2dex.schema import HandSide, to_coco
from ego2dex.schema.coco import (
    coco_keypoints_to_handpose,
    detection_to_coco_ann,
    handpose_to_coco_keypoints,
)
from ego2dex.schema.core import Detection, HandPose
from ego2dex.topology import COCO_HAND_SKELETON, STANDARD21_NAMES


def test_keypoint_flat_roundtrip():
    kp = np.random.RandomState(0).rand(21, 3)
    kp[:, 2] = 1.0
    hand = HandPose.from_arrays(kp, side="right")
    flat = handpose_to_coco_keypoints(hand)
    assert len(flat) == 63
    back = coco_keypoints_to_handpose(flat, side="right")
    assert np.allclose(hand.kp2d_array()[:, :2], back.kp2d_array()[:, :2])
    assert back.side == HandSide.RIGHT.value


def test_detection_to_coco_ann():
    det = Detection(label="cup", bbox=[1, 2, 3, 4], score=0.7)
    ann = detection_to_coco_ann(det, ann_id=1, image_id=0, category_id=2)
    assert ann["bbox"] == [1, 2, 3, 4]
    assert ann["area"] == 12
    assert ann["category_id"] == 2


def test_to_coco_structure(sample_clip):
    coco = to_coco(sample_clip)
    assert {"images", "annotations", "categories"} <= set(coco)
    cats = {c["name"]: c for c in coco["categories"]}
    assert "hand" in cats
    assert cats["hand"]["keypoints"] == list(STANDARD21_NAMES)
    assert cats["hand"]["skeleton"] == [list(e) for e in COCO_HAND_SKELETON]
    assert "cup" in cats  # from the detection/mask label
    # one image per frame; at least the hand + detection + mask annotations
    assert len(coco["images"]) == len(sample_clip.frames)
    assert len(coco["annotations"]) >= 3


def test_coco_skeleton_is_one_based():
    for a, b in COCO_HAND_SKELETON:
        assert a >= 1 and b >= 1
