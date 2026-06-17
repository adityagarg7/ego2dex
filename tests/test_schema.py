"""Schema (de)serialization, validation, RLE round-trips, MANO dim guards."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ego2dex.schema import (
    ClipAnnotation,
    HandPose,
    HandSide,
    MANOParams,
    Mask,
    decode_mask,
    encode_mask,
    mask_area,
    validate_clip_json,
)


def test_clip_json_roundtrip(sample_clip: ClipAnnotation):
    js = sample_clip.model_dump_json()
    again = ClipAnnotation.model_validate_json(js)
    assert again.frames[0].hands[0].side == HandSide.RIGHT.value
    assert again.frames[0].detections[0].label == "cup"
    validate_clip_json(js)


def test_numpy_coercion():
    hp = HandPose.from_arrays(np.random.rand(21, 3), side="left")
    assert isinstance(hp.keypoints_2d, list)
    assert len(hp.keypoints_2d) == 21 and len(hp.keypoints_2d[0]) == 3


def test_handpose_validates_count():
    with pytest.raises(ValidationError):
        HandPose(keypoints_2d=[[0, 0, 1]] * 5)  # not 21


def test_mano_dim_guard():
    with pytest.raises(ValidationError):
        MANOParams(global_orient=[0, 0, 0], pose=[0.0] * 48, betas=[0.0] * 10)  # 48 != 45
    ok = MANOParams(global_orient=[0, 0, 0], pose=[0.0] * 45, betas=[0.0] * 10)
    assert len(ok.full_pose()) == 48


def test_rle_roundtrip():
    rng = np.random.RandomState(0)
    m = (rng.rand(30, 40) > 0.5).astype(np.uint8)
    rle = encode_mask(m)
    assert decode_mask(rle).shape == m.shape
    assert np.array_equal(decode_mask(rle), m)
    assert mask_area(rle) == int(m.sum())


def test_mask_from_binary_and_decode():
    m = np.zeros((20, 25), dtype=np.uint8)
    m[3:8, 4:10] = 1
    mask = Mask.from_binary(m, instance_id=2, label="bowl")
    assert mask.size == [20, 25]
    assert np.array_equal(mask.decode(), m)


def test_clip_save_load(tmp_path, sample_clip):
    p = sample_clip.save(tmp_path / "clip.json")
    loaded = ClipAnnotation.load(p)
    assert len(loaded.frames) == len(sample_clip.frames)
