"""Shared pytest fixtures (CPU-only, no weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego2dex.schema.core import (
    CameraModelType,
    CameraParams,
    ClipAnnotation,
    Detection,
    FrameAnnotation,
    HandPose,
    HandSide,
    MANOParams,
    Mask,
    VideoMeta,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "synthetic"


@pytest.fixture
def synthetic_dir() -> Path:
    assert ASSETS.exists(), "run scripts/make_synthetic.py first"
    return ASSETS


@pytest.fixture
def sample_hand() -> HandPose:
    kp2d = np.random.RandomState(0).rand(21, 3)
    kp2d[:, 2] = 0.9
    kp3d = np.random.RandomState(1).rand(21, 3)
    return HandPose(
        side=HandSide.RIGHT,
        keypoints_2d=kp2d,
        keypoints_3d=kp3d,
        mano=MANOParams(global_orient=[0, 0, 0], pose=[0.0] * 45, betas=[0.0] * 10),
    )


@pytest.fixture
def sample_clip(sample_hand: HandPose) -> ClipAnnotation:
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:20, 15:30] = 1
    frame = FrameAnnotation(
        frame_id=0,
        timestamp=0.0,
        camera=CameraParams(
            model=CameraModelType.PINHOLE,
            width=64,
            height=48,
            intrinsics={"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0},
            distortion=[0, 0, 0, 0, 0],
        ),
        hands=[sample_hand],
        detections=[Detection(label="cup", bbox=[15, 10, 15, 10], score=0.9)],
        masks=[Mask.from_binary(mask, instance_id=1, label="cup")],
    )
    return ClipAnnotation(
        video_meta=VideoMeta(path="x.mp4", fps=30, width=64, height=48, num_frames=1),
        frames=[frame],
    )
