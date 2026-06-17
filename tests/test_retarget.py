"""Retargeting config + the synthetic (dry-run) trajectory path."""

from __future__ import annotations

import numpy as np

from ego2dex.retarget.robots import ROBOTS, get_robot_config, orca_config
from ego2dex.schema.core import HandSide
from ego2dex.stages.base import RunContext, build_stage


def test_orca_is_first_class_16dof():
    cfg = orca_config()
    assert cfg.name == "orca"
    assert cfg.num_dof == 16
    assert cfg.builtin is None  # custom URDF config, not bundled
    assert len(cfg.finger_tip_link_names) == 5


def test_builtin_robots_present():
    for name in ("allegro", "shadow", "leap"):
        assert name in ROBOTS
    assert ROBOTS["shadow"].num_dof == 24  # DOF mismatch handled by mapping
    assert ROBOTS["allegro"].builtin == "allegro"


def test_get_robot_config_override_urdf():
    cfg = get_robot_config("orca", urdf_path="/tmp/orca.urdf")
    assert cfg.resolved_urdf().endswith("orca.urdf")


def test_dry_run_retarget_handles_dof_mismatch():
    # human 21 kpts -> ORCA 16 DOF, no assumption of equal DOF.
    from ego2dex.schema.core import (
        ClipAnnotation,
        FrameAnnotation,
        HandPose,
        VideoMeta,
    )

    frames = []
    for i in range(4):
        kp3d = np.random.RandomState(i).rand(21, 3)
        kp2d = np.random.RandomState(i + 9).rand(21, 3)
        kp2d[:, 2] = 1.0
        frames.append(
            FrameAnnotation(
                frame_id=i,
                hands=[HandPose(side=HandSide.RIGHT, keypoints_2d=kp2d, keypoints_3d=kp3d)],
            )
        )
    clip = ClipAnnotation(
        video_meta=VideoMeta(path="x", fps=30, width=64, height=48, num_frames=4),
        frames=frames,
    )
    stage = build_stage("retarget", "dex_retargeting", {"robot": "orca", "hand_side": "right"})
    stage.bind(RunContext(dry_run=True))
    stage.ensure_loaded()
    clip = stage.process(clip)
    assert len(clip.retargeting) == 1
    r = clip.retargeting[0]
    assert r.robot == "orca"
    assert r.hand_side == "right"
    assert r.trajectory_array().shape == (4, 16)
