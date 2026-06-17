"""Base retargeting stage: hand motion -> robot joint trajectory.

Consumes per-frame 21 keypoints (wrist-relative), the robot URDF + hand-type
config, and emits a :class:`RetargetingResult` (joint trajectory) appended to
``clip.retargeting``. Clip-level (needs the temporal sequence for smoothing).

Two retarget interfaces are exposed (EgoDex notes per-frame high-DOF IK is
unreliable -> robot-centric joint-space alignment is often better):
  * ``optimization`` -- ``dex_retargeting`` (Position/Vector/DexPilot + SeqRetarget)
  * ``aligned``      -- a pluggable learned/aligned retargeter (interface only)
"""

from __future__ import annotations

import numpy as np

from ..schema.core import ClipAnnotation, HandSide, RetargetingResult
from ..topology import wrist_relative
from .robots import RobotConfig, get_robot_config


class RetargetStageBase:
    """Mixin-style base; concrete stages inherit Stage + this (see dex_retargeting)."""

    def collect_hand_sequence(
        self, clip: ClipAnnotation, side: HandSide
    ) -> tuple[list[int], list[float], np.ndarray]:
        """Gather wrist-relative 21x3 keypoints for ``side`` over the clip.

        Returns ``(frame_ids, timestamps, kpts)`` where ``kpts`` is ``(T,21,3)``.
        """
        frame_ids: list[int] = []
        timestamps: list[float] = []
        seq: list[np.ndarray] = []
        for fa in clip.frames:
            for hand in fa.hands:
                if HandSide(hand.side) != side or hand.keypoints_3d is None:
                    continue
                seq.append(wrist_relative(hand.kp3d_array()))
                frame_ids.append(fa.frame_id)
                timestamps.append(fa.timestamp if fa.timestamp is not None else float(fa.frame_id))
                break
        kpts = np.stack(seq, axis=0) if seq else np.zeros((0, 21, 3))
        return frame_ids, timestamps, kpts

    def resolve_robot(self) -> RobotConfig:
        name = self.param("robot", "orca")  # type: ignore[attr-defined]
        overrides = {
            "urdf_path": self.param("urdf_path"),  # type: ignore[attr-defined]
            "retargeting_type": self.param("retargeting_type"),  # type: ignore[attr-defined]
            "hand_type": self.param("hand_type"),  # type: ignore[attr-defined]
        }
        # optional link-name overrides from config
        for key in (
            "finger_tip_link_names",
            "wrist_link_name",
            "target_link_names",
            "target_link_human_indices",
            "target_joint_names",
            "scaling_factor",
        ):
            v = self.param(key)  # type: ignore[attr-defined]
            if v is not None:
                overrides[key] = v
        return get_robot_config(name, **{k: v for k, v in overrides.items() if v is not None})

    def synthetic_result(
        self, robot: RobotConfig, side: HandSide, frame_ids, timestamps, kpts: np.ndarray
    ) -> RetargetingResult:
        """Deterministic dry-run trajectory: maps finger spread to joint angles."""
        dof = robot.num_dof or 16
        joint_names = [f"joint_{i}" for i in range(dof)]
        if kpts.shape[0] == 0:
            traj = np.zeros((0, dof))
        else:
            # crude proxy: per-frame mean fingertip distance from wrist -> a global
            # "closure" scalar broadcast across joints (purely for a runnable graph).
            tips = kpts[:, [4, 8, 12, 16, 20], :]
            closure = 1.0 - np.clip(np.linalg.norm(tips, axis=-1).mean(axis=1), 0, 0.2) / 0.2
            traj = np.tile(closure[:, None], (1, dof)) * 0.5
        return RetargetingResult(
            robot=robot.name,
            hand_side=side,
            optimizer=robot.retargeting_type,
            joint_names=joint_names,
            joint_trajectory=traj,
            frame_ids=frame_ids,
            timestamps=timestamps,
            urdf_path=robot.resolved_urdf(),
        )
