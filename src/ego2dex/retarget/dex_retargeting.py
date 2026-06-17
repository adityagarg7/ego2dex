"""`dex_retargeting` wrapper stage (human hand -> robot joint trajectory).

Standardizes ego2dex on https://github.com/dexsuite/dex-retargeting (MIT,
Pinocchio FK + NLopt SLSQP). Three optimizers, all with temporal smoothing via
``SeqRetargeting``:

  * PositionOptimizer  -- match absolute 3D keypoints (best for offline dataset
    retargeting / imitation learning).
  * VectorOptimizer    -- match keypoint direction vectors (teleop / streaming).
  * DexPilotOptimizer  -- fingertip + palm vectors with contact snapping (precise
    pinching).

Input per frame: 21 wrist-relative keypoints (standard-21). DOF mismatch is
handled entirely by the link/joint mapping in :class:`RobotConfig` -- we never
assume human and robot DOF match.
"""

from __future__ import annotations

import numpy as np

from ..schema.core import ClipAnnotation, HandSide, RetargetingResult
from ..stages.base import RETARGET, Stage
from .base import RetargetStageBase
from .robots import RobotConfig


@RETARGET.register("dex_retargeting", aliases=("dex", "dexretargeting"))
class DexRetargeting(Stage, RetargetStageBase):
    name = "dex_retargeting"
    family = "retarget"
    requires = ("dex_retargeting",)
    extra = "retarget"
    license = "MIT (dex-retargeting + dex-urdf)"
    license_url = "https://github.com/dexsuite/dex-retargeting"
    per_frame = False

    def load(self) -> None:
        self.import_or_raise("dex_retargeting")

    # ------------------------------------------------------------------ #
    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        robot = self.resolve_robot()
        sides = self._requested_sides()
        for side in sides:
            frame_ids, ts, kpts = self.collect_hand_sequence(clip, side)
            if kpts.shape[0] == 0:
                continue
            if self.dry_run:
                result = self.synthetic_result(robot, side, frame_ids, ts, kpts)
            else:
                result = self._retarget(robot, side, frame_ids, ts, kpts)
            clip.retargeting.append(result)
        return clip

    def _requested_sides(self) -> list[HandSide]:
        want = self.param("hand_side", "both")
        if want in ("left", "right"):
            return [HandSide(want)]
        return [HandSide.RIGHT, HandSide.LEFT]

    # ------------------------------------------------------------------ #
    def _retarget(
        self, robot: RobotConfig, side: HandSide, frame_ids, ts, kpts: np.ndarray
    ) -> RetargetingResult:  # pragma: no cover - needs dex_retargeting + URDF
        from dex_retargeting.retargeting_config import RetargetingConfig

        cfg_dict = self._build_config(robot)
        retargeting = RetargetingConfig.from_dict(cfg_dict).build()

        optimizer = retargeting.optimizer
        indices = np.asarray(optimizer.target_link_human_indices)
        is_vector = indices.ndim == 2

        traj: list[list[float]] = []
        for kp in kpts:  # kp: (21,3) wrist-relative
            if is_vector:
                origin, task = indices[0], indices[1]
                ref = kp[task] - kp[origin]
            else:
                ref = kp[indices]
            qpos = retargeting.retarget(ref * robot.scaling_factor)
            traj.append(np.asarray(qpos).tolist())

        return RetargetingResult(
            robot=robot.name,
            hand_side=side,
            optimizer=robot.retargeting_type,
            joint_names=list(retargeting.joint_names),
            joint_trajectory=traj,
            frame_ids=frame_ids,
            timestamps=ts,
            urdf_path=robot.resolved_urdf(),
        )

    # ------------------------------------------------------------------ #
    def _build_config(self, robot: RobotConfig) -> dict:  # pragma: no cover - needs deps
        """Build the dex_retargeting config dict for a robot.

        Built-in robots (allegro/shadow/leap) defer to dex_retargeting's bundled
        configs; ORCA uses a custom dict pointing at the user URDF.
        """
        if robot.builtin:
            from dex_retargeting.constants import (
                HandType,
                RetargetingType,
                RobotName,
                get_default_config_path,
            )
            from dex_retargeting.retargeting_config import RetargetingConfig

            path = get_default_config_path(
                RobotName[robot.builtin],
                RetargetingType[robot.retargeting_type],
                HandType[robot.hand_type],
            )
            return RetargetingConfig.load_from_file(path).__dict__  # already complete

        urdf = robot.resolved_urdf()
        if not urdf:
            raise FileNotFoundError(
                "ORCA retargeting needs a URDF. Set retarget config params.urdf_path "
                "(or EGO2DEX_ORCA_URDF) to your orcahand URDF, and verify the link "
                "names in retarget/robots.py::orca_config against it."
            )
        t = robot.retargeting_type
        base = {
            "type": t,
            "urdf_path": urdf,
            "add_dummy_free_joint": False,
            "scaling_factor": robot.scaling_factor,
            "low_pass_alpha": robot.low_pass_alpha,
        }
        if robot.target_joint_names:
            base["target_joint_names"] = robot.target_joint_names
        if t == "position":
            base.update(
                target_link_names=robot.target_link_names,
                target_link_human_indices=np.asarray(robot.target_link_human_indices),
            )
        elif t == "vector":
            n = len(robot.finger_tip_link_names)
            base.update(
                target_origin_link_names=[robot.wrist_link_name] * n,
                target_task_link_names=robot.finger_tip_link_names,
                target_link_human_indices=np.array([[0] * n, robot.target_link_human_indices]),
            )
        elif t == "dexpilot":
            base.update(
                finger_tip_link_names=robot.finger_tip_link_names,
                wrist_link_name=robot.wrist_link_name,
            )
        else:
            raise ValueError(f"Unknown retargeting_type '{t}'")
        return base
