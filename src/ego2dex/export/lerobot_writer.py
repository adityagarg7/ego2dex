"""LeRobotDataset export (imitation-learning pipelines).

Writes an episode into a LeRobotDataset (HF ``lerobot``). Features include the
retargeted robot ``action`` / ``observation.state`` (joint angles), camera
intrinsics/extrinsics, and the language task. Needs ``pip install lerobot``.

LeRobot's schema evolves; this wires the common ``create`` + ``add_frame`` +
``save_episode`` flow and flags the exact call site to adjust to your version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema.core import ClipAnnotation, HandSide
from ..stages.base import EXPORT
from .base import ExportStageBase


@EXPORT.register("lerobot", aliases=("lerobot_writer",))
class LeRobotWriter(ExportStageBase):
    name = "lerobot"
    requires = ("lerobot",)
    extra = "lerobot"
    license = "Apache-2.0 (LeRobot)"
    license_url = "https://github.com/huggingface/lerobot"

    def write(self, clip: ClipAnnotation) -> Path:  # pragma: no cover - needs lerobot
        self.import_or_raise("lerobot")
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        out = self.out_dir()
        repo_id = self.param("repo_id", f"ego2dex/{clip.video_meta.source}")
        side = HandSide(self.param("hand_side", "right"))
        result = next((r for r in clip.retargeting if HandSide(r.hand_side) == side), None)
        if result is None:
            raise ValueError(
                f"No retargeting result for hand_side={side.value}; run the retarget "
                "stage before the lerobot export."
            )
        traj = result.trajectory_array()
        dof = traj.shape[1] if traj.size else (len(result.joint_names) or 16)

        features = {
            "action": {"dtype": "float32", "shape": (dof,), "names": result.joint_names},
            "observation.state": {"dtype": "float32", "shape": (dof,), "names": result.joint_names},
        }
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(clip.video_meta.sampled_fps or clip.video_meta.fps or 30),
            root=out / "lerobot",
            features=features,
        )
        task = self.param("language", "manipulation")
        for i in range(traj.shape[0]):
            q = traj[i].astype(np.float32)
            dataset.add_frame({"action": q, "observation.state": q, "task": task})
        dataset.save_episode()
        return out / "lerobot"
