"""Base for camera-pose / SLAM stages.

These recover per-frame camera intrinsics + extrinsics (and optionally a sparse
point cloud / trajectory). For GoPro there is no MPS, so we run SLAM/SfM
(DROID-SLAM, COLMAP). For Project Aria we read MPS directly (no SLAM needed).
Clip-level (per_frame=False): a stage may need the whole sequence.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import CameraParams, ClipAnnotation
from ...utils.geometry import make_se3
from ..base import Stage


class PoseStageBase(Stage):
    family = "pose"
    per_frame = False

    def estimate(self, clip: ClipAnnotation) -> None:
        raise NotImplementedError  # pragma: no cover

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        if self.dry_run:
            self._synthetic(clip)
        else:
            self.estimate(clip)
        return clip

    def _synthetic(self, clip: ClipAnnotation) -> None:
        """A smooth forward-translating camera trajectory (world->cam SE3)."""
        for i, fa in enumerate(clip.frames):
            if fa.camera is None:
                fa.camera = CameraParams(width=clip.video_meta.width, height=clip.video_meta.height)
            # camera moves +1cm/frame in z; world->cam = inverse of that motion.
            t = np.array([0.0, 0.0, -0.01 * i])
            fa.camera.extrinsics = make_se3(np.eye(3), t).tolist()
            fa.camera.extrinsics_direction = "world_to_cam"
