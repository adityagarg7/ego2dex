"""DROID-SLAM camera-pose stage (deep visual SLAM).

Dense bundle-adjustment visual SLAM (NeurIPS 2021). License: research (BSD-ish,
see repo). Recovers a dense camera trajectory for moving-camera GoPro clips; it
is also the SLAM front-end option for Dyn-HaMR (alongside VIPE). Needs torch +
custom CUDA ops; wired, not bundled.
"""

from __future__ import annotations

from ...schema.core import ClipAnnotation
from ..base import POSE
from .base import PoseStageBase


@POSE.register("droid", aliases=("droid_slam", "droid-slam"))
class DroidSLAM(PoseStageBase):
    name = "droid"
    requires = ("torch", "droid_slam")
    extra = "pose"
    license = "Research (DROID-SLAM)"
    license_url = "https://github.com/princeton-vl/DROID-SLAM"

    def estimate(self, clip: ClipAnnotation) -> None:
        self.import_or_raise("torch")
        raise NotImplementedError(
            "Wire DROID-SLAM here: stream the (undistorted) frames + intrinsics "
            "through the tracker, then write the recovered world->cam poses onto "
            "clip.frames[*].camera.extrinsics. (VIPE is the Dyn-HaMR alternative.)"
        )
