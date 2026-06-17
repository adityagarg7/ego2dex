"""Dyn-HaMR hand-pose stage (temporal, moving-camera, world-frame, two hands).

CVPR 2025, arXiv:2412.12861. The reference *video* pipeline: SLAM (VIPE/DROID) +
HaMeR + WiLoR + ViTPose + an interacting-hand motion prior -> 4D global MANO.
Batch/offline -- it consumes a whole clip, not single frames.

License: MIT (code) + MANO non-commercial. Install from source:
    git clone https://github.com/ZhengdiYu/Dyn-HaMR
"""

from __future__ import annotations

from ...schema.core import ClipAnnotation
from ...topology import HandConvention
from ..base import HANDS
from .base import HandStageBase


@HANDS.register("dynhamr", aliases=("dyn-hamr", "dyn_hamr"))
class DynHaMR(HandStageBase):
    name = "dynhamr"
    requires = ("torch",)
    extra = "dynhamr"
    license = "MIT (code) + MANO non-commercial"
    license_url = "https://github.com/ZhengdiYu/Dyn-HaMR"
    keypoint_convention = HandConvention.STANDARD21
    per_frame = False  # CLIP-level (4D world-frame optimization over all frames)

    def load(self) -> None:
        self.import_or_raise("torch")
        raise NotImplementedError(
            "Dyn-HaMR runs a multi-stage offline optimization (SLAM init -> per-"
            "frame HaMeR/WiLoR -> motion-prior fitting). Wire its `run_opt` entry "
            "point here; it produces world-frame MANO for both hands across the clip."
        )

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        # Clip-level: in dry-run, fall back to the per-frame synthetic hands so
        # downstream stages (retarget/export) have temporally-consistent input.
        if self.dry_run:
            num_hands = int(self.param("num_hands", 2))
            for fa, img in self.iter_frames(clip):
                fa.hands.extend(self._synthetic(num_hands, img, fa.frame_id, clip))
            return clip
        traj = self._run_world_optimization(clip)  # noqa: F841
        raise NotImplementedError(
            "Map Dyn-HaMR's world-frame MANO trajectory into per-frame HandPose "
            "(side=left/right, world keypoints_3d, MANO params)."
        )

    def _run_world_optimization(self, clip: ClipAnnotation):  # pragma: no cover
        raise NotImplementedError
