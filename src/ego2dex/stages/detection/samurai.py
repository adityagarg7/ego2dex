"""SAMURAI: motion-aware SAM2 for robust single-object tracking.

arXiv:2411.11922. License: Apache-2.0. Adds a Kalman-filter motion model + a
motion-aware memory selection to SAM2 for robustness under the fast motion and
occlusion typical of egocentric video. Best for a single tracked object; wired,
not bundled.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Detection, FrameAnnotation, Mask
from ..base import SEGMENTATION
from .base import SegmentationStageBase


@SEGMENTATION.register("samurai")
class SAMURAI(SegmentationStageBase):
    name = "samurai"
    requires = ("torch", "sam2")
    extra = "samurai"
    license = "Apache-2.0"
    license_url = "https://github.com/yangchris11/samurai"

    def load(self) -> None:
        self.import_or_raise("torch")
        raise NotImplementedError(
            "Wire SAMURAI here: build its motion-aware SAM2 video predictor, "
            "initialize from the first-frame box, and propagate with the Kalman "
            "motion model for occlusion-robust single-object egocentric tracking."
        )

    def infer_masks(
        self, image: np.ndarray, dets: list[Detection], fa: FrameAnnotation
    ) -> list[Mask]:
        raise NotImplementedError  # pragma: no cover
