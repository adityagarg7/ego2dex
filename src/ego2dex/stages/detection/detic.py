"""Detic detection (21k-class, box + mask).

Detecting Twenty-thousand Classes using Image-level Supervision (ECCV 2022).
License: Apache-2.0. Built on detectron2; supports a custom vocabulary via CLIP
text embeddings. Heavy detectron2 setup -> wired, not bundled.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Detection, FrameAnnotation
from ..base import DETECTION
from .base import DetectionStageBase


@DETECTION.register("detic")
class Detic(DetectionStageBase):
    name = "detic"
    requires = ("torch", "detectron2")
    extra = "detection"
    license = "Apache-2.0"
    license_url = "https://github.com/facebookresearch/Detic"

    def load(self) -> None:
        self.import_or_raise("torch")
        self.import_or_raise("detectron2")
        raise NotImplementedError(
            "Wire Detic here: build the detectron2 predictor from a Detic config, "
            "set a custom vocabulary via CLIP text classifier weights, then run "
            "predictor(image). Map instances.pred_boxes/pred_classes/scores and "
            "pred_masks into Detection + Mask."
        )

    def infer(self, image: np.ndarray, prompt: list[str], fa: FrameAnnotation) -> list[Detection]:
        raise NotImplementedError  # pragma: no cover
