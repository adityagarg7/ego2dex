"""DEVA: open-world long-clip multi-object ID consistency.

Tracking Anything with Decoupled Video Segmentation (ICCV 2023, arXiv:2309.03903).
License: GPLv3 (NON-permissive). Decouples per-frame segmentation (e.g.
Grounded-SAM) from a temporal propagation/association model for consistent IDs
over long egocentric clips. Wired, not bundled.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Detection, FrameAnnotation, Mask
from ..base import SEGMENTATION
from .base import SegmentationStageBase


@SEGMENTATION.register("deva")
class DEVA(SegmentationStageBase):
    name = "deva"
    requires = ("torch",)
    extra = "deva"
    license = "GPL-3.0"
    license_url = "https://github.com/hkchengrex/Tracking-Anything-with-DEVA"

    def load(self) -> None:
        self.import_or_raise("torch")
        raise NotImplementedError(
            "Wire DEVA here: feed per-frame Grounded-SAM(2) proposals into DEVA's "
            "decoupled propagation to get globally-consistent instance IDs over the "
            "clip; emit Mask(instance_id=...) + Track entries."
        )

    def infer_masks(
        self, image: np.ndarray, dets: list[Detection], fa: FrameAnnotation
    ) -> list[Mask]:
        raise NotImplementedError  # pragma: no cover
