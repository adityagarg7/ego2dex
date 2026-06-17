"""WildHands hand-pose stage (egocentric-specialized, single image).

ECCV 2024, arXiv:2312.06583. Camera-intrinsics keypoint embedding; SOTA on the
ARCTIC ego split, strong zero-shot on H2O / AssemblyHands / EPIC / EgoExo4D.

License: CC-BY-NC (NON-commercial) + MANO. Install from source:
    git clone https://github.com/ap229997/hands
"""

from __future__ import annotations

import numpy as np

from ...schema.core import HandPose
from ...topology import HandConvention
from ..base import HANDS
from .base import HandStageBase


@HANDS.register("wildhands")
class WildHands(HandStageBase):
    name = "wildhands"
    requires = ("torch",)
    extra = "wildhands"
    license = "CC-BY-NC (research only) + MANO non-commercial"
    license_url = "https://github.com/ap229997/hands"
    keypoint_convention = HandConvention.STANDARD21

    def load(self) -> None:
        self.import_or_raise("torch")
        raise NotImplementedError(
            "Wire WildHands here. It embeds camera intrinsics into the keypoint "
            "head, so pass the per-frame CameraParams.K() into the model input."
        )

    def infer(self, image: np.ndarray, frame_id: int) -> list[HandPose]:  # pragma: no cover
        raise NotImplementedError(
            "Run WildHands forward on the hand crop + intrinsics; map its MANO + "
            "21 keypoints into HandPose (camera-intrinsics-aware egocentric model)."
        )
