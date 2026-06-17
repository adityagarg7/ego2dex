"""EgoHOS stage: pixel masks of hands + interacted objects + contact boundaries.

ECCV 2022, arXiv:2208.03826. License: MIT. Egocentric-specific segmentation:
left/right hand masks, the directly/indirectly interacted object masks, and the
contact boundary. We store the contact region as ``HandObjectInteraction.
contact_mask`` (RLE). Repo: https://github.com/owenzlz/EgoHOS (mmsegmentation).
"""

from __future__ import annotations

import numpy as np

from ...schema.core import FrameAnnotation, HandObjectInteraction
from ..base import HOI
from .base import HOIStageBase, hand_bbox_from_kp


@HOI.register("egohos")
class EgoHOS(HOIStageBase):
    name = "egohos"
    requires = ("torch", "mmseg")
    extra = "hoi"
    license = "MIT"
    license_url = "https://github.com/owenzlz/EgoHOS"

    def load(self) -> None:
        self.import_or_raise("torch")
        raise NotImplementedError(
            "Wire EgoHOS here: load the mmsegmentation config + checkpoint and run "
            "inference to get {left_hand, right_hand, obj1/2, cb} masks; convert "
            "the hand+contact regions into HandObjectInteraction.contact_mask (RLE)."
        )

    def infer(self, image: np.ndarray, fa: FrameAnnotation) -> list[HandObjectInteraction]:
        raise NotImplementedError  # pragma: no cover

    # Provided helper so downstream code can build the interaction once EgoHOS
    # masks are available, keeping the contact mask + hand box consistent.
    @staticmethod
    def interaction_from_masks(hand_mask: np.ndarray, contact_mask: np.ndarray, side, kp2d):
        from ...schema.core import Mask

        return HandObjectInteraction(
            hand_side=side,
            hand_bbox=hand_bbox_from_kp(np.asarray(kp2d)),
            contact_mask=Mask.from_binary(contact_mask, label="contact"),
        )
