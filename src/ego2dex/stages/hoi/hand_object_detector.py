"""100DOH hand_object_detector stage.

"Understanding Human Hands in Contact at Internet Scale" (CVPR 2020,
arXiv:2006.06669). Per hand: box, side (L/R), **contact state** in
{no-contact, self, other-person, portable-object, stationary-object}, and the
held-object box. License: research-only (Faster R-CNN / 100DOH).

Repo: https://github.com/ddshan/hand_object_detector (needs a CUDA build of the
custom RoI ops). Wired, not bundled.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import ContactState, FrameAnnotation, HandObjectInteraction
from ..base import HOI
from .base import HOIStageBase

# 100DOH contact-state index -> our enum (order from the original repo).
CONTACT_INDEX = {
    0: ContactState.NO_CONTACT,
    1: ContactState.SELF_CONTACT,
    2: ContactState.OTHER_PERSON,
    3: ContactState.PORTABLE_OBJECT,
    4: ContactState.STATIONARY_OBJECT,
}


@HOI.register("hand_object_detector", aliases=("100doh", "hod"))
class HandObjectDetector(HOIStageBase):
    name = "hand_object_detector"
    requires = ("torch",)
    extra = "hoi"
    license = "Research-only (100DOH / Faster R-CNN)"
    license_url = "https://github.com/ddshan/hand_object_detector"

    def load(self) -> None:
        self.import_or_raise("torch")
        raise NotImplementedError(
            "Wire the 100DOH Faster R-CNN here (res101 handobj_100K weights). It "
            "emits per-hand: box, lr (L/R), state (5-class contact), and an "
            "offset vector to the held object's box. Map state via CONTACT_INDEX."
        )

    def infer(self, image: np.ndarray, fa: FrameAnnotation) -> list[HandObjectInteraction]:
        raise NotImplementedError  # pragma: no cover
