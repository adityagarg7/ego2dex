"""Base for caption/VLM stages (Qwen2.5-VL, Florence-2, Molmo)."""

from __future__ import annotations

import numpy as np

from ...schema.core import Caption, ClipAnnotation, FrameAnnotation
from ..base import Stage


class CaptionStageBase(Stage):
    family = "caption"

    def infer(self, image: np.ndarray, fa: FrameAnnotation) -> Caption:
        raise NotImplementedError  # pragma: no cover

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        for fa, img in self.iter_frames(clip):
            if self.dry_run or img is None:
                fa.caption = self._synthetic(fa)
            else:
                fa.caption = self.infer(img, fa)
        return clip

    def _synthetic(self, fa: FrameAnnotation) -> Caption:
        nouns = [t for t in (fa.tags.tags if fa.tags else []) if t != "hand"][:2]
        obj = nouns[0] if nouns else "an object"
        # a deterministic contact point at the right-hand index fingertip (kp 8)
        points = []
        for hand in fa.hands:
            kp = hand.kp2d_array()
            points.append({"x": float(kp[8, 0]), "y": float(kp[8, 1]), "label": "index_tip"})
        return Caption(
            frame_caption=f"A person's hand interacting with {obj}.",
            action=f"manipulate {obj}",
            region_captions=[{"label": n, "text": n} for n in nouns],
            points=points,
            source=f"{self.name}(dry-run)",
        )
