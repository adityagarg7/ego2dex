"""Grounded-SAM-2: the recommended text -> boxes -> masks -> tracked-IDs chain.

https://github.com/IDEA-Research/Grounded-SAM-2 (Apache-2.0). This is the
integration to mirror: Grounding DINO converts a text prompt to boxes, SAM 2
turns boxes into masks and propagates IDs across the clip.

Registered under the ``detection`` family because it is prompt-driven; it
populates BOTH ``detections`` and ``masks`` (+ ``tracks``) on each frame.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import ClipAnnotation, Detection, FrameAnnotation, Mask, Track
from ..base import DETECTION
from .base import DetectionStageBase, SegmentationStageBase, prompt_for_frame
from .grounding_dino import GroundingDINO
from .sam2 import SAM2


@DETECTION.register("grounded_sam2", aliases=("grounded-sam-2", "gsam2"))
class GroundedSAM2(DetectionStageBase):
    name = "grounded_sam2"
    requires = ("torch", "transformers", "sam2")
    extra = "detection"
    license = "Apache-2.0 (GDINO + SAM2)"
    license_url = "https://github.com/IDEA-Research/Grounded-SAM-2"

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._gdino = GroundingDINO(**params)
        self._sam2 = SAM2(**params)

    def bind(self, ctx) -> None:
        super().bind(ctx)
        self._gdino.bind(ctx)
        self._sam2.bind(ctx)

    def load(self) -> None:
        self._gdino.ensure_loaded()
        self._sam2.ensure_loaded()

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        # Reuse the family bases' synthetic paths in dry-run for a full graph test.
        if self.dry_run:
            det_helper = _DryDet(**self.params)
            seg_helper = _DrySeg(**self.params)
            det_helper.bind(self.ctx)
            seg_helper.bind(self.ctx)
            clip = det_helper.process(clip)
            clip = seg_helper.process(clip)
            return clip

        label_to_id: dict[str, int] = {}
        next_id = 1
        for fa, img in self.iter_frames(clip):
            prompt = prompt_for_frame(self, fa)
            dets = self._gdino.infer(img, prompt, fa)
            fa.detections.extend(dets)
            masks = self._sam2.infer_masks(img, dets, fa)
            for det, m in zip(dets, masks, strict=False):
                if det.label not in label_to_id:
                    label_to_id[det.label] = next_id
                    next_id += 1
                m.instance_id = label_to_id[det.label]
                m.label = det.label
            fa.masks.extend(masks)
        for label, tid in label_to_id.items():
            clip.tracks.append(Track(instance_id=tid, label=label))
        return clip

    def infer(self, image: np.ndarray, prompt: list[str], fa: FrameAnnotation) -> list[Detection]:
        return self._gdino.infer(image, prompt, fa)


class _DryDet(DetectionStageBase):
    name = "grounded_sam2_det"


class _DrySeg(SegmentationStageBase):
    name = "grounded_sam2_seg"

    def infer_masks(self, image, dets, fa) -> list[Mask]:  # pragma: no cover
        raise NotImplementedError
