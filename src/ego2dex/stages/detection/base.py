"""Bases for open-vocab detection and segmentation/tracking stages.

Code for the whole detect+segment+track chain lives in this package (per the
repo layout): Grounding DINO / YOLO-World / Detic / OWLv2 (boxes) and
SAM2 / DEVA / SAMURAI (masks + temporal IDs), plus the Grounded-SAM-2 chain.

Open-vocabulary prompts come from (in priority order): the stage's ``classes``
param, the frame's RAM++ ``tags`` (auto-prompt), else a sensible default.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import ClipAnnotation, Detection, FrameAnnotation, Mask, Track
from ..base import Stage

DEFAULT_PROMPT = ["hand", "object"]


def prompt_for_frame(stage: Stage, fa: FrameAnnotation) -> list[str]:
    """Resolve the open-vocab class list for a frame."""
    classes = stage.param("classes")
    if classes:
        return list(classes)
    if fa.tags and fa.tags.tags:
        return list(fa.tags.tags)
    return list(DEFAULT_PROMPT)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
class DetectionStageBase(Stage):
    family = "detection"

    def infer(self, image: np.ndarray, prompt: list[str], fa: FrameAnnotation) -> list[Detection]:
        raise NotImplementedError  # pragma: no cover

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        for fa, img in self.iter_frames(clip):
            prompt = prompt_for_frame(self, fa)
            if self.dry_run or img is None:
                dets = self._synthetic(prompt, img, fa, clip)
            else:
                dets = self.infer(img, prompt, fa)
            fa.detections.extend(dets)
        return clip

    def _synthetic(
        self,
        prompt: list[str],
        image: np.ndarray | None,
        fa: FrameAnnotation,
        clip: ClipAnnotation,
    ) -> list[Detection]:
        h = image.shape[0] if image is not None else clip.video_meta.height or 256
        w = image.shape[1] if image is not None else clip.video_meta.width or 256
        dets: list[Detection] = []
        labels = [p for p in prompt if p != "hand"][:3] or ["object"]
        for i, label in enumerate(labels):
            bw, bh = 0.2 * w, 0.2 * h
            x = (0.2 + 0.25 * i) * w + ((fa.frame_id % 8) - 4) * 0.005 * w
            y = 0.55 * h
            dets.append(Detection(label=label, score=0.9, bbox=[x, y, bw, bh]))
        return dets


# --------------------------------------------------------------------------- #
# Segmentation / tracking
# --------------------------------------------------------------------------- #
class SegmentationStageBase(Stage):
    family = "segmentation"

    def infer_masks(
        self, image: np.ndarray, dets: list[Detection], fa: FrameAnnotation
    ) -> list[Mask]:
        raise NotImplementedError  # pragma: no cover

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        label_to_id: dict[str, int] = {}
        next_id = 1
        track_seen: dict[int, Track] = {}
        for fa, img in self.iter_frames(clip):
            # stable instance ids by label (a stand-in for SAM2 memory tracking)
            for det in fa.detections:
                if det.label not in label_to_id:
                    label_to_id[det.label] = next_id
                    next_id += 1
            if self.dry_run or img is None:
                masks = self._synthetic(fa, img, clip, label_to_id)
            else:
                masks = self.infer_masks(img, fa.detections, fa)
                for m in masks:
                    if m.label and m.instance_id is None:
                        m.instance_id = label_to_id.setdefault(m.label, len(label_to_id) + 1)
            fa.masks.extend(masks)
            for m in masks:
                if m.instance_id is None:
                    continue
                tr = track_seen.get(m.instance_id)
                if tr is None:
                    track_seen[m.instance_id] = Track(
                        instance_id=m.instance_id,
                        label=m.label or "object",
                        first_frame=fa.frame_id,
                        last_frame=fa.frame_id,
                        score=m.score,
                    )
                else:
                    tr.last_frame = fa.frame_id
        # merge tracks into the clip (de-dup by instance_id)
        existing = {t.instance_id for t in clip.tracks}
        for tid, tr in sorted(track_seen.items()):
            if tid not in existing:
                clip.tracks.append(tr)
        return clip

    def _synthetic(self, fa, image, clip, label_to_id) -> list[Mask]:
        h = image.shape[0] if image is not None else clip.video_meta.height or 256
        w = image.shape[1] if image is not None else clip.video_meta.width or 256
        masks: list[Mask] = []
        for det in fa.detections:
            x, y, bw, bh = [int(round(v)) for v in det.bbox]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + max(1, bw)), min(h, y + max(1, bh))
            if x1 <= x0 or y1 <= y0:
                continue
            binary = np.zeros((h, w), dtype=np.uint8)
            binary[y0:y1, x0:x1] = 1
            masks.append(
                Mask.from_binary(
                    binary,
                    instance_id=label_to_id.get(det.label),
                    label=det.label,
                    score=det.score,
                )
            )
        return masks
