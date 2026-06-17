"""Base for hand-object interaction stages (100DOH detector, EgoHOS)."""

from __future__ import annotations

import numpy as np

from ...schema.core import (
    ClipAnnotation,
    ContactState,
    FrameAnnotation,
    HandObjectInteraction,
    HandSide,
)
from ..base import Stage


def hand_bbox_from_kp(kp2d: np.ndarray, pad: float = 0.15) -> list[float]:
    """Tight (padded) [x,y,w,h] around a hand's visible 2D keypoints."""
    pts = kp2d[kp2d[:, 2] > 0][:, :2] if kp2d.shape[1] >= 3 else kp2d[:, :2]
    if pts.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    return [
        float(x0 - pad * w),
        float(y0 - pad * h),
        float(w * (1 + 2 * pad)),
        float(h * (1 + 2 * pad)),
    ]


class HOIStageBase(Stage):
    family = "hoi"

    def infer(self, image: np.ndarray, fa: FrameAnnotation) -> list[HandObjectInteraction]:
        raise NotImplementedError  # pragma: no cover

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        for fa, img in self.iter_frames(clip):
            if self.dry_run or img is None:
                fa.interactions.extend(self._synthetic(fa))
            else:
                fa.interactions.extend(self.infer(img, fa))
        return clip

    def _synthetic(self, fa: FrameAnnotation) -> list[HandObjectInteraction]:
        out: list[HandObjectInteraction] = []
        # nearest detection (by box center) becomes the held object, if any.
        det_centers = [
            (i, (d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2))
            for i, d in enumerate(fa.detections)
        ]
        for hand in fa.hands:
            kp = hand.kp2d_array()
            hbox = hand_bbox_from_kp(kp)
            held_id = None
            held_box = None
            if det_centers:
                hx = hbox[0] + hbox[2] / 2
                hy = hbox[1] + hbox[3] / 2
                j, _ = min(det_centers, key=lambda c: (c[1][0] - hx) ** 2 + (c[1][1] - hy) ** 2)
                held_id = fa.detections[j].instance_id
                held_box = fa.detections[j].bbox
            out.append(
                HandObjectInteraction(
                    hand_side=HandSide(hand.side),
                    hand_bbox=hbox,
                    contact_state=ContactState.PORTABLE_OBJECT
                    if held_box
                    else ContactState.NO_CONTACT,
                    held_object_id=held_id,
                    held_object_bbox=held_box,
                    score=0.9,
                )
            )
        return out
