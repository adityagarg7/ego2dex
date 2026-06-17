"""Draw ego2dex annotations onto frames (hands, boxes, masks, caption).

Uses OpenCV only (a core dep); writes annotated images to disk (no GUI). Doubles
as a pipeline stage (``viz/overlay``) and a function for the ``ego2dex viz`` CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema.core import ClipAnnotation, FrameAnnotation, HandSide
from ..stages.base import VIZ, Stage
from ..topology import HAND_EDGES

_SIDE_COLOR = {
    HandSide.LEFT: (255, 120, 0),  # BGR -> blue-ish
    HandSide.RIGHT: (0, 160, 255),  # orange-ish
    HandSide.UNKNOWN: (0, 255, 0),
}


def draw_frame(image: np.ndarray, fa: FrameAnnotation, alpha: float = 0.4) -> np.ndarray:
    """Return a copy of ``image`` with the frame's annotations drawn on it."""
    import cv2

    canvas = image.copy()
    overlay = image.copy()

    # masks (filled, blended)
    rng = np.random.default_rng(7)
    for m in fa.masks:
        try:
            binary = m.decode().astype(bool)
        except Exception:
            continue
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        overlay[binary] = color
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # detection boxes
    for d in fa.detections:
        x, y, w, h = [int(round(v)) for v in d.bbox]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{d.label} {d.score:.2f}",
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # hands (skeleton)
    for hand in fa.hands:
        color = _SIDE_COLOR.get(HandSide(hand.side), (0, 255, 0))
        kp = hand.kp2d_array()
        for a, b in HAND_EDGES:
            pa, pb = kp[a], kp[b]
            if pa[2] > 0 and pb[2] > 0:
                cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
        for x, y, c in kp:
            if c > 0:
                cv2.circle(canvas, (int(x), int(y)), 3, color, -1)

    # caption banner
    if fa.caption and fa.caption.frame_caption:
        cv2.putText(
            canvas,
            fa.caption.frame_caption[:80],
            (8, canvas.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


@VIZ.register("overlay", aliases=("viz",))
class VizOverlay(Stage):
    name = "overlay"
    family = "viz"
    license = "MIT (ego2dex)"
    per_frame = True

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        import cv2

        out = Path(self.param("output_dir") or (self.ctx.output_dir / "viz"))
        out.mkdir(parents=True, exist_ok=True)
        for fa, img in self.iter_frames(clip):
            if img is None:
                continue
            annotated = draw_frame(img, fa)
            cv2.imwrite(str(out / f"{fa.frame_id:06d}.jpg"), annotated)
        self.ctx.extras.setdefault("exports", []).append(str(out))
        return clip
