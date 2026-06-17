"""Base class for 3D hand-pose stages (HaMeR / WiLoR / MediaPipe / ...).

Concrete stages override :meth:`infer`. The base owns frame iteration, the
deterministic **dry-run** path (synthetic but topologically valid 21-keypoint
hands so the whole graph runs with no weights), and convention bookkeeping.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import HandPose, HandSide, MANOParams
from ...topology import NUM_HAND_KEYPOINTS, HandConvention
from ..base import ClipAnnotation, Stage

# A canonical, spread right-hand layout in a unit square (x right, y down).
# Used only to synthesize deterministic hands for dry-run / tests / viz.
_CANONICAL_HAND_2D: np.ndarray = np.array(
    [
        [0.50, 0.95],  # 0 wrist
        [0.30, 0.85],
        [0.22, 0.72],
        [0.16, 0.60],
        [0.12, 0.50],  # thumb
        [0.40, 0.55],
        [0.39, 0.38],
        [0.385, 0.27],
        [0.38, 0.18],  # index
        [0.50, 0.52],
        [0.50, 0.33],
        [0.50, 0.21],
        [0.50, 0.10],  # middle
        [0.60, 0.55],
        [0.61, 0.36],
        [0.615, 0.25],
        [0.62, 0.16],  # ring
        [0.70, 0.60],
        [0.72, 0.46],
        [0.73, 0.37],
        [0.74, 0.30],  # pinky
    ],
    dtype=np.float64,
)


def canonical_hand_3d(scale: float = 0.09) -> np.ndarray:
    """A wrist-origin metric-ish 21x3 hand (meters), for synthetic outputs."""
    xy = (_CANONICAL_HAND_2D - _CANONICAL_HAND_2D[0]) * np.array([1.0, -1.0])
    z = np.linspace(0.0, -0.02, NUM_HAND_KEYPOINTS)  # gentle curl toward camera
    pts = np.concatenate([xy * scale, z[:, None]], axis=1)
    return pts.astype(np.float64)


class HandStageBase(Stage):
    family = "hands"
    keypoint_convention: HandConvention = HandConvention.STANDARD21

    def infer(self, image: np.ndarray, frame_id: int) -> list[HandPose]:  # pragma: no cover
        """Run the real model on one BGR image -> list of HandPose. Override me."""
        raise NotImplementedError(
            f"{self.cls_name()}.infer() is not wired to weights yet. "
            "Run with run.dry_run=true for the synthetic path, or install the "
            "model extra + weights (see docs/install.md) and implement the call."
        )

    # ------------------------------------------------------------------ #
    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        num_hands = int(self.param("num_hands", 2))
        for fa, img in self.iter_frames(clip):
            if self.dry_run or img is None:
                hands = self._synthetic(num_hands, img, fa.frame_id, clip)
            else:
                hands = self.infer(img, fa.frame_id)
            fa.hands.extend(hands)
        return clip

    # ------------------------------------------------------------------ #
    def _synthetic(
        self, num_hands: int, image: np.ndarray | None, frame_id: int, clip: ClipAnnotation
    ) -> list[HandPose]:
        h = image.shape[0] if image is not None else clip.video_meta.height or 256
        w = image.shape[1] if image is not None else clip.video_meta.width or 256
        out: list[HandPose] = []
        sides = [HandSide.RIGHT, HandSide.LEFT][:num_hands]
        # gentle, deterministic horizontal drift across frames for tracking demos
        drift = ((frame_id % 10) - 5) * 0.01
        for i, side in enumerate(sides):
            cx = (0.35 + 0.3 * i) + drift
            bw, bh = 0.22 * w, 0.30 * h
            tmpl = _CANONICAL_HAND_2D.copy()
            if side == HandSide.LEFT:
                tmpl[:, 0] = 1.0 - tmpl[:, 0]  # mirror
            kp2d = np.empty((NUM_HAND_KEYPOINTS, 3))
            kp2d[:, 0] = cx * w + (tmpl[:, 0] - 0.5) * bw
            kp2d[:, 1] = 0.45 * h + (tmpl[:, 1] - 0.5) * bh
            kp2d[:, 2] = 0.95
            kp3d = canonical_hand_3d() + np.array([0.1 * (i - 0.5), 0.0, 0.4])
            mano = MANOParams(
                global_orient=[0.0, 0.0, 0.0],
                pose=[0.0] * 45,
                betas=[0.0] * 10,
                trans=kp3d[0].tolist(),
            )
            out.append(
                HandPose(
                    side=side,
                    keypoints_2d=kp2d,
                    keypoints_3d=kp3d,
                    mano=mano,
                    keypoint_convention=self.keypoint_convention,
                    score=0.99,
                )
            )
        return out
