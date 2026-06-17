"""Temporal smoothing sub-stage for per-frame hand estimates.

Two modes:
  * ``oneeuro`` -- the 1€ (One-Euro) filter. Pure-python, real-time, no weights;
    fully implemented and used by default (works in the CPU smoke pipeline).
  * ``smoothnet`` -- SmoothNet (ECCV 2022, cure-lab/SmoothNet), a learned window
    smoother for batch refinement; needs torch + weights (wired, not bundled).

Clip-level: it rewrites ``keypoints_2d`` / ``keypoints_3d`` of existing hands.
"""

from __future__ import annotations

import math

import numpy as np

from ...schema.core import ClipAnnotation, HandSide
from ..base import HANDS, Stage


def one_euro_filter(
    signal: np.ndarray,
    freq: float = 30.0,
    mincutoff: float = 1.0,
    beta: float = 0.3,
    dcutoff: float = 1.0,
) -> np.ndarray:
    """Apply a 1€ filter along axis 0 of ``signal`` (shape ``(T, ...)``)."""
    sig = np.asarray(signal, dtype=np.float64)
    if sig.shape[0] < 2:
        return sig

    def _alpha(cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    out = np.empty_like(sig)
    out[0] = sig[0]
    dx_prev = np.zeros_like(sig[0])
    x_prev = sig[0]
    for t in range(1, sig.shape[0]):
        dx = (sig[t] - x_prev) * freq
        a_d = _alpha(dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = mincutoff + beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) * freq)
        x_hat = a * sig[t] + (1 - a) * x_prev
        out[t] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat
    return out


@HANDS.register("smoothing", aliases=("smooth", "oneeuro"))
class HandSmoothing(Stage):
    name = "smoothing"
    family = "hands"
    license = "MIT (1€ filter / SmoothNet)"
    license_url = "https://github.com/cure-lab/SmoothNet"
    per_frame = False

    def load(self) -> None:
        if self.param("method", "oneeuro") == "smoothnet" and not self.dry_run:
            self.import_or_raise("torch")
            raise NotImplementedError(
                "Wire SmoothNet here: load the pretrained window model and run it "
                "over the (T, 21*3) keypoint sequence. Use method='oneeuro' for a "
                "weightless real-time alternative."
            )

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        freq = clip.video_meta.sampled_fps or clip.video_meta.fps or 30.0
        mincutoff = float(self.param("mincutoff", 1.0))
        beta = float(self.param("beta", 0.3))
        for side in (HandSide.LEFT, HandSide.RIGHT):
            self._smooth_side(clip, side, freq, mincutoff, beta)
        return clip

    def _smooth_side(self, clip, side, freq, mincutoff, beta) -> None:
        seq = []  # (frame_index_in_clip, HandPose)
        for fi, fa in enumerate(clip.frames):
            for hand in fa.hands:
                if HandSide(hand.side) == side:
                    seq.append((fi, hand))
                    break
        if len(seq) < 3:
            return
        kp2d = np.stack([h.kp2d_array() for _, h in seq], axis=0)  # (T,21,3)
        sm2d = one_euro_filter(kp2d[..., :2], freq, mincutoff, beta)
        kp2d[..., :2] = sm2d
        have3d = all(h.keypoints_3d is not None for _, h in seq)
        if have3d:
            kp3d = np.stack([h.kp3d_array() for _, h in seq], axis=0)  # type: ignore[union-attr]
            kp3d = one_euro_filter(kp3d, freq, mincutoff, beta)
        for i, (_, hand) in enumerate(seq):
            hand.keypoints_2d = kp2d[i].tolist()
            if have3d:
                hand.keypoints_3d = kp3d[i].tolist()
