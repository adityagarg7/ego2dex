#!/usr/bin/env python
"""Generate a tiny synthetic egocentric clip into assets/synthetic/.

These few small frames let the smoke pipeline run end-to-end on CPU with no
real video. Regenerate with: `python scripts/make_synthetic.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "assets" / "synthetic"
W, H, N = 160, 120, 8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(N):
        img = np.full((H, W, 3), 40, dtype=np.uint8)
        # a moving "object" rectangle + a "skin" blob so frames aren't blank.
        # Deterministic (no noise) -> tiny PNGs + reproducible smoke runs.
        x = 20 + i * 8
        cv2.rectangle(img, (x, 60), (x + 30, 90), (90, 140, 200), -1)
        cv2.circle(img, (60 + i * 4, 70), 18, (120, 170, 210), -1)
        cv2.imwrite(str(OUT / f"{i:06d}.png"), img)
    print(f"wrote {N} frames to {OUT}")


if __name__ == "__main__":
    main()
