"""Video probing + frame iteration (OpenCV/ffmpeg backend).

Keeps fps, timestamps, resolution; supports fps-based sampling, a fixed stride,
or a hard frame cap. GoPro ``.mp4`` works out of the box; ``.insv`` needs a
prior ffmpeg/Max-Export pass (see :mod:`ego2dex.io.gopro`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".insv"}


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def probe_video(path: str | Path) -> dict:
    """Return ``{fps, width, height, num_frames, codec, duration_sec}``."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip("\x00")
    finally:
        cap.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "codec": codec or None,
        "duration_sec": (num_frames / fps) if fps else None,
    }


class VideoReader:
    """Iterate ``(frame_id, timestamp_sec, image_bgr)`` with optional sampling.

    ``sample_fps`` keeps roughly that many frames/second; ``stride`` keeps every
    Nth frame; ``max_frames`` caps the total. ``frame_id`` is the original index
    in the source video (so timestamps stay correct).
    """

    def __init__(
        self,
        path: str | Path,
        sample_fps: float | None = None,
        stride: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.path = str(path)
        self.meta = probe_video(path)
        self.fps = self.meta["fps"]
        if sample_fps and self.fps:
            self.stride = max(1, int(round(self.fps / sample_fps)))
        else:
            self.stride = max(1, int(stride or 1))
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(self.path)
        kept = 0
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    yield idx, idx / self.fps if self.fps else float(idx), frame
                    kept += 1
                    if self.max_frames and kept >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()
