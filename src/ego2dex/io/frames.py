"""Unified frame source + in-memory frame store.

A *source* is either a video file or a directory of pre-extracted images. Both
yield ``(frame_id, timestamp_sec, image_bgr)``. The pipeline ingests a source
into a :class:`FrameStore` (frame_id -> image) that stages read from; pixels are
artifacts and never enter the JSON annotations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema.core import VideoMeta
from .video import VideoReader, is_video_file, probe_video

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


class ImageDirReader:
    """Iterate frames from a directory of images (sorted by filename)."""

    def __init__(
        self,
        path: str | Path,
        sample_fps: float | None = None,
        stride: int | None = None,
        max_frames: int | None = None,
        fps: float = 30.0,
    ) -> None:
        self.dir = Path(path)
        self.files = _list_images(self.dir)
        if not self.files:
            raise FileNotFoundError(f"No images found in directory: {path}")
        self.fps = fps
        self.stride = max(1, int(stride or 1))
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import cv2

        kept = 0
        for idx, f in enumerate(self.files):
            if idx % self.stride != 0:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield idx, idx / self.fps, img
            kept += 1
            if self.max_frames and kept >= self.max_frames:
                break


@dataclass
class FrameStore:
    """Holds decoded frames in memory + the source metadata.

    For very long videos this should be swapped for a lazy/streaming store; the
    interface (``get``, ``__len__``, ``frame_ids``) is intentionally minimal.
    """

    meta: VideoMeta
    _frames: dict[int, np.ndarray] = field(default_factory=dict)
    _timestamps: dict[int, float] = field(default_factory=dict)

    def add(self, frame_id: int, image: np.ndarray, timestamp: float) -> None:
        self._frames[frame_id] = image
        self._timestamps[frame_id] = timestamp

    def get(self, frame_id: int) -> np.ndarray | None:
        return self._frames.get(frame_id)

    def timestamp(self, frame_id: int) -> float | None:
        return self._timestamps.get(frame_id)

    def frame_ids(self) -> list[int]:
        return sorted(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


def open_source(
    input_path: str | Path,
    sample_fps: float | None = None,
    stride: int | None = None,
    max_frames: int | None = None,
    source_hint: str = "auto",
) -> tuple[VideoMeta, Iterator[tuple[int, float, np.ndarray]]]:
    """Open a video file or image directory; return ``(VideoMeta, frame_iter)``."""
    path = Path(input_path)
    if path.is_dir():
        reader = ImageDirReader(path, sample_fps=sample_fps, stride=stride, max_frames=max_frames)
        import cv2

        first = cv2.imread(str(reader.files[0]))
        h, w = (first.shape[0], first.shape[1]) if first is not None else (0, 0)
        meta = VideoMeta(
            path=str(path),
            fps=reader.fps,
            width=w,
            height=h,
            num_frames=len(reader.files),
            source=("aria" if source_hint == "aria" else "generic"),
            sampled_fps=reader.fps / reader.stride if reader.stride else reader.fps,
        )
        return meta, iter(reader)
    if is_video_file(path):
        info = probe_video(path)
        reader = VideoReader(path, sample_fps=sample_fps, stride=stride, max_frames=max_frames)
        source = "gopro" if source_hint in ("auto", "gopro") else source_hint
        meta = VideoMeta(
            path=str(path),
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            num_frames=info["num_frames"],
            source=source,
            codec=info["codec"],
            duration_sec=info["duration_sec"],
            sampled_fps=(info["fps"] / reader.stride) if info["fps"] else None,
        )
        return meta, iter(reader)
    raise FileNotFoundError(f"Input is neither a directory of images nor a video: {input_path}")


def load_into_store(
    input_path: str | Path,
    sample_fps: float | None = None,
    stride: int | None = None,
    max_frames: int | None = None,
    source_hint: str = "auto",
) -> FrameStore:
    """Decode a source fully into an in-memory :class:`FrameStore`."""
    meta, frames = open_source(
        input_path,
        sample_fps=sample_fps,
        stride=stride,
        max_frames=max_frames,
        source_hint=source_hint,
    )
    store = FrameStore(meta=meta)
    kept = 0
    for fid, ts, img in frames:
        store.add(fid, img, ts)
        kept += 1
    store.meta.num_frames = kept if kept else store.meta.num_frames
    return store
