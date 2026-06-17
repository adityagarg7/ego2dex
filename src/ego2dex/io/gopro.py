"""GoPro-specific ingestion helpers.

GoPro ``.mp4`` is a standard H.264/H.265 stream and works directly through
:mod:`ego2dex.io.video`. Two GoPro-isms are handled here:

  * ``.insv`` (MAX / 360) files must be stitched/flattened first (GoPro Player /
    ffmpeg). We detect them and raise an actionable error.
  * GoPro is wide-FOV (Wide/SuperView/HyperView). Intrinsics rarely sit in EXIF,
    so we expose an FOV-based pinhole guess and an undistortion path. Recover
    real intrinsics with SLAM/COLMAP (see :mod:`ego2dex.stages.pose`).

GPMF telemetry (gyro/accel/GPS) can be extracted with ``gpmf`` / ``exiftool``;
we stub the hook but do not depend on it.
"""

from __future__ import annotations

from pathlib import Path

from ..schema.core import CameraParams
from .camera import gopro_default_intrinsics
from .video import probe_video

# Rough horizontal FOV by GoPro digital lens (degrees), for the intrinsics guess.
GOPRO_FOV_PRESETS: dict[str, float] = {
    "narrow": 90.0,
    "linear": 92.0,
    "medium": 100.0,
    "wide": 118.0,
    "superview": 122.0,
    "hyperview": 130.0,
}


def is_insv(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".insv"


def ensure_flat_mp4(path: str | Path) -> Path:
    """Validate a GoPro input is a flat .mp4; reject 360 ``.insv`` with guidance."""
    p = Path(path)
    if is_insv(p):
        raise NotImplementedError(
            f"{p.name} is a 360 .insv file. Flatten/stitch it to an equirect or "
            "pinhole .mp4 first (GoPro Player export or `ffmpeg`), then re-run."
        )
    return p


def guess_intrinsics(path: str | Path, lens: str = "wide") -> CameraParams:
    """Best-effort pinhole intrinsics from frame size + a GoPro lens FOV preset."""
    info = probe_video(path)
    fov = GOPRO_FOV_PRESETS.get(lens.lower(), 118.0)
    return gopro_default_intrinsics(info["width"], info["height"], hfov_deg=fov)
