"""Project Aria ingestion: ``.vrs`` RGB stream + MPS outputs.

All ``projectaria_tools`` imports are lazy (it is the ``[aria]`` extra). Stream
and MPS-file identifiers follow the Aria conventions in the task spec:

  * RGB stream ``214-1``; SLAM cameras ``1201-1`` / ``1201-2``; eye ``211-1``.
  * MPS files: ``closed_loop_trajectory.csv``, ``semidense_points.csv.gz``,
    ``online_calibration.jsonl``, ``general_eye_gaze.csv``.

Camera calibration from Aria is the Fisheye624 / RadTanThinPrism model -- see
:class:`ego2dex.io.camera.Fisheye624Camera`.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..schema.core import CameraModelType, CameraParams
from ..utils.geometry import make_se3, quat_to_rotmat

ARIA_RGB_STREAM = "214-1"
ARIA_SLAM_STREAMS = ("1201-1", "1201-2")
ARIA_EYE_STREAM = "211-1"

_ARIA_EXTRA_HINT = (
    "Project Aria ingestion needs `projectaria-tools`. "
    "Install with: pip install 'ego2dex[aria]'  (or `pip install projectaria-tools`)."
)


def _require_aria() -> Any:
    try:
        import projectaria_tools.core as aria_core

        return aria_core
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(_ARIA_EXTRA_HINT) from e


def iter_rgb_frames(
    vrs_path: str | Path, stream: str = ARIA_RGB_STREAM
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_id, timestamp_sec, rgb_image)`` from the Aria RGB stream."""
    aria_core = _require_aria()  # noqa: F841 (kept for an explicit error)
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise FileNotFoundError(f"Could not open VRS: {vrs_path}")
    sid = StreamId(stream)
    n = provider.get_num_data(sid)
    for i in range(n):
        img_data = provider.get_image_data_by_index(sid, i)
        frame = img_data[0].to_numpy_array()
        ts_ns = img_data[1].capture_timestamp_ns
        yield i, ts_ns * 1e-9, frame


def read_online_calibration(jsonl_path: str | Path) -> list[CameraParams]:
    """Parse ``online_calibration.jsonl`` -> per-record RGB :class:`CameraParams`.

    Aria stores the RGB camera as a Fisheye624 (15 params). We surface the
    intrinsics + 12 distortion coeffs in our schema's Fisheye624 layout.
    """
    import json

    out: list[CameraParams] = []
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cams = rec.get("CameraCalibrations", [])
        for cam in cams:
            if "camera-rgb" not in cam.get("Label", ""):
                continue
            proj = cam.get("Projection", {})
            params = proj.get("Params", [])
            if len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                out.append(
                    CameraParams(
                        model=CameraModelType.FISHEYE624,
                        intrinsics={"f": float(f), "cx": float(cx), "cy": float(cy)},
                        distortion=[float(x) for x in params[3:15]],
                    )
                )
    return out


def read_closed_loop_trajectory(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse MPS ``closed_loop_trajectory.csv`` -> list of pose records.

    Each record carries ``timestamp_s`` and a world->device SE(3) (``T``), built
    from the translation + Hamilton quaternion columns.
    """
    records: list[dict[str, Any]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                t = np.array(
                    [
                        float(row["tx_world_device"]),
                        float(row["ty_world_device"]),
                        float(row["tz_world_device"]),
                    ]
                )
                q = np.array(
                    [
                        float(row["qw_world_device"]),
                        float(row["qx_world_device"]),
                        float(row["qy_world_device"]),
                        float(row["qz_world_device"]),
                    ]
                )
                ts = float(row.get("tracking_timestamp_us", row.get("utc_timestamp_ns", 0)))
                records.append(
                    {
                        "timestamp_s": ts * (1e-6 if "us" in str(row.keys()) else 1e-9),
                        "T_world_device": make_se3(quat_to_rotmat(q), t),
                    }
                )
            except (KeyError, ValueError):
                continue
    return records


def read_semidense_points(csv_gz_path: str | Path) -> np.ndarray:
    """Parse MPS ``semidense_points.csv.gz`` -> ``(N, 3)`` world point cloud."""
    pts: list[list[float]] = []
    with gzip.open(csv_gz_path, "rt") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                pts.append([float(row["px_world"]), float(row["py_world"]), float(row["pz_world"])])
            except (KeyError, ValueError):
                continue
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3))


def read_general_eye_gaze(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse MPS ``general_eye_gaze.csv`` -> per-timestamp yaw/pitch (+depth)."""
    out: list[dict[str, Any]] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append(
                    {
                        "timestamp_s": float(row["tracking_timestamp_us"]) * 1e-6,
                        "yaw_rad": float(row["yaw_rads_cpf"]),
                        "pitch_rad": float(row["pitch_rads_cpf"]),
                        "depth_m": float(row.get("depth_m", "nan")),
                    }
                )
            except (KeyError, ValueError):
                continue
    return out
