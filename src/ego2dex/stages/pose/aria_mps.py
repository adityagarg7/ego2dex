"""Project Aria MPS pose stage (no SLAM needed).

Reads MPS outputs and attaches per-frame Aria camera calibration (Fisheye624)
+ world->device extrinsics + gaze. Uses :mod:`ego2dex.io.aria` parsers (pure
python; no torch/GPU). Provide the MPS directory via ``params.mps_dir``.

Expected files: ``closed_loop_trajectory.csv``, ``online_calibration.jsonl``,
``semidense_points.csv.gz``, ``general_eye_gaze.csv``.
"""

from __future__ import annotations

from pathlib import Path

from ...schema.core import CameraModelType, CameraParams, ClipAnnotation, Gaze
from ...utils.geometry import invert_se3
from ..base import POSE
from .base import PoseStageBase


@POSE.register("aria_mps", aliases=("aria", "mps"))
class AriaMPS(PoseStageBase):
    name = "aria_mps"
    requires = ()  # pure-python CSV/JSONL parsing; projectaria-tools only for VRS
    extra = "aria"
    license = "Aria data license (see Project Aria terms)"
    license_url = "https://www.projectaria.com/datasets/"

    def estimate(self, clip: ClipAnnotation) -> None:
        from ...io.aria import (
            read_closed_loop_trajectory,
            read_general_eye_gaze,
            read_online_calibration,
        )

        mps_dir = self.param("mps_dir")
        if not mps_dir or not Path(mps_dir).exists():
            raise FileNotFoundError(
                "AriaMPS needs params.mps_dir pointing at the MPS output folder."
            )
        mps = Path(mps_dir)
        calib = self._maybe(read_online_calibration, mps / "online_calibration.jsonl")
        traj = self._maybe(read_closed_loop_trajectory, mps / "closed_loop_trajectory.csv") or []
        gaze = self._maybe(read_general_eye_gaze, mps / "general_eye_gaze.csv") or []

        rgb_cam = calib[0] if calib else None
        for fa in clip.frames:
            pose = self._nearest(traj, fa.timestamp) if fa.timestamp is not None else None
            cam = (
                rgb_cam.model_copy() if rgb_cam else CameraParams(model=CameraModelType.FISHEYE624)
            )
            if pose is not None:
                cam.extrinsics = invert_se3(pose["T_world_device"]).tolist()  # world->cam
                cam.extrinsics_direction = "world_to_cam"
            fa.camera = cam
            g = self._nearest(gaze, fa.timestamp) if fa.timestamp is not None else None
            if g is not None:
                fa.gaze = Gaze(depth_m=g.get("depth_m"))

    @staticmethod
    def _maybe(fn, path):
        try:
            return fn(path) if Path(path).exists() else None
        except Exception:
            return None

    @staticmethod
    def _nearest(records: list[dict], ts: float):
        if not records:
            return None
        return min(records, key=lambda r: abs(r.get("timestamp_s", 0.0) - ts))
