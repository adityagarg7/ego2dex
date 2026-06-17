"""EgoDex-style HDF5 export (imitation / retargeted-action pretraining).

Mirrors the EgoDex / robomimic layout (arXiv:2505.11709): per-frame camera
intrinsics + extrinsics, per-hand 3D keypoints + MANO + per-joint confidence,
the retargeted robot joint trajectory, and language task strings. Missing values
are NaN-filled so the arrays stay dense ``(T, ...)``.

Needs ``h5py`` (the ``[export]`` extra).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from ..io.camera import build_camera
from ..schema.core import ClipAnnotation, HandSide
from ..stages.base import EXPORT
from .base import ExportStageBase


@EXPORT.register("hdf5", aliases=("h5", "egodex"))
class HDF5Writer(ExportStageBase):
    name = "hdf5"
    requires = ("h5py",)
    extra = "export"
    license = "MIT (ego2dex)"

    def write(self, clip: ClipAnnotation) -> Path:
        h5py = self.import_or_raise("h5py")
        out = self.out_dir()
        path = out / self.param("filename", "episode.hdf5")

        frames = clip.frames
        T = len(frames)
        fids = [fa.frame_id for fa in frames]

        intr = np.full((T, 3, 3), np.nan)
        extr = np.full((T, 4, 4), np.nan)
        for i, fa in enumerate(frames):
            if fa.camera is not None:
                with contextlib.suppress(Exception):
                    intr[i] = build_camera(fa.camera).K()
                if fa.camera.extrinsics is not None:
                    extr[i] = np.asarray(fa.camera.extrinsics)

        with h5py.File(path, "w") as f:
            f.attrs["fps"] = clip.video_meta.sampled_fps or clip.video_meta.fps
            f.attrs["num_frames"] = T
            f.attrs["source"] = clip.video_meta.source
            f.attrs["schema_version"] = clip.schema_version
            f.attrs["language"] = self._language(clip)
            f.create_dataset("frame_ids", data=np.asarray(fids, dtype=np.int64))

            cam = f.create_group("camera")
            cam.create_dataset("intrinsics", data=intr)
            cam.create_dataset("extrinsics", data=extr)

            hands = f.create_group("hands")
            for side in (HandSide.LEFT, HandSide.RIGHT):
                g = hands.create_group(side.value)
                kp3d, conf, pose, betas = self._hand_arrays(frames, side, T)
                g.create_dataset("keypoints_3d", data=kp3d)
                g.create_dataset("confidence", data=conf)
                g.create_dataset("mano_pose", data=pose)  # 48 = 3 global + 45
                g.create_dataset("mano_betas", data=betas)

            if clip.retargeting:
                rg = f.create_group("robot")
                for r in clip.retargeting:
                    side = r.hand_side if isinstance(r.hand_side, str) else r.hand_side.value
                    d = rg.create_dataset(f"{r.robot}_{side}", data=r.trajectory_array())
                    d.attrs["joint_names"] = [str(j) for j in r.joint_names]
                    d.attrs["optimizer"] = r.optimizer
                    d.attrs["frame_ids"] = np.asarray(r.frame_ids, dtype=np.int64)
        return path

    # ------------------------------------------------------------------ #
    def _hand_arrays(self, frames, side: HandSide, T: int):
        kp3d = np.full((T, 21, 3), np.nan)
        conf = np.full((T, 21), np.nan)
        pose = np.full((T, 48), np.nan)
        betas = np.full((T, 10), np.nan)
        for i, fa in enumerate(frames):
            for hand in fa.hands:
                if HandSide(hand.side) != side:
                    continue
                conf[i] = hand.kp2d_array()[:, 2]
                if hand.keypoints_3d is not None:
                    kp3d[i] = hand.kp3d_array()
                if hand.mano is not None:
                    pose[i] = np.asarray(hand.mano.full_pose())[:48]
                    betas[i] = np.asarray(hand.mano.betas)
                break
        return kp3d, conf, pose, betas

    def _language(self, clip: ClipAnnotation) -> str:
        if self.param("language"):
            return str(self.param("language"))
        texts = [s.text for s in clip.action_segments if s.text]
        if texts:
            return " ; ".join(texts)
        caps = [fa.caption.action for fa in clip.frames if fa.caption and fa.caption.action]
        return caps[0] if caps else ""
