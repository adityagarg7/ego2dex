"""MediaPipe Hand Landmarker stage (lightweight, CPU, no MANO).

License: Apache-2.0. 21 landmarks per hand in the standard order (==OpenPose==
FreiHAND). Emits ``hand_landmarks`` (normalized image coords, wrist-relative z)
as pixel ``keypoints_2d`` and ``hand_world_landmarks`` (metric, hand-center
origin) as ``keypoints_3d`` -- the dependency-free default / smoke path.

Live path needs ``pip install 'ego2dex[mediapipe]'`` and the
``hand_landmarker.task`` model bundle (~7 MB):
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
Set its path via ``params.model_path`` (or ``EGO2DEX_MEDIAPIPE_TASK``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ...schema.core import HandPose, HandSide
from ...topology import NUM_HAND_KEYPOINTS, HandConvention
from ..base import HANDS
from .base import HandStageBase


@HANDS.register("mediapipe", aliases=("mp_hands",))
class MediaPipeHands(HandStageBase):
    name = "mediapipe"
    requires = ("mediapipe",)
    extra = "mediapipe"
    license = "Apache-2.0"
    license_url = "https://github.com/google-ai-edge/mediapipe"
    keypoint_convention = HandConvention.MEDIAPIPE

    def load(self) -> None:
        mp = self.import_or_raise("mediapipe")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = self.param("model_path") or os.environ.get("EGO2DEX_MEDIAPIPE_TASK")
        if not model_path or not Path(model_path).exists():
            raise ImportError(
                "MediaPipe HandLandmarker needs the 'hand_landmarker.task' bundle. "
                "Download it (see this module's docstring) and pass "
                "params.model_path=/path/to/hand_landmarker.task, or set "
                "EGO2DEX_MEDIAPIPE_TASK. (Or run with run.dry_run=true.)"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=int(self.param("num_hands", 2)),
            min_hand_detection_confidence=float(self.param("min_detection_confidence", 0.5)),
            min_hand_presence_confidence=float(self.param("min_presence_confidence", 0.5)),
            min_tracking_confidence=float(self.param("min_tracking_confidence", 0.5)),
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def infer(self, image: np.ndarray, frame_id: int) -> list[HandPose]:
        import cv2

        mp = self._mp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        hands: list[HandPose] = []
        for i, lms in enumerate(result.hand_landmarks):
            kp2d = np.array(
                [[lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0) or 1.0] for lm in lms],
                dtype=np.float64,
            )
            kp3d = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                wl = result.hand_world_landmarks[i]
                kp3d = np.array([[lm.x, lm.y, lm.z] for lm in wl], dtype=np.float64)
            side = HandSide.UNKNOWN
            score = 1.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                # MediaPipe handedness is from the image's POV (mirror of reality)
                side = HandSide.RIGHT if cat.category_name == "Right" else HandSide.LEFT
                score = float(cat.score)
            if kp2d.shape[0] != NUM_HAND_KEYPOINTS:
                continue
            hands.append(
                HandPose(
                    side=side,
                    keypoints_2d=kp2d,
                    keypoints_3d=kp3d,
                    keypoint_convention=self.keypoint_convention,
                    score=score,
                )
            )
        return hands
