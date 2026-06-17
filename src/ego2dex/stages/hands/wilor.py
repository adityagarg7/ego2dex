"""WiLoR hand-pose stage (real-time / multi-hand).

arXiv:2409.12259. End-to-end detect+reconstruct, multi-hand, >130 FPS detector /
~156 FPS refiner. Outputs MANO + 21 joints like HaMeR.

License: models CC-BY-NC-ND (NON-commercial) + Ultralytics AGPL + MANO. Weights
on HuggingFace. Easiest install is the community mini-package:
    pip install git+https://github.com/warmshao/WiLoR-mini
"""

from __future__ import annotations

import numpy as np

from ...schema.core import HandPose, HandSide, MANOParams
from ...topology import HandConvention
from ..base import HANDS
from .base import HandStageBase


@HANDS.register("wilor")
class WiLoR(HandStageBase):
    name = "wilor"
    requires = ("torch",)
    extra = "wilor"
    license = "CC-BY-NC-ND (models) + AGPL (detector) + MANO non-commercial"
    license_url = "https://github.com/rolpotamias/WiLoR"
    keypoint_convention = HandConvention.STANDARD21

    def load(self) -> None:
        torch = self.import_or_raise("torch")
        self._torch = torch
        self.device = self.param("device", self.ctx.device)
        # Preferred: WiLoR-mini pipeline (bundles detector + refiner + weights via HF).
        try:
            from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
                WiLorHandPose3dEstimationPipeline,
            )

            dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
            self._pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=dtype)
        except ImportError as e:
            raise ImportError(
                "WiLoR live path needs WiLoR-mini: "
                "pip install git+https://github.com/warmshao/WiLoR-mini "
                "(or wire the upstream rolpotamias/WiLoR repo)."
            ) from e

    def infer(self, image: np.ndarray, frame_id: int) -> list[HandPose]:  # pragma: no cover
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        outputs = self._pipe.predict(rgb)  # list of dicts, one per detected hand
        hands: list[HandPose] = []
        for o in outputs:
            wilor = o["wilor_preds"]
            kp2d_xy = np.asarray(wilor["pred_keypoints_2d"]).reshape(-1, 2)
            kp2d = np.concatenate([kp2d_xy, np.ones((kp2d_xy.shape[0], 1))], axis=1)
            kp3d = np.asarray(wilor["pred_keypoints_3d"]).reshape(-1, 3)
            mano = MANOParams(
                global_orient=np.asarray(wilor["global_orient"]).reshape(-1)[:3].tolist(),
                pose=np.asarray(wilor["hand_pose"]).reshape(-1)[:45].tolist(),
                betas=np.asarray(wilor["betas"]).reshape(-1)[:10].tolist(),
            )
            side = HandSide.RIGHT if o.get("is_right", 1) else HandSide.LEFT
            hands.append(
                HandPose(
                    side=side,
                    keypoints_2d=kp2d,
                    keypoints_3d=kp3d,
                    mano=mano,
                    keypoint_convention=self.keypoint_convention,
                    score=float(o.get("score", 1.0)),
                )
            )
        return hands
