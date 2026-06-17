"""HaMeR hand-pose stage (PRIMARY / default).

"Reconstructing Hands in 3D with Transformers", CVPR 2024 (arXiv:2312.05251).
ViT-H, per-hand-crop -> MANO (48 pose + 10 shape) + 778-vtx mesh + 21 joints.
Egocentric-proven (EgoExo4D Ego-Pose Hands 2nd place; EPIC-KITCHENS/Ego4D).

License: MIT (code) + MANO (NON-commercial, research-only, gated). You must
supply MANO_RIGHT.pkl/MANO_LEFT.pkl. Install:
    pip install -e '.[hamer]'  &&  git clone https://github.com/geopavlakos/hamer
    bash fetch_demo_data.sh    # downloads ViT-H + ViTPose weights to _DATA/

HaMeR bundles a ViTDet/ViTPose hand detector; we use it to crop hands.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import HandPose, HandSide, MANOParams
from ...topology import (
    DEFAULT_MANO_TIP_VERTEX_IDS,
    MANO_FULL_POSE_DIM,
    HandConvention,
)
from ..base import HANDS
from .base import HandStageBase


@HANDS.register("hamer")
class HaMeR(HandStageBase):
    name = "hamer"
    requires = ("torch", "hamer")
    extra = "hamer"
    license = "MIT (code) + MANO non-commercial (weights/model)"
    license_url = "https://github.com/geopavlakos/hamer"
    keypoint_convention = HandConvention.STANDARD21  # HaMeR exports OpenPose-21 order

    def load(self) -> None:
        torch = self.import_or_raise("torch")
        # HaMeR's package layout: `from hamer.models import load_hamer, DEFAULT_CHECKPOINT`
        hamer_models = self.import_or_raise("hamer.models")
        self._torch = torch
        self.device = self.param("device", self.ctx.device)

        ckpt = self.param("checkpoint", getattr(hamer_models, "DEFAULT_CHECKPOINT", None))
        if not ckpt:
            raise ImportError(
                "HaMeR checkpoint not found. Run hamer's fetch_demo_data.sh and pass "
                "params.checkpoint=_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
            )
        # Real load (exercised only with weights present):
        self._model, self._model_cfg = hamer_models.load_hamer(ckpt)
        self._model = self._model.to(self.device).eval()
        # Bundled hand detector (ViTDet/Regnety) + ViTPose keypoint detector.
        from hamer.utils import recursive_to  # noqa: F401  (used in infer)

        self._detector = self._load_detector()

    def _load_detector(self):  # pragma: no cover - needs weights
        # HaMeR demo uses detectron2 ViTDet + a ViTPose wrapper. See hamer/demo.py.
        raise NotImplementedError(
            "Wire HaMeR's bundled ViTDet+ViTPose detector here (see hamer/demo.py). "
            "It crops per-hand boxes that feed the ViT-H reconstruction head."
        )

    def infer(self, image: np.ndarray, frame_id: int) -> list[HandPose]:  # pragma: no cover
        """BGR image -> HaMeR MANO + 21 joints per detected hand.

        Pipeline (from hamer/demo.py), to wire when weights are available:
          1. detector -> hand boxes + right/left flags
          2. build per-hand crop batch (ViTDet 256x256), run ``self._model``
          3. out['pred_mano_params'] -> {global_orient, hand_pose(45), betas(10)}
          4. out['pred_keypoints_3d'] (21x3, OpenPose order), project with cam
        """
        torch = self._torch
        hands: list[HandPose] = []
        _boxes, _is_right = self._detector(image)
        with torch.no_grad():
            raise NotImplementedError(
                "Insert the HaMeR forward pass + MANO param extraction here. "
                "Map out['pred_keypoints_3d'] (OpenPose-21) into HandPose; mesh "
                f"tips use vertex ids {DEFAULT_MANO_TIP_VERTEX_IDS}."
            )
        return hands

    # Convenience for tests / downstream: assemble a HandPose from raw arrays.
    @staticmethod
    def handpose_from_outputs(
        kp2d: np.ndarray,
        kp3d: np.ndarray,
        global_orient: np.ndarray,
        hand_pose_45: np.ndarray,
        betas: np.ndarray,
        side: HandSide,
        vertices: np.ndarray | None = None,
        score: float = 1.0,
    ) -> HandPose:
        go = np.asarray(global_orient).reshape(-1)
        hp = np.asarray(hand_pose_45).reshape(-1)
        if go.shape[0] + hp.shape[0] != MANO_FULL_POSE_DIM:
            raise ValueError(
                f"global({go.shape[0]}) + pose({hp.shape[0]}) must = {MANO_FULL_POSE_DIM}"
            )
        return HandPose(
            side=side,
            keypoints_2d=kp2d,
            keypoints_3d=kp3d,
            mano=MANOParams(
                global_orient=go.tolist(),
                pose=hp.tolist(),
                betas=np.asarray(betas).reshape(-1).tolist(),
            ),
            vertices=vertices,
            keypoint_convention=HandConvention.STANDARD21,
            score=score,
        )
