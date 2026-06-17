"""SAM 2 segmentation + video tracking (PRIMARY).

arXiv:2408.00714. Box/point -> mask + streaming-memory video propagation,
multi-object. License: Apache-2.0. Install from source:
    pip install git+https://github.com/facebookresearch/sam2
    # checkpoints: download_ckpts.sh (sam2.1_hiera_large.pt, ...)

This stage uses SAM2's *image* predictor to turn each frame's detection boxes
into masks. For true memory-based tracking across frames use the video
predictor (``build_sam2_video_predictor``) -- see ``samurai`` / ``deva`` for
motion-aware / long-clip ID consistency variants.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Detection, FrameAnnotation, Mask
from ..base import SEGMENTATION
from .base import SegmentationStageBase


@SEGMENTATION.register("sam2", aliases=("sam2.1",))
class SAM2(SegmentationStageBase):
    name = "sam2"
    requires = ("torch", "sam2")
    extra = "sam2"
    license = "Apache-2.0"
    license_url = "https://github.com/facebookresearch/sam2"

    def load(self) -> None:
        self.import_or_raise("torch")
        sam2_build = self.import_or_raise("sam2.build_sam")
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        cfg = self.param("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        ckpt = self.param("checkpoint")
        if not ckpt:
            raise ImportError(
                "SAM2 needs a checkpoint. Download with sam2's download_ckpts.sh and "
                "pass params.checkpoint=/path/to/sam2.1_hiera_large.pt"
            )
        model = sam2_build.build_sam2(cfg, ckpt, device=self.device)
        self._predictor = SAM2ImagePredictor(model)

    def infer_masks(
        self, image: np.ndarray, dets: list[Detection], fa: FrameAnnotation
    ) -> list[Mask]:
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)
        masks: list[Mask] = []
        for det in dets:
            x, y, bw, bh = det.bbox
            box = np.array([x, y, x + bw, y + bh])
            m, scores, _ = self._predictor.predict(box=box[None, :], multimask_output=False)
            binary = (m[0] > 0).astype(np.uint8)
            masks.append(Mask.from_binary(binary, label=det.label, score=float(scores[0])))
        return masks
