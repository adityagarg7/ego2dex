"""Grounding DINO open-vocabulary detection (PRIMARY; open weights, local).

ECCV 2024, arXiv:2303.05499. Text prompt ("cup. hand. plate.") -> boxes +
phrases + scores. ~52.5 AP COCO (Swin-L). License: Apache-2.0.

We load via HuggingFace ``transformers`` (``IDEA-Research/grounding-dino-base``),
which is the least-friction route. The original repo (IDEA-Research/GroundingDINO)
and MM-Grounding-DINO (mmdetection) are alternatives for fine-tuning.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Detection, FrameAnnotation
from ..base import DETECTION
from .base import DetectionStageBase


@DETECTION.register("grounding_dino", aliases=("gdino", "groundingdino"))
class GroundingDINO(DetectionStageBase):
    name = "grounding_dino"
    requires = ("torch", "transformers")
    extra = "detection"
    license = "Apache-2.0"
    license_url = "https://github.com/IDEA-Research/GroundingDINO"

    def load(self) -> None:
        self.import_or_raise("torch")
        self.import_or_raise("transformers")
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = self.param("model_id", "IDEA-Research/grounding-dino-base")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model = self._model.to(self.device).eval()
        self._torch = torch

    def infer(self, image: np.ndarray, prompt: list[str], fa: FrameAnnotation) -> list[Detection]:
        import cv2
        from PIL import Image

        h, w = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # GDINO wants a lowercased, period-separated prompt.
        text = ". ".join(p.lower().strip() for p in prompt) + "."
        inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=float(self.param("box_threshold", 0.3)),
            text_threshold=float(self.param("text_threshold", 0.25)),
            target_sizes=[(h, w)],
        )[0]
        dets: list[Detection] = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"], strict=False
        ):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            dets.append(
                Detection(label=str(label), score=float(score), bbox=[x0, y0, x1 - x0, y1 - y0])
            )
        return dets
