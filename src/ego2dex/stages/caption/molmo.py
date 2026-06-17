"""Molmo pointing stage (contact / affordance points).

arXiv:2409.17146. HF ``allenai/Molmo-7B-D-0924``. License: Apache-2.0. Molmo can
*point* at things, which is ideal for grounding contact / affordance points that
feed grasp reasoning. We store points in ``Caption.points``. Install: ``[caption]``.
(InternVL3, arXiv:2504.10479, MIT, is an alternative general VLM option.)
"""

from __future__ import annotations

import re

import numpy as np

from ...schema.core import Caption, FrameAnnotation
from ..base import CAPTION
from .base import CaptionStageBase


@CAPTION.register("molmo")
class Molmo(CaptionStageBase):
    name = "molmo"
    requires = ("torch", "transformers")
    extra = "caption"
    license = "Apache-2.0"
    license_url = "https://huggingface.co/allenai/Molmo-7B-D-0924"

    def load(self) -> None:
        self.import_or_raise("torch")
        self.import_or_raise("transformers")
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = self.param("model_id", "allenai/Molmo-7B-D-0924")
        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        ).eval()
        self._torch = torch

    def infer(self, image: np.ndarray, fa: FrameAnnotation) -> Caption:  # pragma: no cover
        import cv2
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w = image.shape[:2]
        q = self.param("query", "Point to where the hand is contacting the object.")
        inputs = self._processor.process(images=[pil], text=q)
        inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}
        out = self._model.generate_from_batch(
            inputs, self._make_gen_cfg(), tokenizer=self._processor.tokenizer
        )
        text = self._processor.tokenizer.decode(
            out[0, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )
        return Caption(points=self._parse_points(text, w, h), source="molmo")

    def _make_gen_cfg(self):  # pragma: no cover
        from transformers import GenerationConfig

        return GenerationConfig(max_new_tokens=128, stop_strings="<|endoftext|>")

    @staticmethod
    def _parse_points(text: str, w: int, h: int) -> list[dict]:
        # Molmo emits points like: <point x="34.5" y="60.1" alt="...">...</point>
        # coordinates are percentages of width/height.
        pts = []
        for m in re.finditer(r'x="?([\d.]+)"?\s+y="?([\d.]+)"?', text):
            pts.append(
                {
                    "x": float(m.group(1)) / 100.0 * w,
                    "y": float(m.group(2)) / 100.0 * h,
                    "label": "contact",
                }
            )
        return pts
