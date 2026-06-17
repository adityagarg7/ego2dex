"""Qwen2.5-VL caption stage (PRIMARY VLM; captions + grounded boxes/points).

arXiv:2502.13923. HF ``Qwen/Qwen2.5-VL-7B-Instruct`` (3B/7B Apache-2.0). Used
for frame captions + action phrases (and, via prompting, grounded boxes/points).
Install: ``[caption]`` extra (transformers + accelerate + qwen-vl-utils).
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Caption, FrameAnnotation
from ..base import CAPTION
from .base import CaptionStageBase

_PROMPT = (
    "Describe what the person's hands are doing in one sentence, then give a short "
    "verb-object action label. Format: 'CAPTION: ... | ACTION: ...'."
)


@CAPTION.register("qwen2_5_vl", aliases=("qwen", "qwen2.5-vl"))
class Qwen2_5VL(CaptionStageBase):
    name = "qwen2_5_vl"
    requires = ("torch", "transformers")
    extra = "caption"
    license = "Apache-2.0 (3B/7B)"
    license_url = "https://github.com/QwenLM/Qwen2.5-VL"

    def load(self) -> None:
        self.import_or_raise("torch")
        self.import_or_raise("transformers")
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_id = self.param("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map=self.param("device_map", "auto")
        ).eval()
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._torch = torch

    def infer(self, image: np.ndarray, fa: FrameAnnotation) -> Caption:  # pragma: no cover
        import cv2
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": pil}, {"type": "text", "text": _PROMPT}],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[pil], return_tensors="pt").to(
            self._model.device
        )
        gen = self._model.generate(**inputs, max_new_tokens=int(self.param("max_new_tokens", 128)))
        trimmed = gen[:, inputs.input_ids.shape[1] :]
        out = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        caption, action = out, None
        if "ACTION:" in out:
            caption, action = out.split("ACTION:", 1)
            caption = caption.replace("CAPTION:", "").strip(" |")
            action = action.strip()
        return Caption(frame_caption=caption.strip(), action=action, source="qwen2_5_vl")
