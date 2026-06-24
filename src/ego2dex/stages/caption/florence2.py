"""Florence-2 caption/region stage (tiny, fast, local).

arXiv:2311.06242. HF ``microsoft/Florence-2-large``. License: MIT. Task tokens
drive boxes / region captions / OV-detection / referring-seg. We use it for a
frame caption (+ optional detailed caption / OD). Install: ``[caption]`` extra.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Caption, FrameAnnotation
from ..base import CAPTION
from .base import CaptionStageBase


@CAPTION.register("florence2", aliases=("florence-2", "florence_2"))
class Florence2(CaptionStageBase):
    name = "florence2"
    requires = ("torch", "transformers")
    extra = "caption"
    license = "MIT"
    license_url = "https://huggingface.co/microsoft/Florence-2-large"

    def load(self) -> None:
        self.import_or_raise("torch")
        self.import_or_raise("transformers")
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = self.param("model_id", "microsoft/Florence-2-large")
        dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._torch = torch

    def _run_task(self, pil, task: str, text: str = ""):  # pragma: no cover - needs weights
        prompt = task + text
        inputs = self._processor(text=prompt, images=pil, return_tensors="pt").to(self.device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
            
        gen = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=int(self.param("max_new_tokens", 256)),
            num_beams=3,
            do_sample=False,
        )
        text_out = self._processor.batch_decode(gen, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(
            text_out, task=task, image_size=(pil.width, pil.height)
        )

    def infer(self, image: np.ndarray, fa: FrameAnnotation) -> Caption:  # pragma: no cover
        import cv2
        from PIL import Image
        from ...schema.core import Tags

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Get the object tags
        od_res = self._run_task(pil, "<OD>")
        labels = od_res.get("<OD>", {}).get("labels", [])
        unique_tags = list(set([str(l).strip().lower() for l in labels]))
        fa.tags = Tags(tags=unique_tags, source="florence2_od")
        
        # 2. Get the detailed sentence caption
        cap_res = self._run_task(pil, "<DETAILED_CAPTION>")
        caption_text = cap_res.get("<DETAILED_CAPTION>", "Auto-tagged by Florence-2")

        return Caption(
            frame_caption=caption_text,
            source="florence2",
        )
