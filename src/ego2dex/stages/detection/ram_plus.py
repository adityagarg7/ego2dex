"""RAM++ open-set image tagging (tags double as Grounding DINO auto-prompts).

Recognize Anything Plus (arXiv:2310.15200). License: Apache-2.0. Produces an
open-vocabulary tag list per frame; ego2dex stores it as ``FrameAnnotation.tags``
and downstream detection uses those tags as the open-vocab prompt.

Registered under the ``caption`` family (it is a tagging/VLM-side model) but
lives in this package so it sits next to the detector it feeds. Install:
    pip install recognize-anything   # provides the `ram` package
"""

from __future__ import annotations

import numpy as np

from ...schema.core import ClipAnnotation, Tags
from ..base import CAPTION, Stage

_SYNTH_TAGS = ["hand", "cup", "table", "bowl", "person", "kitchen"]


@CAPTION.register("ram_plus", aliases=("ram++", "ramplus"))
class RAMPlus(Stage):
    name = "ram_plus"
    family = "caption"
    requires = ("torch", "ram")
    extra = "caption"
    license = "Apache-2.0"
    license_url = "https://github.com/xinyu1205/recognize-anything"

    def load(self) -> None:
        self.import_or_raise("torch")
        ram_models = self.import_or_raise("ram.models")
        from ram import get_transform

        ckpt = self.param("checkpoint")
        if not ckpt:
            raise ImportError(
                "RAM++ needs weights: download ram_plus_swin_large_14m.pth and pass "
                "params.checkpoint=/path/to/ram_plus_swin_large_14m.pth"
            )
        image_size = int(self.param("image_size", 384))
        self._model = (
            ram_models.ram_plus(pretrained=ckpt, image_size=image_size, vit="swin_l")
            .eval()
            .to(self.device)
        )
        self._transform = get_transform(image_size=image_size)

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        for fa, img in self._frames(clip):
            fa.tags = self._tags(img) if not (self.dry_run or img is None) else self._synthetic()
        return clip

    def _frames(self, clip):
        store = self.ctx.frames
        for fa in clip.frames:
            yield fa, (store.get(fa.frame_id) if store is not None else None)

    def _synthetic(self) -> Tags:
        return Tags(tags=list(_SYNTH_TAGS), source="ram_plus(dry-run)")

    def _tags(self, image: np.ndarray) -> Tags:  # pragma: no cover - needs weights
        import cv2
        from PIL import Image
        from ram import inference_ram

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        x = self._transform(pil).unsqueeze(0).to(self.device)
        res = inference_ram(x, self._model)
        tags = [t.strip() for t in str(res[0]).split("|") if t.strip()]
        return Tags(tags=tags, source="ram_plus")
