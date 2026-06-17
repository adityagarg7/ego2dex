"""COCO export stage: images/annotations/categories with keypoints + RLE masks."""

from __future__ import annotations

from pathlib import Path

from ..schema.coco import save_coco
from ..schema.core import ClipAnnotation
from ..stages.base import EXPORT
from .base import ExportStageBase


@EXPORT.register("coco", aliases=("coco_writer",))
class COCOWriter(ExportStageBase):
    name = "coco"
    license = "MIT (ego2dex)"

    def write(self, clip: ClipAnnotation) -> Path:
        out = self.out_dir()
        fname = self.param("filename", "annotations.coco.json")
        path = out / fname
        save_coco(clip, path, indent=2 if self.param("pretty", True) else None)
        return path
