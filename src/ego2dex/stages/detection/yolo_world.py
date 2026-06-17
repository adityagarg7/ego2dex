"""YOLO-World open-vocabulary detection (real-time, ~52 FPS).

arXiv:2401.17270. License: GPLv3 / commercial (Ultralytics) -> NON-permissive.
Loaded via the Ultralytics package.
"""

from __future__ import annotations

import numpy as np

from ...schema.core import Detection, FrameAnnotation
from ..base import DETECTION
from .base import DetectionStageBase


@DETECTION.register("yolo_world", aliases=("yoloworld",))
class YOLOWorld(DetectionStageBase):
    name = "yolo_world"
    requires = ("ultralytics",)
    extra = "detection"
    license = "GPLv3 / Ultralytics commercial"
    license_url = "https://docs.ultralytics.com/models/yolo-world/"

    def load(self) -> None:
        ultra = self.import_or_raise("ultralytics")
        weights = self.param("weights", "yolov8x-worldv2.pt")
        self._model = ultra.YOLOWorld(weights)

    def infer(self, image: np.ndarray, prompt: list[str], fa: FrameAnnotation) -> list[Detection]:
        self._model.set_classes(list(prompt))
        results = self._model.predict(image, conf=float(self.param("conf", 0.25)), verbose=False)
        dets: list[Detection] = []
        for r in results:
            names = r.names
            for b in r.boxes:
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
                cls = int(b.cls[0])
                dets.append(
                    Detection(
                        label=str(names.get(cls, cls)),
                        score=float(b.conf[0]),
                        bbox=[x0, y0, x1 - x0, y1 - y0],
                    )
                )
        return dets
