"""Base for export stages. Each writer serializes the ClipAnnotation to disk.

Export stages are clip-level and run regardless of dry-run (they serialize
whatever annotations the upstream stages produced). Output goes under
``ctx.output_dir`` (overridable per-writer with ``params.output_dir``).
"""

from __future__ import annotations

from pathlib import Path

from ..schema.core import ClipAnnotation
from ..stages.base import Stage


class ExportStageBase(Stage):
    family = "export"
    per_frame = False

    def out_dir(self) -> Path:
        sub = self.param("output_dir")
        base = Path(sub) if sub else self.ctx.output_dir
        base.mkdir(parents=True, exist_ok=True)
        return base

    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        path = self.write(clip)
        self.ctx.extras.setdefault("exports", []).append(str(path))
        return clip

    def write(self, clip: ClipAnnotation) -> Path:  # pragma: no cover - abstract
        raise NotImplementedError
