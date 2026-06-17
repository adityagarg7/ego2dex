"""The Pipeline: build an ordered list of stages from config and run them.

Run flow::

    ingest (frames -> FrameStore + ClipAnnotation skeleton, default camera)
      -> stage_1.process(clip) -> stage_2.process(clip) -> ... -> clip

Each stage loads lazily and may be skipped gracefully if its heavy deps are
missing (logged, unless ``run.strict``). In ``run.dry_run`` mode no weights load
at all -- stages emit deterministic synthetic outputs so the whole graph is
exercisable on CPU with no models (the basis of ``tests/test_pipeline.py``).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .config import load_config, to_dict
from .io.frames import load_into_store
from .io.gopro import gopro_default_intrinsics
from .schema.core import ClipAnnotation, FrameAnnotation
from .stages.base import RunContext, Stage, build_stage
from .utils.logging import get_logger

log = get_logger("ego2dex.pipeline")


class Pipeline:
    """An ordered list of stages plus the run/IO configuration."""

    def __init__(
        self,
        stages: list[Stage],
        io_cfg: dict[str, Any] | None = None,
        run_cfg: dict[str, Any] | None = None,
        mano_cfg: dict[str, Any] | None = None,
        name: str = "pipeline",
    ) -> None:
        self.stages = stages
        self.io_cfg = io_cfg or {}
        self.run_cfg = run_cfg or {}
        self.mano_cfg = mano_cfg or {}
        self.name = name

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: DictConfig | str | Path) -> Pipeline:
        if not isinstance(cfg, DictConfig):
            cfg = load_config(cfg)
        d = to_dict(cfg)
        stages: list[Stage] = []
        for spec in d.get("stages", []):
            if not spec.get("enabled", True):
                continue
            stage = build_stage(spec["family"], spec["name"], spec.get("params", {}))
            stages.append(stage)
        return cls(
            stages=stages,
            io_cfg=d.get("io", {}),
            run_cfg=d.get("run", {}),
            mano_cfg=d.get("mano", {}),
            name=d.get("name", "pipeline"),
        )

    # ------------------------------------------------------------------ #
    def _make_context(self, output_dir: str | Path | None) -> RunContext:
        out = Path(output_dir or self.run_cfg.get("output_dir", "outputs/run"))
        mano_dir = self.mano_cfg.get("model_dir")
        return RunContext(
            output_dir=out,
            device=self.run_cfg.get("device", "cpu"),
            dry_run=bool(self.run_cfg.get("dry_run", False)),
            strict=bool(self.run_cfg.get("strict", False)),
            mano_dir=Path(mano_dir) if mano_dir else None,
        )

    def _ingest(self, input_path: str | Path, ctx: RunContext) -> ClipAnnotation:
        """Load frames into a FrameStore and build the ClipAnnotation skeleton."""
        store = load_into_store(
            input_path,
            sample_fps=self.io_cfg.get("sample_fps"),
            stride=self.io_cfg.get("stride"),
            max_frames=self.io_cfg.get("max_frames"),
            source_hint=self.io_cfg.get("source", "auto"),
        )
        ctx.frames = store
        meta = store.meta
        default_cam = None
        if meta.source == "gopro" and meta.width:
            default_cam = gopro_default_intrinsics(meta.width, meta.height)
        frames = [
            FrameAnnotation(
                frame_id=fid,
                timestamp=store.timestamp(fid),
                camera=default_cam.model_copy() if default_cam else None,
            )
            for fid in store.frame_ids()
        ]
        log.info("Ingested %d frames from %s (source=%s)", len(frames), input_path, meta.source)
        return ClipAnnotation(video_meta=meta, frames=frames)

    # ------------------------------------------------------------------ #
    def run(self, input_path: str | Path, output_dir: str | Path | None = None) -> ClipAnnotation:
        """Run the full pipeline over ``input_path`` and return the ClipAnnotation."""
        ctx = self._make_context(output_dir)
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        clip = self._ingest(input_path, ctx)

        mode = "DRY-RUN" if ctx.dry_run else "live"
        log.info("Pipeline '%s' [%s] running %d stage(s)", self.name, mode, len(self.stages))

        for stage in self.stages:
            stage.bind(ctx)
            label = stage.cls_name()
            try:
                stage.ensure_loaded()
            except ImportError as e:
                if ctx.strict:
                    raise
                log.warning("Skipping stage '%s' (deps missing): %s", label, e)
                continue
            except Exception as e:
                if ctx.strict:
                    raise
                log.warning("Stage '%s' failed to load (%s); skipping.", label, e)
                continue
            try:
                clip = stage.process(clip)
                log.info("  - %s done", label)
            except NotImplementedError as e:
                if ctx.strict:
                    raise
                log.warning("Stage '%s' not fully implemented for live run: %s", label, e)
            except Exception as e:
                if ctx.strict:
                    raise
                log.warning("Stage '%s' errored (%s); continuing.", label, e)
            finally:
                with contextlib.suppress(Exception):
                    stage.teardown()
        return clip

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        lines = [f"Pipeline '{self.name}' ({len(self.stages)} stages):"]
        for i, s in enumerate(self.stages, 1):
            lic = f" [license: {s.license}]" if s.license else ""
            lines.append(f"  {i:>2}. {s.cls_name()}{lic}")
        return "\n".join(lines)
