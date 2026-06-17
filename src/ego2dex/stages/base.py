"""Stage ABC + run context + the per-family registries.

A :class:`Stage` reads the shared :class:`~ego2dex.schema.core.ClipAnnotation`,
adds its outputs, and returns it. Heavy models load lazily in :meth:`Stage.load`
(never at import time). Every stage declares ``requires`` (deps) and ``license``
metadata so the pipeline can warn on non-permissive components and skip stages
whose deps are missing.

The registries below are the single hub the pipeline uses to build any stage by
``(family, name)``. ``retarget`` and ``export`` packages register into the
``RETARGET`` / ``EXPORT`` registries here too, so they are first-class stages.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..schema.core import ClipAnnotation, FrameAnnotation
from ..utils.logging import get_logger, warn_license
from ..utils.registry import Registry

log = get_logger("ego2dex.stage")


# --------------------------------------------------------------------------- #
# Run context (shared mutable state the pipeline threads through stages)
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    """Per-run shared state. Stages read frames/config from here."""

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    device: str = "cpu"
    dry_run: bool = False
    strict: bool = False
    frames: Any = None  # ego2dex.io.frames.FrameStore (avoid hard import)
    mano_dir: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stage ABC
# --------------------------------------------------------------------------- #
class Stage(ABC):
    """Base class for all pipeline stages."""

    name: str = "stage"
    family: str = "generic"
    requires: tuple[str, ...] = ()  # importable module names of heavy deps
    extra: str | None = None  # the pip extra that provides ``requires``
    license: str | None = None
    license_url: str | None = None
    per_frame: bool = True  # documentation hint; clip-level stages set False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.ctx: RunContext = RunContext()
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def setup(self, params: dict[str, Any] | None = None) -> None:
        """Merge stage params (called once before the run)."""
        if params:
            self.params.update(dict(params))

    def bind(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def load(self) -> None:
        """Lazily load heavy models/weights. Default: nothing to load."""

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.license:
            warn_license(self.cls_name(), self.license, self.license_url)
        if not self.dry_run:
            self.load()
        self._loaded = True

    @abstractmethod
    def process(self, clip: ClipAnnotation) -> ClipAnnotation:
        """Consume + augment the clip annotation, return it."""

    def teardown(self) -> None:
        """Release GPU memory / handles. Default: nothing."""

    # -- helpers -----------------------------------------------------------
    @property
    def dry_run(self) -> bool:
        return self.ctx.dry_run

    @property
    def device(self) -> str:
        return self.ctx.device

    def cls_name(self) -> str:
        return f"{self.family}/{self.name}"

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def import_or_raise(self, module: str, extra: str | None = None) -> Any:
        """Import a heavy optional dep or raise a clear, actionable ImportError."""
        extra = extra or self.extra
        try:
            return importlib.import_module(module)
        except ImportError as e:
            hint = f"pip install 'ego2dex[{extra}]'" if extra else f"pip install {module}"
            raise ImportError(
                f"Stage '{self.cls_name()}' needs '{module}', which is not installed. "
                f"Install it with: {hint}. (Heavy model deps are optional extras; the "
                f"core package stays light. See docs/install.md.)"
            ) from e

    def deps_available(self) -> bool:
        """True if all declared ``requires`` modules can be imported."""
        return all(importlib.util.find_spec(mod) is not None for mod in self.requires)

    def iter_frames(
        self, clip: ClipAnnotation
    ) -> Iterator[tuple[FrameAnnotation, np.ndarray | None]]:
        """Yield ``(FrameAnnotation, image_bgr_or_None)`` for per-frame stages."""
        store = self.ctx.frames
        for fa in clip.frames:
            img = store.get(fa.frame_id) if store is not None else None
            yield fa, img

    def __repr__(self) -> str:
        return f"<Stage {self.cls_name()} params={self.params}>"


# --------------------------------------------------------------------------- #
# Registries (one per family) + the lookup hub
# --------------------------------------------------------------------------- #
HANDS: Registry[Stage] = Registry("hands")
DETECTION: Registry[Stage] = Registry("detection")
SEGMENTATION: Registry[Stage] = Registry("segmentation")
HOI: Registry[Stage] = Registry("hoi")
CAPTION: Registry[Stage] = Registry("caption")
POSE: Registry[Stage] = Registry("pose")
RETARGET: Registry[Stage] = Registry("retarget")
EXPORT: Registry[Stage] = Registry("export")
INGEST: Registry[Stage] = Registry("ingest")
VIZ: Registry[Stage] = Registry("viz")

FAMILY_REGISTRIES: dict[str, Registry[Stage]] = {
    "ingest": INGEST,
    "hands": HANDS,
    "detection": DETECTION,
    "segmentation": SEGMENTATION,
    "hoi": HOI,
    "caption": CAPTION,
    "pose": POSE,
    "retarget": RETARGET,
    "export": EXPORT,
    "viz": VIZ,
}


def get_registry(family: str) -> Registry[Stage]:
    if family not in FAMILY_REGISTRIES:
        raise KeyError(f"Unknown stage family '{family}'. Known: {sorted(FAMILY_REGISTRIES)}")
    return FAMILY_REGISTRIES[family]


def build_stage(family: str, name: str, params: dict[str, Any] | None = None) -> Stage:
    """Instantiate a stage by ``(family, name)`` and apply its params."""
    stage = get_registry(family).create(name)
    stage.setup(params or {})
    return stage


def available_stages() -> dict[str, list[str]]:
    """Map family -> registered stage names (for ``ego2dex info``)."""
    return {fam: reg.keys() for fam, reg in FAMILY_REGISTRIES.items()}
