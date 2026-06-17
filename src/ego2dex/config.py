"""Config loading (OmegaConf / Hydra-compatible YAML).

A *pipeline* config selects an ordered list of stages and sets run/IO options.
Each stage entry is either inline (``{family, name, params}``) or an
``{include: path}`` reference to a standalone per-family stage config (the files
under ``configs/<family>/<name>.yaml``). Standalone stage files are themselves
valid ``StageSpec`` documents, so every YAML in ``configs/`` is independently
loadable and testable.

Example pipeline entry forms::

    stages:
      - {family: ingest, name: default}
      - {include: configs/hands/mediapipe.yaml}
      - {include: configs/export/json.yaml, params: {pretty: true}}  # override
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# Default run/IO blocks merged under any pipeline config that omits them.
_DEFAULTS: dict[str, Any] = {
    "name": "pipeline",
    "description": "",
    "io": {
        "sample_fps": None,
        "stride": None,
        "max_frames": None,
        "source": "auto",  # auto | gopro | aria | generic
        "undistort": False,
    },
    "run": {
        "device": "cpu",
        "dry_run": False,
        "strict": False,
        "output_dir": "outputs/run",
    },
    "mano": {
        "model_dir": None,  # user-supplied MANO_RIGHT.pkl / MANO_LEFT.pkl directory
    },
    "stages": [],
}


def _candidate_bases(config_path: Path) -> list[Path]:
    """Directories to resolve relative ``include`` paths against."""
    here = config_path.resolve().parent
    bases = [Path.cwd(), here, here.parent, here.parent.parent]
    # de-dup preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for b in bases:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _resolve_include(include: str, config_path: Path) -> Path:
    p = Path(include)
    if p.is_absolute() and p.exists():
        return p
    for base in _candidate_bases(config_path):
        cand = base / include
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not resolve include '{include}' referenced by '{config_path}'. "
        f"Tried: {[str(b / include) for b in _candidate_bases(config_path)]}"
    )


def load_stage_config(path: str | Path) -> DictConfig:
    """Load a standalone stage spec file (``{family, name, enabled?, params?}``)."""
    path = Path(path)
    cfg = OmegaConf.load(path)
    if "family" not in cfg or "name" not in cfg:
        raise ValueError(f"Stage config '{path}' must define 'family' and 'name'.")
    return cfg  # type: ignore[return-value]


def _resolve_stages(stages: list[Any], config_path: Path) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for entry in stages:
        entry = (
            OmegaConf.to_container(entry, resolve=True)
            if isinstance(entry, DictConfig)
            else dict(entry)
        )
        if "include" in entry:
            inc_path = _resolve_include(entry["include"], config_path)
            spec = OmegaConf.to_container(load_stage_config(inc_path), resolve=True)
            # inline overrides merge on top of the included spec's params
            override_params = entry.get("params", {}) or {}
            spec_params = spec.get("params", {}) or {}
            spec_params.update(override_params)
            spec["params"] = spec_params
            if "enabled" in entry:
                spec["enabled"] = entry["enabled"]
            resolved.append(spec)
        else:
            entry.setdefault("enabled", True)
            entry.setdefault("params", {})
            resolved.append(entry)
    return resolved


def load_config(path: str | Path) -> DictConfig:
    """Load a pipeline config, fill defaults, and resolve stage ``include``s."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = OmegaConf.load(path)
    merged = OmegaConf.merge(OmegaConf.create(_DEFAULTS), raw)
    stages = OmegaConf.to_container(merged.stages, resolve=True) or []
    merged.stages = _resolve_stages(stages, path)  # type: ignore[assignment]
    return merged  # type: ignore[return-value]


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
