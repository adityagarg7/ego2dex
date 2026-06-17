"""Every shipped YAML loads, and every stage it names resolves + builds."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ego2dex.config import load_config, load_stage_config
from ego2dex.pipeline import Pipeline
from ego2dex.stages.base import build_stage

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"

STAGE_CONFIGS = sorted(p for p in CONFIG_ROOT.rglob("*.yaml") if p.parent.name != "pipeline")
PIPELINE_CONFIGS = sorted((CONFIG_ROOT / "pipeline").glob("*.yaml"))


@pytest.mark.parametrize("path", STAGE_CONFIGS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_stage_config_builds(path: Path):
    cfg = load_stage_config(path)
    params = OmegaConf.to_container(cfg.get("params", {}) or {}, resolve=True)
    stage = build_stage(cfg["family"], cfg["name"], params)
    assert stage.name
    assert stage.family == cfg["family"]


@pytest.mark.parametrize("path", PIPELINE_CONFIGS, ids=lambda p: p.name)
def test_pipeline_config_builds(path: Path):
    cfg = load_config(path)
    pipe = Pipeline.from_config(cfg)
    assert len(pipe.stages) >= 1
    # describe() touches every stage's metadata
    assert "Pipeline" in pipe.describe()


def test_all_families_have_configs():
    families = {p.parent.name for p in STAGE_CONFIGS}
    for fam in (
        "hands",
        "detection",
        "segmentation",
        "hoi",
        "caption",
        "pose",
        "retarget",
        "export",
    ):
        assert fam in families
