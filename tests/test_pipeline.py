"""End-to-end dry-run of the whole pipeline graph on synthetic data (no weights)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ego2dex.config import _resolve_stages
from ego2dex.pipeline import Pipeline
from ego2dex.schema.jsonschema import validate_clip_json


def _full_cfg(out: Path):
    cfg = OmegaConf.create(
        {
            "name": "full",
            "io": {"source": "generic", "max_frames": 5},
            "run": {"device": "cpu", "dry_run": True, "strict": True, "output_dir": str(out)},
            "mano": {"model_dir": None},
            "stages": [
                {"family": "caption", "name": "ram_plus", "params": {}},
                {"family": "pose", "name": "colmap", "params": {}},
                {"family": "detection", "name": "grounding_dino", "params": {}},
                {"family": "segmentation", "name": "sam2", "params": {}},
                {"family": "hands", "name": "mediapipe", "params": {}},
                {"family": "hands", "name": "smoothing", "params": {}},
                {"family": "hoi", "name": "hand_object_detector", "params": {}},
                {"family": "caption", "name": "qwen2_5_vl", "params": {}},
                {
                    "family": "retarget",
                    "name": "dex_retargeting",
                    "params": {"robot": "orca", "hand_side": "both"},
                },
                {"family": "export", "name": "json", "params": {}},
                {"family": "export", "name": "coco", "params": {}},
            ],
        }
    )
    cfg.stages = _resolve_stages(
        OmegaConf.to_container(cfg.stages), Path("configs/pipeline/x.yaml")
    )
    return cfg


def test_smoke_pipeline_runs(synthetic_dir, tmp_path):
    from ego2dex.config import load_config

    pipe = Pipeline.from_config(load_config("configs/pipeline/smoke.yaml"))
    clip = pipe.run(synthetic_dir, output_dir=tmp_path)
    assert len(clip.frames) == 8
    assert all(len(f.hands) == 2 for f in clip.frames)
    assert (tmp_path / "clip_full.json").exists()
    validate_clip_json((tmp_path / "clip_full.json").read_text())


def test_full_graph_dry_run(synthetic_dir, tmp_path):
    pipe = Pipeline.from_config(_full_cfg(tmp_path))
    clip = pipe.run(synthetic_dir, output_dir=tmp_path)

    assert len(clip.frames) == 5
    assert sum(len(f.hands) for f in clip.frames) == 10
    assert sum(len(f.detections) for f in clip.frames) > 0
    assert sum(len(f.masks) for f in clip.frames) > 0
    assert sum(len(f.interactions) for f in clip.frames) == 10
    assert clip.frames[0].tags is not None
    assert clip.frames[0].caption is not None
    assert clip.frames[0].camera.extrinsics is not None
    assert len(clip.tracks) > 0
    # both hands retargeted onto ORCA's 16 DOF
    assert {r.hand_side for r in clip.retargeting} == {"left", "right"}
    for r in clip.retargeting:
        assert r.robot == "orca"
        assert len(r.joint_names) == 16
        assert len(r.joint_trajectory) == 5


def test_missing_deps_skip_gracefully(synthetic_dir, tmp_path):
    # A live (non-dry-run) run with a weight-dependent stage and strict=False
    # should skip that stage rather than crash.
    cfg = OmegaConf.create(
        {
            "name": "skip",
            "io": {"source": "generic", "max_frames": 2},
            "run": {
                "device": "cpu",
                "dry_run": False,
                "strict": False,
                "output_dir": str(tmp_path),
            },
            "mano": {"model_dir": None},
            "stages": [{"family": "hands", "name": "hamer", "params": {}}],
        }
    )
    cfg.stages = _resolve_stages(OmegaConf.to_container(cfg.stages), Path("x.yaml"))
    clip = Pipeline.from_config(cfg).run(synthetic_dir, output_dir=tmp_path)
    # HaMeR needs torch+weights -> skipped -> no hands, but pipeline completes.
    assert sum(len(f.hands) for f in clip.frames) == 0
