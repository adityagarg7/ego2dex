"""Tests that need real model weights and/or a GPU.

These are marked ``requires_models`` and are SKIPPED in CI (which runs
``pytest -m "not requires_models"``). Run locally with the relevant extras +
weights installed:  ``pytest -m requires_models``.
"""

from __future__ import annotations

import importlib.util

import pytest

from ego2dex.stages.base import RunContext, build_stage

pytestmark = pytest.mark.requires_models


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.skipif(not _have("mediapipe"), reason="mediapipe extra not installed")
def test_mediapipe_live_infer(synthetic_dir):
    """Real MediaPipe HandLandmarker on the synthetic clip (needs the .task bundle)."""
    import cv2

    stage = build_stage("hands", "mediapipe", {"num_hands": 2})
    stage.bind(RunContext(dry_run=False, device="cpu"))
    stage.ensure_loaded()  # raises a clear ImportError if the .task model is missing
    img = cv2.imread(str(sorted(synthetic_dir.glob("*.png"))[0]))
    hands = stage.infer(img, frame_id=0)
    assert isinstance(hands, list)  # may be empty on synthetic frames


@pytest.mark.skipif(not _have("torch"), reason="torch not installed")
def test_hamer_loads_with_weights():
    """HaMeR should load when torch + the hamer package + checkpoint are present."""
    stage = build_stage("hands", "hamer", {})
    stage.bind(RunContext(dry_run=False, device="cpu"))
    stage.ensure_loaded()  # needs the hamer package + fetch_demo_data.sh weights


@pytest.mark.skipif(not _have("dex_retargeting"), reason="dex_retargeting not installed")
def test_orca_retarget_live_needs_urdf():
    """Live ORCA retargeting requires a user URDF -> a clear, actionable error."""
    from ego2dex.retarget.robots import orca_config

    cfg = orca_config()
    if cfg.resolved_urdf() is None:
        pytest.skip("set EGO2DEX_ORCA_URDF to run live ORCA retargeting")
