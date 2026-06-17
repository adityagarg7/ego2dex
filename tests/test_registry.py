"""Registry behavior + that all expected stages are registered."""

from __future__ import annotations

import pytest

from ego2dex.stages.base import (
    CAPTION,
    DETECTION,
    EXPORT,
    HANDS,
    HOI,
    POSE,
    RETARGET,
    SEGMENTATION,
    available_stages,
    build_stage,
)
from ego2dex.utils.registry import Registry


def test_registry_register_and_get():
    reg: Registry = Registry("demo")

    @reg.register("foo", aliases=("f",))
    class Foo:
        name = "foo"

    assert reg.get("foo") is Foo
    assert reg.get("f") is Foo
    assert "foo" in reg
    assert reg.create("foo").__class__ is Foo


def test_registry_unknown_key_message():
    reg: Registry = Registry("demo")
    with pytest.raises(KeyError, match="Unknown 'demo' stage"):
        reg.get("missing")


def test_registry_duplicate_raises():
    reg: Registry = Registry("demo")

    @reg.register("dup")
    class A:
        pass

    with pytest.raises(KeyError, match="already registered"):

        @reg.register("dup")
        class B:
            pass


def test_expected_primary_stages_present():
    assert "mediapipe" in HANDS and "hamer" in HANDS and "wilor" in HANDS
    assert "grounding_dino" in DETECTION and "grounded_sam2" in DETECTION
    assert "sam2" in SEGMENTATION and "samurai" in SEGMENTATION
    assert "hand_object_detector" in HOI and "egohos" in HOI
    assert "qwen2_5_vl" in CAPTION and "ram_plus" in CAPTION
    assert "aria_mps" in POSE and "colmap" in POSE
    assert "dex_retargeting" in RETARGET
    assert {"json", "coco", "hdf5", "lerobot"} <= set(EXPORT.keys())


def test_build_stage_applies_params():
    stage = build_stage("hands", "mediapipe", {"num_hands": 1})
    assert stage.param("num_hands") == 1
    assert stage.cls_name() == "hands/mediapipe"


def test_available_stages_covers_all_families():
    fams = available_stages()
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
        assert fam in fams and len(fams[fam]) > 0
