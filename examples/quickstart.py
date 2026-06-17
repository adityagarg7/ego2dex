#!/usr/bin/env python
"""Quickstart: run the CPU dry-run pipeline and inspect the annotations.

    python examples/quickstart.py

No weights, no GPU, no network — uses the bundled synthetic clip and the
deterministic dry-run path. Then shows how to read the schema, export COCO, and
retarget onto the ORCA hand (dry-run).
"""

from __future__ import annotations

from pathlib import Path

from ego2dex import Pipeline, load_config
from ego2dex.schema import to_coco
from ego2dex.schema.core import HandSide

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # 1) Build + run the smoke pipeline (dry-run, CPU).
    cfg = load_config(ROOT / "configs/pipeline/smoke.yaml")
    pipe = Pipeline.from_config(cfg)
    print(pipe.describe())
    clip = pipe.run(ROOT / "assets/synthetic", output_dir="outputs/quickstart")

    # 2) Inspect the unified annotation object.
    print(f"\nframes: {len(clip.frames)}  source: {clip.video_meta.source}")
    f0 = clip.frames[0]
    for hand in f0.hands:
        kp = hand.kp2d_array()
        print(
            f"  hand[{hand.side}] wrist=({kp[0, 0]:.1f},{kp[0, 1]:.1f}) "
            f"conv={hand.keypoint_convention} mano={'yes' if hand.mano else 'no'}"
        )

    # 3) Export to COCO (keypoints + categories).
    coco = to_coco(clip)
    print(
        f"\nCOCO: {len(coco['images'])} images, {len(coco['annotations'])} anns, "
        f"categories={[c['name'] for c in coco['categories']]}"
    )

    # 4) Retarget onto ORCA (dry-run; live needs a URDF — see docs/retargeting.md).
    from ego2dex.stages.base import RunContext, build_stage

    stage = build_stage("retarget", "dex_retargeting", {"robot": "orca", "hand_side": "right"})
    stage.bind(RunContext(dry_run=True))
    stage.ensure_loaded()
    clip = stage.process(clip)
    for r in clip.retargeting:
        if HandSide(r.hand_side) == HandSide.RIGHT:
            print(f"\nORCA retarget: {len(r.joint_trajectory)} frames x {len(r.joint_names)} DOF")


if __name__ == "__main__":
    main()
