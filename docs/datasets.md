# Dataset survey — justifying the schema as a superset

`ego2dex`'s annotation schema is designed so its output is a **superset**
compatible with the major egocentric / hand-object datasets. The table maps each
dataset to the annotation types it provides; the union is what
[`annotation_schema.md`](annotation_schema.md) implements.

## Comparison table (dataset × annotation type)

Legend: ✅ provided · ➖ partial/subset · ❌ not provided.

| Dataset | Ego? | 2D kpts | 3D kpts | MANO | Obj boxes | Masks/seg | 6DoF obj | Contact/HOI | Camera/SLAM | Gaze | Narration/Action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ego4D** | ✅ | ➖ (hand boxes) | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ FHO state-change | ➖ scans | ➖ subset | ✅ narrations, verb-noun, moments |
| **Ego-Exo4D** | ✅ | ✅ 21 | ✅ 21 | ❌ | ❌ | ✅ (Relations) | ❌ | ➖ | ✅ Aria MPS | ✅ | ✅ keystep, 3 language layers |
| **H2O** | ✅+exo | ✅ 21×2 | ✅ 21×3 | ✅ | — | ➖ | ✅ | ➖ | ✅ pose+intr | ❌ | ✅ verb/action (36) |
| **DexYCB** | ❌ multi | ✅ 21 | ✅ 21 | ✅ (`pose_m` 51) | — | ✅ `seg` | ✅ YCB | ➖ | ✅ 8 views | ❌ | ❌ |
| **HOI4D** | ✅ | ✅ 21 | ✅ | ✅ (right) | — | ✅ 2D/3D/4D panoptic | ✅ cat-level | ➖ | ✅ RGB-D | ❌ | ✅ action seg |
| **OakInk / OakInk2** | ➖ 1 ego+3 exo | ✅ | ✅ | ✅ (+SMPL-X body) | — | ➖ | ✅ | ✅ affordance/intent | ✅ | ❌ | ✅ task programs + text |
| **AssemblyHands** | ✅ (490K ego) | ✅ | ✅ 21 | ➖ | — | ❌ | ❌ | ❌ | ✅ | ❌ | ➖ |
| **ARCTIC** | ✅ ego split | ✅ | ✅ | ✅ two-hand | — | ➖ | ✅ articulated | ✅ contact | ✅ | ❌ | ➖ |
| **HOT3D** | ✅ Aria+Quest3 | ✅ | ✅ | ✅ (+UmeTrack) | — | ➖ | ✅ | ➖ | ✅ multiview | ➖ | ➖ |

## Per-dataset notes & citations

- **Ego4D** — CVPR 2022, [2110.07058](https://arxiv.org/abs/2110.07058).
  Narrations (#C/#O), verb-noun, moments, **FHO state-change keyframes**
  (pre/contact/**PNR**/post), hand+object **boxes** (no keypoints/MANO),
  next-active-object + time-to-contact, gaze subset, 3D scans. CLI `pip install
  ego4d`. JSON schemas: `fho_hands_*.json` (left_hand/right_hand boxes),
  `fho_scod_*.json`, `narrations.json`.
- **Ego-Exo4D** — CVPR 2024, [2311.18259](https://arxiv.org/abs/2311.18259).
  **21 kpts/hand 2D+3D** (`annotation2D`/`annotation3D`, joint names
  `left_wrist`, `left_thumb_1..4`, …), 17 body kpts, instance masks (Relations),
  keystep segments, 3 language layers, gaze, Aria MPS (trajectory/point
  cloud/online_calib). **No MANO, no 6DoF object pose.** Downloader `egoexo`.
  SOTA hand: HP-ViT+/PCIE_Pose 8.31 mm PA-MPJPE.
- **H2O** — ICCV 2021, [2104.11181](https://arxiv.org/abs/2104.11181). Two-hand
  21×3 + **MANO** (`hand_pose_MANO`), 6DoF object pose, cam pose+intrinsics, depth
  (Azure Kinect), verb/action (36 classes), 5 views (4 exo + 1 ego). Plain-text
  per-frame files.
- **DexYCB** — CVPR 2021, [2104.04631](https://arxiv.org/abs/2104.04631). 21
  2D+3D, **MANO** (`pose_m` 51 = 45 PCA + 3 global + 3 trans, betas 10), 6DoF YCB
  pose, **segmentation** (`seg`), 8 RealSense views. `.npz` per frame. NVlabs
  toolkit (GPLv3).
- **HOI4D** — CVPR 2022, [2203.01577](https://arxiv.org/abs/2203.01577).
  Egocentric, 2.4–3M RGB-D, MANO (right), 21 kpts, category-level 6DoF pose,
  2D/3D/4D panoptic seg, articulated objects + CAD, action segmentation.
- **OakInk / OakInk2** — CVPR 2022 / 2024,
  [2203.15709](https://arxiv.org/abs/2203.15709) /
  [2403.19417](https://arxiv.org/abs/2403.19417). MANO (OakInk2 bimanual + SMPL-X
  body), 6DoF object pose, affordance/intent, task programs + dependency graphs +
  text, 1 ego + 3 exo (OakInk2).
- **AssemblyHands** — CVPR 2023, [2304.12301](https://arxiv.org/abs/2304.12301).
  3M images / 490K ego, 21 3D.
- **ARCTIC** — CVPR 2023, [2204.13662](https://arxiv.org/abs/2204.13662). Two-hand
  MANO + articulated object + contact; ego split (WildHands SOTA).
- **HOT3D** — CVPR 2025, [2411.19167](https://arxiv.org/abs/2411.19167). Aria +
  Quest3 multiview ego, MANO + UmeTrack hands, 6DoF objects.

## Format references

- **COCO** — RLE `{size, counts}`, keypoints `[x,y,v]`, **1-based** skeleton.
- **EPIC-KITCHENS-100** — CSV storing **both** timestamps and frame indices
  (mirrored by `ActionSegment`, which keeps seconds + frames + fps).
- **ActivityNet** JSON, **Charades** CSV — temporal action references.

## What `ego2dex` adds beyond any single dataset

The union above + **retargeted robot joint trajectories** (per robot hand) +
**explicit camera-convention metadata** (world↔cam direction, Hamilton
scalar-first quaternions) + **per-frame open-vocab tags/captions/points**. This
makes one `ClipAnnotation` enough to drive both pretraining pathways without
re-deriving geometry. Gaps `ego2dex` does not fill from images alone (e.g. dense
metric depth, exact 6DoF object pose without a model) are left optional in the
schema.
