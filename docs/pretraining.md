# Pretraining pathways

`ego2dex` emits artifacts for **two** downstream pretraining pathways. Both are
*consumers* of the annotation pipeline — `ego2dex` produces the perception/data
side; the policy/representation training lives elsewhere (e.g. the user's
`fv-orca-hand-rl`).

## 1. Visual-representation pretraining

Goal: learn transferable visual features from egocentric manipulation video
(R3M / MVP / VC-1 / Voltron-style).

- **Export**: frame sequences + the rich per-frame annotations as a
  webdataset/parquet manifest (`export/json` gives the per-frame JSON + manifest;
  pair it with the frame images you extracted).
- **What helps**: tags/captions (language supervision), hand keypoints + masks
  (auxiliary heads), action segments (temporal contrastive windows).
- **No robot needed** — this path does not require retargeting or a URDF.

```bash
ego2dex run -c configs/pipeline/default.yaml -i clip.mp4 -o out/   # writes JSON manifest
# then point your R3M/VC-1 loader at out/frames/*.json + the frame images
```

## 2. Retargeted-action / imitation learning

Goal: learn dexterous **action** policies from human video (DexMV / DexCap /
EgoDex-style), to pretrain or bootstrap a controller like the ORCA PPO policy.

- **Export**: per-frame 21 keypoints + camera intrinsics/extrinsics +
  **retargeted robot joint trajectories** + object states.
- **Writers**:
  - **EgoDex-style HDF5** (`export/hdf5`): camera `intrinsics (T,3,3)` /
    `extrinsics (T,4,4)`, per-hand `keypoints_3d (T,21,3)`, `mano_pose (T,48)`,
    per-joint `confidence (T,21)`, per-robot `joint_trajectory (T,DOF)` with
    `joint_names`, and a `language` task string.
  - **LeRobotDataset** (`export/lerobot`): `action` / `observation.state` =
    robot joint angles, `task` = language; ready for LeRobot training loops.
  - **COCO / JSON** for keypoint/box/mask supervision.

```bash
# full chain incl. ORCA retarget + HDF5 + LeRobot
ego2dex run -c configs/pipeline/default.yaml -i clip.mp4 -o out/
ego2dex export --clip out/clip_full.json --writer hdf5 -o out/
ego2dex export --clip out/clip_full.json --writer lerobot -o out/
```

## How the pieces line up with the user's stack

```text
ego2dex (this repo)                         downstream (NOT here)
─────────────────────────────              ───────────────────────────────
perception + retargeting           ──▶     imitation / BC pretrain
  • 21 kpts + MANO + camera                  • LeRobot / HDF5 episodes
  • retargeted ORCA joint traj      ──▶     RL fine-tune (fv-orca-hand-rl)
  • masks / tags / captions                  • PPO + shaped reward (orca_sim)
                                   ──▶     VLA policy (OpenVLA / π0 / GR00T)
```

A **VLA** outputs robot actions and is a *policy* parallel to the PPO controller;
it **consumes** `ego2dex` data and is intentionally **not** part of the labeling
loop (see [`sota_survey.md`](sota_survey.md) §J). A **VLM** is the perception-side
counterpart and **is** used (tags/captions/pointing).

## Practical notes

- **Confidence** is preserved per keypoint (HDF5 `confidence`, JSON
  `keypoints_2d[...,2]`) so you can mask low-quality frames.
- **Camera conventions** are explicit (world↔cam direction + Hamilton
  scalar-first quaternion) so SE(3) math is unambiguous across writers.
- **Frames are artifacts**: extract them next to the annotations; never commit
  them.
