# Retargeting — 21 keypoints → robot dexterous hand

`ego2dex` standardizes on
[`dex-retargeting`](https://github.com/dexsuite/dex-retargeting) (MIT, Pinocchio
FK + NLopt SLSQP) so swapping robot hands is a one-line config change. The
**ORCA hand (16 active DOF)** is first-class; the design generalizes to arbitrary
URDFs / DOF counts (Allegro 16, LEAP 16, Shadow 24, 21-DOF hands, …).

## Input contract

The retargeter consumes **21 keypoints in a wrist-relative frame** (standard-21
order) + the robot URDF + a hand-type config (which robot links are the
task/fingertip targets, which joints are optimized). `ego2dex` builds the input
from each `HandPose.keypoints_3d` via `topology.wrist_relative(...)` (wrist → origin).

```text
HandPose.keypoints_3d (21×3, camera frame)
  └─ wrist_relative ─▶ 21×3 (wrist origin)
        └─ dex_retargeting (Position | Vector | DexPilot + SeqRetargeting)
              └─ RetargetingResult.joint_trajectory  (T × DOF)
```

## DOF mismatch is handled by the mapping, never assumed away

Human 21 keypoints / MANO 45 → robot 16 or 21 or 24. We **do not** assume equal
DOF. `dex-retargeting` optimizes the robot's joints so that the chosen **robot
target links** reach the chosen **human keypoint indices**:

- `target_link_human_indices` selects which of the 21 human keypoints are targets
  (e.g. the 5 fingertips `[4, 8, 12, 16, 20]`, optionally + wrist `0`).
- `target_link_names` / `finger_tip_link_names` are the robot links that should
  reach them.
- `target_joint_names` (optional) restricts which joints are optimized; `None`
  uses all actuated joints in the URDF.

The optimizer absorbs any DOF count — a 16-DOF ORCA and a 24-DOF Shadow use the
same 5 fingertip targets.

## The three optimizers

| Optimizer | Matches | Best for |
|---|---|---|
| **Position** | absolute 3D keypoint positions | offline dataset retargeting for **imitation learning** (DexMV-style) |
| **Vector** | keypoint **direction vectors** (translation-invariant) | streaming **teleop** (AnyTeleop-style) |
| **DexPilot** | fingertip-to-fingertip + palm-to-tip vectors w/ contact snapping | precise **pinching** |

All three smooth over time via `SeqRetargeting`.

## ORCA setup (user-supplied URDF)

ORCA is not bundled with `dex-retargeting`, so you provide the URDF (from your
`orca_sim` / orcahand assets):

```yaml
# configs/retarget/orca.yaml
family: retarget
name: dex_retargeting
params:
  robot: orca
  hand_side: both
  retargeting_type: position
  urdf_path: ${EGO2DEX_ORCA_URDF}      # or an absolute path
  finger_tip_link_names: [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
  wrist_link_name: palm
```

**Verify the link names against your URDF** (`<link name="...">`) — the defaults
in `retarget/robots.py::orca_config` are placeholders. Set `EGO2DEX_ORCA_URDF`
or pass `--urdf` to `ego2dex retarget`.

```bash
ego2dex retarget --clip outputs/run/clip_full.json --robot orca \
  --urdf /path/to/orcahand.urdf --live -o outputs/retarget
```

## Built-in hands

Allegro / Shadow / LEAP reuse `dex-retargeting`'s bundled configs + URDFs (from
[`dex-urdf`](https://github.com/dexsuite/dex-urdf)); just set `robot:` and
`retargeting_type:`. Shadow is 24-DOF, Allegro/LEAP 16-DOF — no schema change.

## Optimization vs learned/aligned retarget

Per-frame high-DOF IK can be unreliable; **EgoDex** argues for robot-centric
**joint-space alignment**. `ego2dex` exposes both:

- **`optimization`** — the `dex-retargeting` path above (default).
- **`aligned`** — a pluggable learned/aligned retargeter interface
  (`RetargetStageBase`); plug a model that maps human kinematics → robot joints
  directly, bypassing per-frame IK. Wire it as a new `@RETARGET.register(...)`
  stage producing `RetargetingResult`.
