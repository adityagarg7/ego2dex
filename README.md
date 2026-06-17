# ego2dex

**Egocentric video → SOTA hand/object/interaction annotations → dexterous-hand
retargeting & training exports.**

`ego2dex` is a modular pipeline that ingests **egocentric video** (primarily
GoPro, also Project Aria) and produces training-ready annotations for **dexterous
robot hands**: 3D hand/finger joints + MANO, open-vocabulary object boxes + masks
+ tracks, hand–object interaction, image tags + captions, camera/SLAM, and
**retargeting** of the recovered hand motion onto robot hands — primarily the
**ORCA hand (16 DOF)**, generalizing to arbitrary URDFs (Allegro, Shadow, LEAP, …).

It produces the **perception/data** side that feeds pretraining and imitation; it
is **not** a policy/controller. (Downstream, that data drives e.g. a PPO
controller or a VLA policy.)

```text
ingest → (camera/SLAM) → hands → detection → segmentation/tracking
       → hand-object-interaction → tags → captions
       → annotation store → retargeting → export → viz
```

> See the full diagram in [`docs/diagrams/pipeline.md`](docs/diagrams/pipeline.md).

## Why

The annotation taxonomy is a **superset** compatible with the major egocentric
datasets (Ego4D, Ego-Exo4D, H2O, DexYCB, HOI4D, OakInk2, AssemblyHands, ARCTIC,
HOT3D — see [`docs/datasets.md`](docs/datasets.md)), so one `ClipAnnotation` is
enough to drive both **visual-representation** and **retargeted-action**
pretraining without re-deriving geometry.

## Design constraints (that shape everything)

- **Imports light, runs without weights.** Core deps are numpy/opencv/pillow/
  omegaconf/pydantic/typer/rich/tqdm (+ pycocotools/jsonschema). Every heavy model
  is an **optional extra**, imported lazily inside its stage's `load()`.
- **No GPU / no multi-GB downloads in CI.** A deterministic **dry-run** path runs
  the *entire* graph on CPU with synthetic outputs; weight-needing tests are
  marked `requires_models` and skipped.
- **Artifacts never committed.** Weights, datasets, videos, frames, outputs are
  gitignored. Code in git; artifacts out of git.
- **Licensing is first-class.** Many models are non-permissive; enabling one
  prints a runtime warning. **MANO is research-only and gated** — you supply it.

## The SOTA stack (default ⭐ + fallbacks)

| Role | Default ⭐ | Fallbacks | License (default) |
|---|---|---|---|
| 3D hands + MANO | **HaMeR** | WiLoR, MediaPipe, Dyn-HaMR, WildHands, Hamba | MIT + MANO |
| Smoothing | **1€ filter** | SmoothNet | MIT |
| Open-vocab detection | **Grounding DINO** | YOLO-World, Detic, OWLv2 | Apache-2.0 |
| Segmentation + tracking | **SAM 2 / Grounded-SAM-2** | DEVA, SAMURAI | Apache-2.0 |
| Hand-object interaction | **100DOH** | EgoHOS | research-only / MIT |
| Tags | **RAM++** | — | Apache-2.0 |
| Captions / points | **Qwen2.5-VL** | Florence-2, Molmo, InternVL3 | Apache-2.0 |
| Camera / SLAM | **Aria MPS** / **COLMAP** | DROID-SLAM, VIPE | Aria / BSD |
| Retargeting | **dex-retargeting → ORCA** | Allegro, Shadow, LEAP | MIT |
| Export | **JSON / COCO / HDF5 / LeRobot** | — | MIT/Apache |

Full survey with arXiv + GitHub + license: [`docs/sota_survey.md`](docs/sota_survey.md).
License matrix: [`docs/licenses.md`](docs/licenses.md).

## Install

```bash
# pixi (preferred)
pixi install && pixi run smoke

# or pip / venv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make smoke
```

Heavy models are opt-in extras (`pip install -e ".[hamer]"`, `".[detection,sam2]"`,
`".[retarget,export]"`, …). Source-installed models + weights:
`bash scripts/install_models.sh --help`. See [`docs/install.md`](docs/install.md).

## Quickstart

```bash
# CPU, no weights, no network — runs the whole graph on a synthetic clip
ego2dex run --config configs/pipeline/smoke.yaml --input assets/synthetic --output outputs/smoke
ego2dex info        # list registered stages
python examples/quickstart.py
```

```python
from ego2dex import Pipeline, load_config
clip = Pipeline.from_config(load_config("configs/pipeline/smoke.yaml")).run("assets/synthetic")
print(len(clip.frames), "frames;", sum(len(f.hands) for f in clip.frames), "hands")
clip.save("outputs/clip.json")
```

A live GoPro run (needs GPU + weights):

```bash
ego2dex run -c configs/pipeline/default.yaml -i my_clip.mp4 -o outputs/run
```

## Two pretraining pathways

- **Visual-representation** (R3M/MVP/VC-1/Voltron): frame sequences + rich
  annotations → JSON/parquet manifest.
- **Retargeted-action / imitation** (DexMV/DexCap/EgoDex): per-frame 21 keypoints
  + camera + **retargeted robot joint trajectories** → EgoDex-style HDF5 /
  LeRobotDataset / COCO / JSON.

See [`docs/pretraining.md`](docs/pretraining.md).

## ORCA retargeting

```bash
export EGO2DEX_ORCA_URDF=/path/to/orcahand.urdf      # your orca_sim / orcahand asset
ego2dex retarget --clip outputs/run/clip_full.json --robot orca --urdf "$EGO2DEX_ORCA_URDF" --live
```

21 wrist-relative keypoints → `dex-retargeting` (Position/Vector/DexPilot +
`SeqRetargeting`) → robot joint trajectory. **DOF mismatch** (human 21 / MANO 45
→ ORCA 16) is handled by the link/joint mapping — no equal-DOF assumption. See
[`docs/retargeting.md`](docs/retargeting.md).

## ⚠️ Licensing & MANO

`ego2dex`'s **own code is MIT** (© 2026 Aditya Garg). It orchestrates third-party
models with their own licenses, several **non-commercial / copyleft / API-only**.
**MANO** (every MANO-based hand model + dataset annotation inherits this) is
**research-only and gated** at <https://mano.is.tue.mpg.de> — `ego2dex` never
vendors it; you download `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` yourself. A
fully-permissive, commercial-OK subset exists (MediaPipe + Grounding DINO + SAM2 +
EgoHOS + VLMs + COLMAP + dex-retargeting), at the cost of MANO parameters. See
[`docs/licenses.md`](docs/licenses.md).

## Docs

[architecture](docs/architecture.md) ·
[SOTA survey](docs/sota_survey.md) ·
[annotation schema](docs/annotation_schema.md) ·
[datasets](docs/datasets.md) ·
[retargeting](docs/retargeting.md) ·
[pretraining](docs/pretraining.md) ·
[licenses](docs/licenses.md) ·
[install](docs/install.md)

## Development

```bash
make check     # ruff lint + format-check + mypy (non-blocking) + pytest
make test      # pytest -m "not requires_models"
```

Tests run CPU-only with no weights. CI runs lint + format-check + pytest on
py3.10/3.11. License: [MIT](LICENSE).
