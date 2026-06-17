# Install

`ego2dex`'s **core is light** and installs with no GPU and no model weights. Heavy
models are **optional extras**, installed per stage. Dep lists are kept in sync
across `pyproject.toml`, `requirements.txt`, and `pixi.toml`.

## Quick start (core only — imports + dry-run)

### pixi (preferred)

```bash
pixi install            # core, light, CPU-only
pixi run smoke          # CPU dry-run pipeline on the synthetic clip
pixi run test           # pytest -m "not requires_models"
```

### pip / venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + dev tooling
make smoke                          # or: ego2dex run -c configs/pipeline/smoke.yaml -i assets/synthetic -o outputs/smoke
```

### Colab / plain pip

```bash
pip install -r requirements.txt    # core deps
pip install -e .
```

## Optional model extras

Install only what a stage needs (each pulls torch / heavy deps):

| Extra | Enables | Notes |
|---|---|---|
| `mediapipe` | MediaPipe Hands (CPU, no MANO) | + download `hand_landmarker.task` |
| `hamer` | HaMeR | + clone repo + `fetch_demo_data.sh` + **MANO** |
| `wilor` | WiLoR | `pip install git+https://github.com/warmshao/WiLoR-mini` |
| `detection` | Grounding DINO / YOLO-World / Detic / transformers | |
| `sam2` | SAM 2 | `pip install git+https://github.com/facebookresearch/sam2` + ckpts |
| `caption` | Qwen2.5-VL / Florence-2 / Molmo / RAM++ | transformers + accelerate |
| `hoi` | 100DOH / EgoHOS | repo-specific builds |
| `pose` | COLMAP/DROID helpers | COLMAP is a system binary |
| `aria` | Project Aria VRS ingestion | `projectaria-tools` |
| `retarget` | dex-retargeting (+ Pinocchio + NLopt) | ORCA needs your URDF |
| `export` | h5py + pyarrow | HDF5 / parquet writers |

```bash
pip install -e ".[mediapipe]"            # CPU smoke, real model
pip install -e ".[detection,sam2]"       # Grounded-SAM-2 chain
pip install -e ".[hamer,retarget,export]"
```

## Git-installed models

Several models are not on PyPI. `scripts/install_models.sh` documents and (with
flags) automates the source installs + weight downloads in a **GPU** environment:

```bash
bash scripts/install_models.sh --help
bash scripts/install_models.sh hamer sam2 grounding_dino    # example
```

## ⚠️ MANO (required for any MANO-based hand model)

MANO is **research-only and gated**. `ego2dex` never vendors it.

1. Register + accept the license at <https://mano.is.tue.mpg.de>.
2. Download `MANO_RIGHT.pkl` / `MANO_LEFT.pkl`.
3. Point stages at the directory via the pipeline config:

```yaml
mano:
  model_dir: /path/to/mano   # contains MANO_RIGHT.pkl, MANO_LEFT.pkl
```

See [`licenses.md`](licenses.md) for the full per-component license matrix.

## ORCA URDF

ORCA retargeting needs your URDF (from `orca_sim` / orcahand assets):

```bash
export EGO2DEX_ORCA_URDF=/path/to/orcahand.urdf
```

or set `params.urdf_path` in `configs/retarget/orca.yaml`. Verify the link names
against your URDF (see [`retargeting.md`](retargeting.md)).

## What runs without weights vs. needs the install step

- **No weights**: package import; the whole pipeline graph in `dry_run`; the
  smoke pipeline; the full test suite (`-m "not requires_models"`); JSON/COCO
  export; camera math; topology; retarget **dry-run**.
- **Needs weights/extras**: any live model stage (HaMeR, SAM2, GDINO, Qwen, …),
  real MediaPipe inference (small bundle), live `dex-retargeting` (URDF), HDF5
  export (h5py), LeRobot export (lerobot).
