# Examples

| File | What it shows | Needs |
|---|---|---|
| `quickstart.py` | Run the CPU dry-run pipeline, read the schema, export COCO, retarget onto ORCA | core only (no weights) |

```bash
pip install -e .
python examples/quickstart.py
```

For a real (live) run, install the relevant extras + weights
(`scripts/install_models.sh`, `docs/install.md`) and flip `run.dry_run` to
`false` in the pipeline config — e.g.:

```bash
pip install -e ".[mediapipe]"   # + download hand_landmarker.task
ego2dex run -c configs/pipeline/smoke.yaml -i assets/synthetic -o outputs/smoke --strict
# (set model_path in configs/hands/mediapipe.yaml and run.dry_run=false)
```
