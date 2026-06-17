# Architecture

`ego2dex` is a `Pipeline` that runs an ordered list of **Stages** over a video.
Each stage reads the shared `ClipAnnotation`, adds its outputs, and passes it on.
Stages are selected **by name** from per-family **registries** and configured by
**OmegaConf YAML**. Everything is optional/toggleable.

```text
ingest → (camera/SLAM) → hands → detection → segmentation/tracking
       → hand-object-interaction → tags → captions
       → annotation store → retargeting → export → viz
```

See the mermaid diagram in [`diagrams/pipeline.md`](diagrams/pipeline.md).

## Core abstractions

- **`Stage`** (`stages/base.py`): `name`, `family`, `requires` (heavy deps),
  `extra` (the pip extra), `license`/`license_url`, `per_frame`. Lifecycle:
  `setup(params)` → `bind(ctx)` → lazy `load()` (via `ensure_loaded`, which also
  emits the license warning) → `process(clip)` → `teardown()`.
  `import_or_raise(module, extra)` gives a clear, actionable `ImportError`.
- **`RunContext`**: `output_dir`, `device`, `dry_run`, `strict`, `frames`
  (a `FrameStore`), `mano_dir`, `extras`.
- **Registries** (`utils/registry.py`): one per family — `HANDS`, `DETECTION`,
  `SEGMENTATION`, `HOI`, `CAPTION`, `POSE`, `RETARGET`, `EXPORT`, `VIZ`. Stages
  self-register via `@HANDS.register("mediapipe")` at import time.
- **`Pipeline`** (`pipeline.py`): builds stages from a config list, ingests frames
  into a `FrameStore`, runs stages with progress, handles per-frame vs clip-level
  stages, and **gracefully skips** a stage whose deps can't load (logged, unless
  `run.strict`).

## The two execution modes

- **Live** (`run.dry_run = false`): stages call their real models. Missing
  weights/deps → the stage is skipped (or raises under `strict`).
- **Dry-run** (`run.dry_run = true`): **no weights load at all.** Each stage emits
  deterministic synthetic outputs (topologically valid hands, boxes, masks,
  tracks, interactions, tags, captions, camera trajectory, retargeted joints).
  This makes the entire graph testable on CPU with no models — the basis of
  `make smoke` and `tests/test_pipeline.py`.

## Why lazy + light

The build/CI environment may have **no GPU** and must **not** download multi-GB
weights. So: core deps are light (numpy, opencv, pillow, omegaconf, pydantic,
typer, rich, tqdm, pycocotools, jsonschema); every heavy model is an optional
extra imported **inside** `load()`; tests needing real weights are marked
`@pytest.mark.requires_models` and skipped in CI.

## Data flow & artifacts

Pixels live in the in-memory `FrameStore` (an artifact); only annotations enter
the `ClipAnnotation`/JSON. **Weights, datasets, videos, frames, and outputs are
never committed** — see `.gitignore`. Code in git; artifacts out of git.

## Extending

Add a backend by subclassing the family base, implementing `load()` + `infer()`,
decorating with `@<FAMILY>.register("name")`, and dropping a
`configs/<family>/<name>.yaml`. Add a robot by adding a `RobotConfig` to
`retarget/robots.py`. Add an export by subclassing `ExportStageBase`.
