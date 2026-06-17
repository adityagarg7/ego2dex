"""ego2dex command-line interface (Typer).

Commands:
  run           run a pipeline config over a video / image dir
  validate      build a pipeline from config without running (checks stages)
  info          show version + registered stages
  fetch-weights print/run the model install script (GPU env)
  retarget      retarget an existing clip.json onto a robot hand
  export        run an export writer on an existing clip
  viz           render annotation overlays for an existing clip
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .pipeline import Pipeline
from .stages.base import available_stages
from .version import __version__

app = typer.Typer(add_completion=False, help="ego2dex: egocentric video -> dexterous-hand data.")
console = Console()


@app.command()
def info() -> None:
    """Show version, registered stages, and licenses."""
    console.print(f"[bold]ego2dex[/] v{__version__}")
    table = Table("family", "stages")
    for fam, names in available_stages().items():
        table.add_row(fam, ", ".join(sorted(set(names))))
    console.print(table)
    console.print(
        "[yellow]Licensing:[/] many models are non-permissive (CC-BY-NC / GPL / "
        "research-only). MANO is research-only + gated. See docs/licenses.md."
    )


@app.command()
def validate(config: Path = typer.Option(..., "--config", "-c", help="pipeline YAML")) -> None:
    """Build the pipeline from config (no run) and print its stage list."""
    cfg = load_config(config)
    pipe = Pipeline.from_config(cfg)
    console.print(pipe.describe())
    console.print("[green]OK[/] - config is valid and all stages resolve.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="pipeline YAML"),
    input: Path = typer.Option(..., "--input", "-i", help="video file or image dir"),
    output: Path | None = typer.Option(None, "--output", "-o", help="output dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="force dry-run (no weights)"),
    strict: bool = typer.Option(False, "--strict", help="fail instead of skipping stages"),
) -> None:
    """Run a pipeline over INPUT and write annotations to OUTPUT."""
    cfg = load_config(config)
    if dry_run:
        cfg.run.dry_run = True
    if strict:
        cfg.run.strict = True
    pipe = Pipeline.from_config(cfg)
    console.print(pipe.describe())
    clip = pipe.run(input, output_dir=output)
    n_hands = sum(len(f.hands) for f in clip.frames)
    n_dets = sum(len(f.detections) for f in clip.frames)
    console.print(
        f"[green]Done[/]: {len(clip.frames)} frames, {n_hands} hands, {n_dets} detections, "
        f"{len(clip.retargeting)} retarget result(s)."
    )


@app.command("fetch-weights")
def fetch_weights(
    run_script: bool = typer.Option(False, "--run", help="actually execute the install script"),
) -> None:
    """Show (or run) scripts/install_models.sh for the heavy model weights."""
    script = Path("scripts/install_models.sh")
    if run_script and script.exists():
        import subprocess

        raise SystemExit(subprocess.call(["bash", str(script)]))
    console.print(
        "Heavy weights are NOT bundled. See [bold]scripts/install_models.sh[/] and "
        "[bold]docs/install.md[/]. MANO must be downloaded by you (research-only, gated): "
        "https://mano.is.tue.mpg.de"
    )


@app.command()
def retarget(
    clip: Path = typer.Option(..., "--clip", help="an existing clip_full.json"),
    robot: str = typer.Option("orca", "--robot", help="orca|allegro|shadow|leap"),
    urdf: Path | None = typer.Option(None, "--urdf", help="URDF path (required for orca)"),
    output: Path = typer.Option(Path("outputs/retarget"), "--output", "-o"),
    dry_run: bool = typer.Option(True, "--dry-run/--live"),
) -> None:
    """Retarget an existing clip onto a robot hand and save the trajectory."""
    from .schema.core import ClipAnnotation
    from .stages.base import RunContext, build_stage

    annotation = ClipAnnotation.load(clip)
    stage = build_stage(
        "retarget",
        "dex_retargeting",
        {
            "robot": robot,
            "urdf_path": str(urdf) if urdf else None,
        },
    )
    stage.bind(RunContext(output_dir=output, dry_run=dry_run))
    stage.ensure_loaded()
    annotation = stage.process(annotation)
    output.mkdir(parents=True, exist_ok=True)
    annotation.save(output / "clip_retargeted.json")
    console.print(
        f"[green]Retargeted[/] -> {output / 'clip_retargeted.json'} "
        f"({len(annotation.retargeting)} result(s))"
    )


@app.command()
def export(
    clip: Path = typer.Option(..., "--clip", help="an existing clip_full.json"),
    writer: str = typer.Option("coco", "--writer", help="json|coco|hdf5|lerobot"),
    output: Path = typer.Option(Path("outputs/export"), "--output", "-o"),
) -> None:
    """Run a single export writer on an existing clip."""
    from .schema.core import ClipAnnotation
    from .stages.base import RunContext, build_stage

    annotation = ClipAnnotation.load(clip)
    stage = build_stage("export", writer, {})
    stage.bind(RunContext(output_dir=output))
    stage.ensure_loaded()
    stage.process(annotation)
    console.print(f"[green]Exported[/] {writer} -> {output}")


@app.command()
def viz(
    clip: Path = typer.Option(..., "--clip", help="clip_full.json"),
    frames: Path = typer.Option(..., "--frames", help="image dir matching frame ids"),
    output: Path = typer.Option(Path("outputs/viz"), "--output", "-o"),
) -> None:
    """Render annotation overlays for an existing clip."""
    import cv2

    from .schema.core import ClipAnnotation
    from .viz.overlay import draw_frame

    annotation = ClipAnnotation.load(clip)
    files = sorted(p for p in frames.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    output.mkdir(parents=True, exist_ok=True)
    for fa, f in zip(annotation.frames, files, strict=False):
        img = cv2.imread(str(f))
        if img is None:
            continue
        cv2.imwrite(str(output / f"{fa.frame_id:06d}.jpg"), draw_frame(img, fa))
    console.print(f"[green]Wrote overlays[/] -> {output}")


def main() -> None:  # console-script entry alt
    app()


if __name__ == "__main__":
    app()
