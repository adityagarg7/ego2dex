"""Native ego2dex JSON export: one JSON per frame + a clip manifest + schema.

Layout under the output dir::

    <out>/frames/000000.json   # FrameAnnotation per frame (masks as RLE)
    <out>/clip.json            # manifest: video_meta, tracks, segments, retargeting
    <out>/clip_full.json       # (optional) the whole ClipAnnotation in one file
    <out>/ego2dex.schema.json  # JSON Schema for validation
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema.core import ClipAnnotation
from ..schema.jsonschema import write_json_schema
from ..stages.base import EXPORT
from .base import ExportStageBase


@EXPORT.register("json", aliases=("json_writer",))
class JSONWriter(ExportStageBase):
    name = "json"
    license = "MIT (ego2dex)"

    def write(self, clip: ClipAnnotation) -> Path:
        out = self.out_dir()
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        indent = 2 if self.param("pretty", True) else None

        frame_files: list[str] = []
        for fa in clip.frames:
            fname = f"{fa.frame_id:06d}.json"
            (frames_dir / fname).write_text(fa.model_dump_json(indent=indent))
            frame_files.append(f"frames/{fname}")

        manifest = {
            "schema_version": clip.schema_version,
            "video_meta": json.loads(clip.video_meta.model_dump_json()),
            "frames": frame_files,
            "tracks": [json.loads(t.model_dump_json()) for t in clip.tracks],
            "action_segments": [json.loads(s.model_dump_json()) for s in clip.action_segments],
            "narrations": [json.loads(n.model_dump_json()) for n in clip.narrations],
            "retargeting": [json.loads(r.model_dump_json()) for r in clip.retargeting],
        }
        (out / "clip.json").write_text(json.dumps(manifest, indent=indent))

        if self.param("single_file", True):
            (out / "clip_full.json").write_text(clip.model_dump_json(indent=indent))
        if self.param("write_schema", True):
            write_json_schema(out / "ego2dex.schema.json", which="clip")
        return out / "clip.json"
