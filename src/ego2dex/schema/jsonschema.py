"""JSON Schema generation + validation for the ego2dex annotation format.

We derive the schema straight from the pydantic models (single source of truth)
and additionally validate with the ``jsonschema`` library so downstream
consumers (other languages / tools) can validate ego2dex JSON without Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import ClipAnnotation, FrameAnnotation


def clip_json_schema() -> dict[str, Any]:
    """JSON Schema for a whole :class:`ClipAnnotation`."""
    return ClipAnnotation.model_json_schema()


def frame_json_schema() -> dict[str, Any]:
    """JSON Schema for a single :class:`FrameAnnotation`."""
    return FrameAnnotation.model_json_schema()


def write_json_schema(path: str | Path, which: str = "clip") -> Path:
    """Write the schema to ``path``. ``which`` is 'clip' or 'frame'."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = clip_json_schema() if which == "clip" else frame_json_schema()
    path.write_text(json.dumps(schema, indent=2))
    return path


def validate_clip_json(data: dict[str, Any] | str) -> None:
    """Validate ``data`` against the clip schema. Raises on failure.

    Uses ``jsonschema`` if installed; otherwise falls back to pydantic's own
    validation (which is stricter anyway).
    """
    if isinstance(data, str):
        data = json.loads(data)
    try:
        import jsonschema

        jsonschema.validate(instance=data, schema=clip_json_schema())
    except ImportError:
        ClipAnnotation.model_validate(data)
