"""Stage families. Importing each subpackage self-registers its stages.

Order doesn't matter for registration; the pipeline selects stages by name.
``retarget`` and ``export`` live in their own top-level packages but register
into the same hub (see :mod:`ego2dex.stages.base`); ``ego2dex/__init__`` imports
them so their stages are available too.
"""

from __future__ import annotations

from . import caption, detection, hands, hoi, pose
from .base import (
    FAMILY_REGISTRIES,
    RunContext,
    Stage,
    available_stages,
    build_stage,
    get_registry,
)

__all__ = [
    "FAMILY_REGISTRIES",
    "RunContext",
    "Stage",
    "available_stages",
    "build_stage",
    "caption",
    "detection",
    "get_registry",
    "hands",
    "hoi",
    "pose",
]
