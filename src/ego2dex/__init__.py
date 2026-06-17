"""ego2dex - egocentric video -> dexterous-hand annotations & retargeting.

Importing this package is intentionally CHEAP: it pulls in only the light core
(schema, topology, config, pipeline scaffolding). No torch, no model weights, no
GPU. Heavy model backends are imported lazily inside each stage's ``load()``.

Typical use::

    from ego2dex import Pipeline, load_config
    cfg = load_config("configs/pipeline/smoke.yaml")
    pipe = Pipeline.from_config(cfg)
    clip = pipe.run("assets/synthetic")

See ``docs/architecture.md`` for the full stage graph.
"""

from __future__ import annotations

from . import export as _export
from . import retarget as _retarget

# Importing the stage subpackages triggers self-registration of the built-in
# stages into the per-family registries (side-effect imports, kept lazy-light:
# the modules only register classes; heavy deps load in ``Stage.load()``).
from . import stages as _stages
from . import viz as _viz
from .config import load_config
from .pipeline import Pipeline
from .topology import HandConvention
from .version import __version__

__all__ = [
    "HandConvention",
    "Pipeline",
    "__version__",
    "load_config",
]
