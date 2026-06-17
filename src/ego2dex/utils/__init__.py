"""Light, dependency-free utilities: logging, registry, geometry."""

from .logging import (
    NON_PERMISSIVE_LICENSES,
    get_logger,
    setup_logging,
    warn_license,
)
from .registry import Registry

__all__ = [
    "NON_PERMISSIVE_LICENSES",
    "Registry",
    "get_logger",
    "setup_logging",
    "warn_license",
]
