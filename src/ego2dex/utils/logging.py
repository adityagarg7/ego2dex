"""Logging + runtime license warnings.

Uses ``rich`` if available (it is a core dep) but degrades gracefully to the
stdlib so the module never hard-fails on import.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_LICENSE_WARNED: set[str] = set()


# Substrings (lower-cased) that mark a license as NON-permissive. When a stage
# reports one of these in its ``license`` metadata, we surface a one-time
# runtime warning so the user knows they have opted into restricted terms.
NON_PERMISSIVE_LICENSES: tuple[str, ...] = (
    "cc-by-nc",
    "cc by-nc",
    "non-commercial",
    "noncommercial",
    "research only",
    "research-only",
    "gpl",
    "agpl",
    "api-only",
    "api only",
    "restricted",
    "mano",  # MANO is research-only and gated; flag anything that mentions it
)


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once. Honors ``EGO2DEX_LOG_LEVEL`` env var."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("EGO2DEX_LOG_LEVEL", level)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich is a core dep but stay safe
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    logging.basicConfig(level=level, format=fmt, handlers=[handler], force=False)
    _CONFIGURED = True


def get_logger(name: str = "ego2dex") -> logging.Logger:
    """Return a configured logger (lazily configures the root handler)."""
    setup_logging()
    return logging.getLogger(name)


def is_non_permissive(license_id: str | None) -> bool:
    """True if ``license_id`` looks non-permissive (NC / GPL / research-only)."""
    if not license_id:
        return False
    low = license_id.lower()
    return any(token in low for token in NON_PERMISSIVE_LICENSES)


def warn_license(component: str, license_id: str | None, url: str | None = None) -> None:
    """Emit a *one-time* runtime warning if ``component`` is non-permissive.

    Stages call this from ``load()`` so that enabling, e.g., WiLoR (CC-BY-NC-ND)
    or anything MANO-based (research-only) prints an actionable notice. See
    ``docs/licenses.md`` for the full matrix.
    """
    if not is_non_permissive(license_id):
        return
    key = f"{component}:{license_id}"
    if key in _LICENSE_WARNED:
        return
    _LICENSE_WARNED.add(key)
    log = get_logger("ego2dex.license")
    extra = f" See: {url}" if url else ""
    log.warning(
        "[bold yellow]LICENSE[/]: '%s' is governed by a NON-permissive license "
        "(%s). It is NOT free for commercial use. You are responsible for "
        "compliance. Anything MANO-based is research-only and gated at "
        "https://mano.is.tue.mpg.de.%s",
        component,
        license_id,
        extra,
    )
