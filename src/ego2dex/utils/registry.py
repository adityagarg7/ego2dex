"""A tiny, generic name->class registry.

Each stage family (hands, detection, segmentation, hoi, caption, pose,
retarget, export) owns one ``Registry``. Stages self-register via the
``@REGISTRY.register("name")`` decorator at import time. Configs then select a
stage purely by its string name, so swapping a backbone is a one-line YAML edit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string key to a class (or factory). Generic over the base type."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, type[T]] = {}

    def register(
        self, key: str | None = None, *, aliases: tuple[str, ...] = ()
    ) -> Callable[[type[T]], type[T]]:
        """Class decorator: register under ``key`` (defaults to ``cls.name``)."""

        def _decorator(cls: type[T]) -> type[T]:
            name = key or getattr(cls, "name", None) or cls.__name__
            self._add(name, cls)
            for alias in aliases:
                self._add(alias, cls)
            return cls

        return _decorator

    def _add(self, name: str, cls: type[T]) -> None:
        if name in self._entries and self._entries[name] is not cls:
            raise KeyError(
                f"'{name}' already registered in '{self.name}' registry "
                f"({self._entries[name].__name__})."
            )
        self._entries[name] = cls

    def get(self, key: str) -> type[T]:
        """Return the registered class for ``key`` or raise a helpful error."""
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(
                f"Unknown '{self.name}' stage '{key}'. Available: {sorted(self._entries)}"
            ) from None

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Instantiate the registered class for ``key``."""
        return self.get(key)(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, entries={self.keys()})"
