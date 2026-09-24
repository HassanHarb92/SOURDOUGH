"""Extensible derived-feature registry."""

from __future__ import annotations

from collections.abc import Iterable

from yeast_xrf.features.spec import FeatureSpec


class FeatureRegistry:
    def __init__(self) -> None:
        self._items: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.key in self._items:
            raise ValueError(f"feature already registered: {spec.key}")
        self._items[spec.key] = spec

    def get(self, key: str) -> FeatureSpec:
        return self._items[key]

    def all(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({item.family for item in self._items.values()}))

    def extend(self, specs: Iterable[FeatureSpec]) -> None:
        for spec in specs:
            self.register(spec)


registry = FeatureRegistry()
