"""Derived-artifact manifest contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from yeast_xrf.provenance import canonical_json_sha256


@dataclass(frozen=True)
class FeatureArtifactKey:
    source_sha256: str
    source_dataset: str
    feature_key: str
    parameters: dict[str, Any]
    dimensionality: str
    spacing: tuple[float, ...] | None = None

    def digest(self) -> str:
        return canonical_json_sha256(asdict(self))


def artifact_directory(root: str | Path, key: FeatureArtifactKey) -> Path:
    return Path(root) / key.feature_key / key.digest()
