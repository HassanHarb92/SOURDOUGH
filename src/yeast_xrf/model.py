"""Canonical in-memory models.

These are deliberately conservative.  The exact beamline HDF5 layout is still unknown, so
the first milestone is to *observe* file structure rather than force it into a guessed schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetDescriptor:
    path: str
    shape: tuple[int, ...]
    dtype: str
    ndim: int
    size: int
    nbytes_estimate: int
    chunks: tuple[int, ...] | None = None
    compression: str | None = None
    compression_opts: Any = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupDescriptor:
    path: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class H5Inventory:
    source: Path
    file_size_bytes: int
    root_attrs: dict[str, Any]
    groups: tuple[GroupDescriptor, ...]
    datasets: tuple[DatasetDescriptor, ...]
