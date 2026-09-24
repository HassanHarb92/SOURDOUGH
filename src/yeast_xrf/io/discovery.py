"""Source-file discovery."""

from __future__ import annotations

from pathlib import Path


def discover_h5(data_dir: str | Path, pattern: str = "*.h5") -> list[Path]:
    return sorted(Path(data_dir).expanduser().resolve().glob(pattern))
