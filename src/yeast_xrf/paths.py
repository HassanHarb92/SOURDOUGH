"""Repository-local path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        return cls(Path(root).expanduser().resolve())

    @property
    def data_dir(self) -> Path:
        return self.root / "img.dat"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def inventory_dir(self) -> Path:
        return self.analysis_dir / "inventory"

    @property
    def feature_dir(self) -> Path:
        return self.analysis_dir / "features"

    @property
    def figure_dir(self) -> Path:
        return self.analysis_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.analysis_dir / "tables"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"
