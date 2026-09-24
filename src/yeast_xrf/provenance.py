"""Provenance helpers for derived XRF products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_identity(path: str | Path, *, block_size: int = 1024 * 1024) -> dict[str, Any]:
    """Return a strong file identity for source HDF5 provenance.

    The SHA-256 is intentionally available from day one because later derived maps must be
    invalidated when their source file changes.  For large files this is explicit work rather
    than something the UI should repeat on every refresh.
    """
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
