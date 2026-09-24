"""Safe, assumption-free HDF5 structure inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from yeast_xrf.model import DatasetDescriptor, GroupDescriptor, H5Inventory


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return [_jsonable(x) for x in value.tolist()]
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": [_jsonable(x) for x in value.flat[:16]],
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _attrs(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in obj.attrs.items()}


def inventory_h5(path: str | Path) -> H5Inventory:
    """Inspect groups/datasets/attributes without loading image arrays into memory."""
    source = Path(path).expanduser().resolve()
    groups: list[GroupDescriptor] = []
    datasets: list[DatasetDescriptor] = []

    with h5py.File(source, "r") as handle:
        root_attrs = _attrs(handle)

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            absolute = "/" + name if name else "/"
            if isinstance(obj, h5py.Group):
                groups.append(GroupDescriptor(path=absolute, attrs=_attrs(obj)))
                return
            dtype = np.dtype(obj.dtype)
            datasets.append(
                DatasetDescriptor(
                    path=absolute,
                    shape=tuple(int(v) for v in obj.shape),
                    dtype=str(dtype),
                    ndim=int(obj.ndim),
                    size=int(obj.size),
                    nbytes_estimate=int(obj.size * dtype.itemsize),
                    chunks=None if obj.chunks is None else tuple(int(v) for v in obj.chunks),
                    compression=obj.compression,
                    compression_opts=_jsonable(obj.compression_opts),
                    attrs=_attrs(obj),
                )
            )

        handle.visititems(visitor)

    return H5Inventory(
        source=source,
        file_size_bytes=source.stat().st_size,
        root_attrs=root_attrs,
        groups=tuple(groups),
        datasets=tuple(datasets),
    )


def inventory_to_dict(inv: H5Inventory) -> dict[str, Any]:
    return {
        "source": str(inv.source),
        "file_size_bytes": inv.file_size_bytes,
        "root_attrs": inv.root_attrs,
        "groups": [
            {"path": group.path, "attrs": group.attrs}
            for group in inv.groups
        ],
        "datasets": [
            {
                "path": dataset.path,
                "shape": list(dataset.shape),
                "dtype": dataset.dtype,
                "ndim": dataset.ndim,
                "size": dataset.size,
                "nbytes_estimate": dataset.nbytes_estimate,
                "chunks": None if dataset.chunks is None else list(dataset.chunks),
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
                "attrs": dataset.attrs,
            }
            for dataset in inv.datasets
        ],
    }


def write_inventory(inv: H5Inventory, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory_to_dict(inv), indent=2, sort_keys=True, default=str))
    return output
