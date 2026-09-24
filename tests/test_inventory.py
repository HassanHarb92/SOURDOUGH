from pathlib import Path

import h5py
import numpy as np

from yeast_xrf.io.h5_inventory import inventory_h5


def test_inventory_reads_structure_without_interpretation(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["facility"] = "test"
        group = handle.create_group("xrf")
        group.attrs["note"] = "hello"
        ds = group.create_dataset("map", data=np.zeros((3, 4), dtype=np.float32))
        ds.attrs["units"] = "counts"

    inv = inventory_h5(path)
    assert inv.root_attrs["facility"] == "test"
    assert any(group.path == "/xrf" for group in inv.groups)
    dataset = next(item for item in inv.datasets if item.path == "/xrf/map")
    assert dataset.shape == (3, 4)
    assert dataset.dtype == "float32"
    assert dataset.attrs["units"] == "counts"
