#!/usr/bin/env python3
"""Inventory every HDF5 file under img.dat."""

from pathlib import Path

from yeast_xrf.io.discovery import discover_h5
from yeast_xrf.io.h5_inventory import inventory_h5, write_inventory


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT = ROOT / "analysis" / "inventory"


def main() -> None:
    files = discover_h5(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in files:
        target = OUT / f"{path.name}.inventory.json"
        write_inventory(inventory_h5(path), target)
        print(f"✓ {path.name} -> {target.relative_to(ROOT)}")
    print(f"\n{len(files)} HDF5 files inventoried.")


if __name__ == "__main__":
    main()
