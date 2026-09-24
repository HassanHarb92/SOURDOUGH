#!/usr/bin/env python3
"""Compare all generated HDF5 inventories and report 3D-readiness evidence."""

from pathlib import Path

from yeast_xrf.io.schema_compare import (
    analyze_3d_readiness,
    compare_inventories,
    load_inventories,
    write_reports,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "analysis" / "inventory"
OUTPUT = ROOT / "analysis" / "schema"


def main() -> None:
    inventories = load_inventories(INVENTORY)
    comparison = compare_inventories(inventories)
    readiness = analyze_3d_readiness(inventories, comparison)
    written = write_reports(inventories, comparison, readiness, OUTPUT)

    print("Yeast XRF schema comparison")
    print("=" * 52)
    print(f"Scans:                {comparison['scan_count']}")
    print(f"Unique dataset paths: {comparison['dataset_path_count']}")
    print(f"Common paths:         {len(comparison['common_dataset_paths'])}")
    print(f"Variable paths:       {len(comparison['variable_dataset_paths'])}")
    print()
    print("3D-readiness routes")
    for route, info in readiness["routes"].items():
        print(f"  {route:<22} {info['status']:<12} {info['reason']}")
    print()
    print("Wrote:")
    for name, path in written.items():
        print(f"  {name:<16} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
