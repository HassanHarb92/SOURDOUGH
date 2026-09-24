#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

from yeast_xrf.analysis.study_analysis import (
    analyze_study,
    discover_scans,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUTROOT = ROOT / "analysis" / "validation" / "change11_study_smoke"
SUMMARY = ROOT / "analysis" / "validation" / "change11_study_validation.json"


def main():
    scans = discover_scans(DATA)
    print("SOURDOUGH Change 11A · automated study validation")
    print("=" * 88)
    print(f"Discovered scans: {len(scans)}")

    if OUTROOT.exists():
        shutil.rmtree(OUTROOT)
    OUTROOT.mkdir(parents=True)

    # Real-data execution smoke: two scans, all methods/references,
    # no persisted arrays to keep installation validation compact.
    result = analyze_study(
        DATA,
        output_root=OUTROOT,
        methods=("Fitted", "NNLS", "ROI"),
        references=("US_IC", "DS_IC"),
        save_concentration_arrays=False,
        compute_element_pairs=True,
        max_scans=2,
    )

    s = result["summary"]
    checks = {
        "all_repository_scans_discovered": len(scans) >= 1,
        "smoke_scans_analyzed": s["scans_analyzed"] == 2,
        "smoke_failures_zero": s["scan_failures"] == 0,
        "cells_detected": s["detected_cells"] > 0,
        "cell_element_records_created": (
            s["cell_element_records"] > 0
        ),
        "scan_element_records_created": (
            s["scan_element_records"] > 0
        ),
        "element_pair_records_created": (
            s["element_pair_records"] > 0
        ),
        "provisional_masks_explicit": (
            s["segmentation_status"]
            == "provisional_tfy_v1"
        ),
        "canonical_cells_not_claimed": (
            s["canonical_cells_ready"] is False
        ),
        "artifact_layer_not_claimed": (
            s["artifact_layer_ready"] is False
        ),
    }

    payload = {
        "repository_scan_count": len(scans),
        "smoke_summary": s,
        "checks": checks,
        "passed": all(checks.values()),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2))

    for key, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {key}")

    print("-" * 88)
    print(f"Wrote: {SUMMARY.relative_to(ROOT)}")

    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
