#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

from yeast_xrf.analysis.study_analysis import analyze_study


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT = (
    ROOT
    / "analysis"
    / "validation"
    / "change12_artifact_smoke"
)
REPORT = (
    ROOT
    / "analysis"
    / "validation"
    / "change12_artifact_validation.json"
)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(
        "SOURDOUGH Change 12 · "
        "artifact / contamination screening"
    )
    print("=" * 88)

    result = analyze_study(
        DATA,
        output_root=OUT,
        methods=("Fitted", "NNLS", "ROI"),
        references=("US_IC", "DS_IC"),
        save_concentration_arrays=False,
        compute_element_pairs=False,
        artifact_screening=True,
        max_scans=1,
    )

    summary = result["summary"]
    scans = result["scans"]
    artifacts = result["artifact_candidates"]

    scan_status = (
        str(
            scans.iloc[0].get(
                "artifact_screening_status",
                "",
            )
        )
        if not scans.empty
        else ""
    )

    checks = {
        "scan_analyzed": summary["scans_analyzed"] == 1,
        "no_scan_failure": summary["scan_failures"] == 0,
        "artifact_layer_ready": (
            summary["artifact_layer_ready"] is True
        ),
        "flag_only_policy": (
            summary["artifact_policy"] == "retain_and_flag"
        ),
        "screening_executed": scan_status == "completed",
        "artifact_dataframe_returned": artifacts is not None,
    }

    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "artifact_candidate_count": int(len(artifacts)),
        "note": (
            "A real scan is not required to contain a high-priority "
            "artifact. Validation checks execution and the "
            "non-destructive study artifact layer."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, indent=2, default=str)
    )

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print("-" * 88)
    print(f"Artifact candidates recorded: {len(artifacts)}")
    print(f"Wrote: {REPORT.relative_to(ROOT)}")

    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
