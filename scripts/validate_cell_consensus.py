#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

from yeast_xrf.analysis.study_analysis import analyze_study


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT = ROOT / "analysis" / "validation" / "change13_consensus_smoke"
REPORT = ROOT / "analysis" / "validation" / "change13_consensus_validation.json"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print("SOURDOUGH Change 13A · multichannel cell consensus")
    print("=" * 88)

    result = analyze_study(
        DATA,
        output_root=OUT,
        methods=("Fitted",),
        references=("US_IC",),
        save_concentration_arrays=False,
        compute_element_pairs=False,
        artifact_screening=True,
        multichannel_consensus=True,
        max_scans=1,
    )

    summary = result["summary"]
    consensus = result["cell_consensus"]
    support = result["cell_channel_support"]

    checks = {
        "scan_analyzed": summary["scans_analyzed"] == 1,
        "no_scan_failure": summary["scan_failures"] == 0,
        "consensus_ready": summary["multichannel_consensus_ready"] is True,
        "candidates_created": len(consensus) > 0,
        "support_rows_created": len(support) > 0,
        "review_pending": summary["cell_consensus_review_status"] == "pending",
        "canonical_not_finalized": summary["canonical_cells_finalized"] is False,
    }

    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "candidate_count": len(consensus),
        "support_row_count": len(support),
        "class_counts": (
            consensus["consensus_class"].value_counts().to_dict()
            if not consensus.empty
            else {}
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, default=str))

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print("-" * 88)
    print(f"Consensus candidates: {len(consensus)}")
    print(f"Support rows: {len(support)}")
    print(f"Wrote: {REPORT.relative_to(ROOT)}")

    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
