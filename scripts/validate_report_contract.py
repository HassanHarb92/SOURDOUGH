#!/usr/bin/env python3
from pathlib import Path
import json
import tempfile

import pandas as pd

from yeast_xrf.reporting.report_contract import (
    refresh_report_contract,
)


def main():
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "20260101_000000"
        run.mkdir()

        pd.DataFrame(
            [{"scan": "a.h5", "include": True}]
        ).to_csv(run / "study_manifest.csv", index=False)
        pd.DataFrame(
            [{"scan": "a.h5", "status": "analyzed"}]
        ).to_csv(run / "scans.csv", index=False)
        (run / "study_summary.json").write_text("{}")
        (run / "study_config.json").write_text("{}")
        (run / "study_provenance.json").write_text("{}")

        manifest = refresh_report_contract(run)

        checks = {
            "manifest_written": (
                run / "report" / "report_manifest.json"
            ).is_file(),
            "status_written": (
                run / "report" / "report_status.csv"
            ).is_file(),
            "skeleton_written": (
                run / "report" / "REPORT_SKELETON.md"
            ).is_file(),
            "twelve_sections": (
                len(manifest["sections"]) == 12
            ),
            "not_final_ready": (
                manifest["final_report_ready"] is False
            ),
            "artifact_grounded_policy": (
                manifest["report_policy"][
                    "artifact_grounded_claims_only"
                ]
                is True
            ),
            "organelle_likeness_policy": (
                manifest["report_policy"][
                    "organelle_output_is_likeness_not_identity"
                ]
                is True
            ),
        }

        for name, passed in checks.items():
            print(
                f"{'PASS' if passed else 'FAIL'}  {name}"
            )

        if not all(checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
