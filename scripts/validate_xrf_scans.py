#!/usr/bin/env python3
'''Validate the canonical XRF loader against all current HDF5 scans.'''

from __future__ import annotations

import json
from pathlib import Path

from yeast_xrf.io.xrf_scan import XRFScan

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUTPUT = ROOT / "analysis" / "validation" / "canonical_loader_validation.json"


def main() -> None:
    files = sorted(DATA.glob("*.h5"))
    if not files:
        raise SystemExit("No HDF5 files found.")

    rows = []
    total_errors = 0
    total_warnings = 0

    print("Canonical XRF Loader Validation")
    print("=" * 78)

    for path in files:
        with XRFScan.open(path) as scan:
            issues = scan.validate()
            errors = [x for x in issues if x.severity == "error"]
            warnings = [x for x in issues if x.severity == "warning"]
            total_errors += len(errors)
            total_warnings += len(warnings)

            rows.append(
                {
                    "scan": path.name,
                    "shape_yx": list(scan.shape),
                    "theta": scan.theta,
                    "fitted_channels": len(scan.channel_names("Fitted")),
                    "energy_channels": int(scan.energy.size),
                    "primary_scalers": len(scan.scaler_names("primary")),
                    "legacy_scalers": len(scan.scaler_names("legacy")),
                    "errors": [x.__dict__ for x in errors],
                    "warnings": [x.__dict__ for x in warnings],
                }
            )

            status = "PASS" if not errors else "FAIL"
            print(
                f"{status:<5} {path.name:<23} shape={scan.shape!s:<12} "
                f"channels={len(scan.channel_names('Fitted')):<3} "
                f"energy={scan.energy.size:<5} "
                f"scalers={len(scan.scaler_names('primary'))}/{len(scan.scaler_names('legacy'))} "
                f"theta={scan.theta}"
            )
            for issue in issues:
                print(f"      {issue.severity.upper():<7} {issue.code}: {issue.message}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "scan_count": len(rows),
                "error_count": total_errors,
                "warning_count": total_warnings,
                "scans": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("-" * 78)
    print(f"{len(rows)} scans checked; {total_errors} error(s); {total_warnings} warning(s).")
    print(f"Wrote: {OUTPUT.relative_to(ROOT)}")
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
