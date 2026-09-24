#!/usr/bin/env python3
'''Exercise QC and explicit normalization across all current real XRF scans.'''

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yeast_xrf.analysis.normalization import (
    normalize_by_reference,
    positive_reference_percentile_floor,
)
from yeast_xrf.analysis.qc import (
    build_qc_flags,
    count_rate_diagnostics,
    live_time_diagnostics,
)
from yeast_xrf.io.xrf_scan import XRFScan

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUTPUT = ROOT / "analysis" / "qc" / "qc_normalization_validation.json"


def finite_summary(data: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "finite_count": int(finite.size),
        "finite_fraction": float(finite.size / arr.size) if arr.size else 0.0,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
    }


def main() -> None:
    files = sorted(DATA.glob("*.h5"))
    rows = []
    failures = []

    print("QC + Normalization Real-Data Validation")
    print("=" * 94)

    for path in files:
        try:
            with XRFScan.open(path) as scan:
                us_ic = scan.scaler("US_IC", "primary")
                ds_ic = scan.scaler("DS_IC", "primary")
                icr = scan.scaler("ICR", "primary")
                ocr = scan.scaler("OCR", "primary")
                elt = scan.scaler("ELT", "primary")
                ert = scan.scaler("ERT", "primary")
                stored_dead_time = scan.scaler("Dead_Time", "primary")

                floor_us = positive_reference_percentile_floor(us_ic, 1.0)
                floor_ds = positive_reference_percentile_floor(ds_ic, 1.0)

                channel_rows = {}
                for channel in ("P", "Fe", "Zn"):
                    raw = scan.map(channel, "Fitted")
                    qc = build_qc_flags(
                        raw,
                        high_value_percentile=99.9,
                    )
                    us_norm = normalize_by_reference(
                        raw,
                        us_ic,
                        numerator_label=channel,
                        reference_label="US_IC",
                        denominator_floor=floor_us,
                        source_identity=scan.lightweight_identity(),
                        method="Fitted",
                    )
                    ds_norm = normalize_by_reference(
                        raw,
                        ds_ic,
                        numerator_label=channel,
                        reference_label="DS_IC",
                        denominator_floor=floor_ds,
                        source_identity=scan.lightweight_identity(),
                        method="Fitted",
                    )
                    channel_rows[channel] = {
                        "qc": qc.summary,
                        "us_ic_normalization": us_norm.summary,
                        "ds_ic_normalization": ds_norm.summary,
                    }

                rate_diag = count_rate_diagnostics(icr, ocr)
                live_diag = live_time_diagnostics(elt, ert)

                row = {
                    "scan": path.name,
                    "shape_yx": list(scan.shape),
                    "us_ic_floor_p01_positive": floor_us,
                    "ds_ic_floor_p01_positive": floor_ds,
                    "channels": channel_rows,
                    "ocr_over_icr": finite_summary(rate_diag["ocr_over_icr"]),
                    "implied_dead_time_fraction": finite_summary(
                        rate_diag["implied_dead_time_fraction"]
                    ),
                    "stored_dead_time": finite_summary(stored_dead_time),
                    "live_time_fraction": finite_summary(
                        live_diag["live_time_fraction"]
                    ),
                }
                rows.append(row)
                print(
                    f"PASS  {path.name:<23} shape={scan.shape!s:<12} "
                    f"US_IC p01={floor_us:.6g}  DS_IC p01={floor_ds:.6g}  "
                    f"Fe valid={channel_rows['Fe']['us_ic_normalization']['valid_fraction']:.4f}"
                )
        except Exception as exc:
            failures.append({"scan": path.name, "error": repr(exc)})
            print(f"FAIL  {path.name:<23} {exc}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "scan_count": len(files),
                "pass_count": len(rows),
                "failure_count": len(failures),
                "important_note": (
                    "US_IC/DS_IC normalization is validated as an available explicit "
                    "operation only. This report does not declare either reference the "
                    "scientifically preferred normalization."
                ),
                "results": rows,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("-" * 94)
    print(f"{len(rows)} / {len(files)} scans passed; {len(failures)} failure(s).")
    print(f"Wrote: {OUTPUT.relative_to(ROOT)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
