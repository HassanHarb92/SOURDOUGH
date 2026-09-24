#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yeast_xrf.analysis.maps_concentration import maps_concentration
from yeast_xrf.io.maps_concentration import (
    quantifiable_channels,
    read_maps_calibration_factor,
)
from yeast_xrf.io.xrf_scan import XRFScan


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT = ROOT / "analysis" / "quantification" / "change10_concentration_validation.json"

METHODS = ("Fitted", "NNLS", "ROI")
REFERENCES = ("US_IC", "DS_IC")


def summary(arr):
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "finite_count": int(finite.size),
        "finite_fraction": (
            float(finite.size / arr.size) if arr.size else 0.0
        ),
        "negative_count": int(np.count_nonzero(finite < 0)),
        "min": float(np.min(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def main():
    files = sorted(DATA.glob("*.h5"))
    failures = []
    rows = []

    print("SOURDOUGH Change 10 · automatic MAPS concentration validation")
    print("=" * 118)

    for path in files:
        scan_failures = 0
        with XRFScan.open(path) as scan:
            scalers = {
                "US_IC": scan.scaler("US_IC", "primary"),
                "DS_IC": scan.scaler("DS_IC", "primary"),
            }

            for method in METHODS:
                for reference in REFERENCES:
                    channels = quantifiable_channels(
                        path,
                        method=method,
                        reference=reference,
                    )
                    if not channels:
                        failures.append(
                            {
                                "scan": path.name,
                                "method": method,
                                "reference": reference,
                                "error": "no quantifiable channels",
                            }
                        )
                        scan_failures += 1
                        continue

                    for channel in channels:
                        try:
                            factor = read_maps_calibration_factor(
                                path,
                                method=method,
                                reference=reference,
                                channel=channel,
                            )
                            raw = scan.map(channel, method)
                            product = maps_concentration(
                                path,
                                raw,
                                scalers[reference],
                                method=method,
                                reference=reference,
                                channel=channel,
                                floor_percentile=1.0,
                            )
                            stats = summary(product.data)
                            if stats["finite_count"] == 0:
                                raise ValueError(
                                    "concentration product has no finite pixels"
                                )
                            rows.append(
                                {
                                    "scan": path.name,
                                    "method": method,
                                    "reference": reference,
                                    "channel": channel,
                                    "factor": factor.factor,
                                    "curve_value": factor.curve_value,
                                    "factor_curve_relative_error": factor.relative_error,
                                    "factor_row": factor.quant_row,
                                    "curve_position": [
                                        factor.curve_row,
                                        factor.curve_col,
                                    ],
                                    "unit": product.unit,
                                    "statistics": stats,
                                    "provenance": product.provenance,
                                }
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "scan": path.name,
                                    "method": method,
                                    "reference": reference,
                                    "channel": channel,
                                    "error": repr(exc),
                                }
                            )
                            scan_failures += 1

        print(
            f"{'PASS' if scan_failures == 0 else 'FAIL'}  "
            f"{path.name:<23} "
            f"products={sum(1 for r in rows if r['scan'] == path.name):>3} "
            f"failures={scan_failures}"
        )

    payload = {
        "scan_count": len(files),
        "product_count": len(rows),
        "failure_count": len(failures),
        "formula": (
            "areal_density_ug_cm2 = "
            "(counts_per_sec / reference_scaler) / calibration_factor"
        ),
        "supported_references": list(REFERENCES),
        "supported_methods": list(METHODS),
        "scientific_contract": {
            "raw_hdf5_modified": False,
            "negative_values_clipped": False,
            "automatic_factor_requires_curve_match": True,
            "output_unit": "µg/cm²",
            "pixel_sum_is_total_mass": False,
        },
        "products": rows,
        "failures": failures,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("-" * 118)
    print(f"Products validated: {len(rows)}")
    print(f"Failures:           {len(failures)}")
    print(f"Wrote: {OUT.relative_to(ROOT)}")

    if failures:
        print()
        print("First failures:")
        for failure in failures[:20]:
            print(" ", failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
