#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yeast_xrf.analysis.beam_quantification import (
    beam_normalize,
    normalization_comparison,
    transmission_diagnostic,
)
from yeast_xrf.io.maps_quantification import inspect_maps_quantification
from yeast_xrf.io.xrf_scan import XRFScan


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT = ROOT / "analysis" / "quantification" / "change09_validation.json"


def finite_summary(data):
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "finite_count": int(finite.size),
        "finite_fraction": float(finite.size / arr.size) if arr.size else 0.0,
        "min": float(np.min(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def main():
    files = sorted(DATA.glob("*.h5"))
    rows = []
    failures = []

    print("SOURDOUGH Change 09 · Beam normalization + quantification readiness")
    print("=" * 108)

    for path in files:
        try:
            with XRFScan.open(path) as scan:
                us = scan.scaler("US_IC", "primary")
                ds = scan.scaler("DS_IC", "primary")

                channels = {}
                for channel in ("P", "S", "K", "Ca", "Fe", "Cu", "Zn"):
                    raw = scan.map(channel, "Fitted")
                    us_product = beam_normalize(
                        raw,
                        us,
                        numerator_label=channel,
                        reference_label="US_IC",
                        floor_percentile=1.0,
                        method="Fitted",
                        source_identity=scan.lightweight_identity(),
                    )
                    ds_product = beam_normalize(
                        raw,
                        ds,
                        numerator_label=channel,
                        reference_label="DS_IC",
                        floor_percentile=1.0,
                        method="Fitted",
                        source_identity=scan.lightweight_identity(),
                    )
                    channels[channel] = {
                        "us_ic": finite_summary(us_product.data),
                        "ds_ic": finite_summary(ds_product.data),
                        "comparison": normalization_comparison(
                            us_product.data,
                            ds_product.data,
                        ),
                    }

                transmission = transmission_diagnostic(us, ds)
                quant = inspect_maps_quantification(
                    path,
                    method="Fitted",
                    reference="US_IC",
                )

                rows.append(
                    {
                        "scan": path.name,
                        "shape_yx": list(scan.shape),
                        "channels": channels,
                        "transmission": finite_summary(transmission["data"]),
                        "standard": quant["standard"],
                        "calibration_shapes": quant["shapes"],
                        "unit_candidates": quant["unit_candidates"],
                        "readiness": quant["readiness"],
                    }
                )

                print(
                    f"PASS  {path.name:<23} "
                    f"standard={quant['readiness']['standard_metadata_present']} "
                    f"curve={quant['readiness']['calibration_curve_present']} "
                    f"element-info={quant['readiness']['element_info_present']} "
                    f"auto-concentration="
                    f"{quant['readiness']['automatic_areal_density_conversion_ready']}"
                )

        except Exception as exc:
            failures.append({"scan": path.name, "error": repr(exc)})
            print(f"FAIL  {path.name:<23} {exc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "scan_count": len(files),
                "pass_count": len(rows),
                "failure_count": len(failures),
                "scientific_boundary": (
                    "Beam normalization is operational. Automatic conversion to areal "
                    "density remains locked until the stored MAPS calibration coefficient "
                    "semantics are verified."
                ),
                "results": rows,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("-" * 108)
    print(f"{len(rows)} / {len(files)} scans passed; {len(failures)} failure(s).")
    print(f"Wrote: {OUT.relative_to(ROOT)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
