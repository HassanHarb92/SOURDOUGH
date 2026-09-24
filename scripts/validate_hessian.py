#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from yeast_xrf.features.hessian import coordinate_hessian
from yeast_xrf.io.xrf_scan import XRFScan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "hessian" / "hessian_validation.json"


def summarize(data):
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "finite_fraction": float(finite.size / arr.size) if arr.size else 0.0,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
    }


def main():
    files = sorted((ROOT / "img.dat").glob("*.h5"))
    rows, failures = [], []

    print("Hessian + Curvature Real-Data Validation")
    print("=" * 100)

    for path in files:
        try:
            with XRFScan.open(path) as scan:
                channels = {}
                for name in ("P", "Fe", "Zn"):
                    raw = scan.map(name, "Fitted")
                    unit = scan.channel_info(name, "Fitted").unit or "unknown"
                    h = coordinate_hessian(
                        raw,
                        scan.x,
                        scan.y,
                        input_label=name,
                        input_unit=unit,
                        sigma_pixels=1.0,
                        quadratic_radius=2,
                    )

                    products = {
                        "ixx": h.ixx,
                        "iyy": h.iyy,
                        "ixy": h.ixy,
                        "mixed_disagreement": h.mixed_disagreement,
                        "laplacian": h.laplacian,
                        "determinant": h.determinant,
                        "lambda_min": h.lambda_min,
                        "lambda_max": h.lambda_max,
                        "principal_orientation": h.principal_orientation_deg,
                        "curvedness": h.curvedness,
                        "shape_index": h.shape_index,
                        "bright_ridge": h.bright_ridge,
                        "dark_valley": h.dark_valley,
                        "bright_blob": h.bright_blob,
                        "dark_blob": h.dark_blob,
                    }
                    for key, value in products.items():
                        if value.shape != raw.shape:
                            raise ValueError(
                                f"{name}/{key} shape {value.shape} != {raw.shape}"
                            )

                    mixed = h.mixed_disagreement[np.isfinite(h.mixed_disagreement)]
                    channels[name] = {
                        key: summarize(value) for key, value in products.items()
                    }
                    channels[name]["mixed_qc"] = {
                        "median_absolute_disagreement": (
                            float(np.median(mixed)) if mixed.size else None
                        ),
                        "max_absolute_disagreement": (
                            float(np.max(mixed)) if mixed.size else None
                        ),
                    }
                    channels[name]["y_axis"] = (
                        h.metadata["iyy"].provenance["y_axis"]
                    )

                rows.append(
                    {
                        "scan": path.name,
                        "shape_yx": list(scan.shape),
                        "channels": channels,
                    }
                )
                repeated = channels["Fe"]["y_axis"]["repeated_adjacent_count"]
                print(
                    f"PASS  {path.name:<23} shape={scan.shape!s:<12} "
                    f"P/Fe/Zn  repeated-Y={repeated}"
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
                "coordinate_unit": None,
                "note": (
                    "Ixx/Iyy use local quadratic fits against actual coordinate values. "
                    "Ixy is the symmetric average of the two mixed-derivative paths."
                ),
                "results": rows,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("-" * 100)
    print(f"{len(rows)} / {len(files)} scans passed; {len(failures)} failure(s).")
    print(f"Wrote: {OUT.relative_to(ROOT)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
