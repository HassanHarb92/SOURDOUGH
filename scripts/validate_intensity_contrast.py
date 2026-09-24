#!/usr/bin/env python3
'''Validate Change-06 feature maps on all current scans.'''

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from yeast_xrf.features.background import background_subtracted, gaussian_background
from yeast_xrf.features.intensity import asinh_feature, robust_zscore_feature
from yeast_xrf.features.local_stats import (
    local_contrast_z, local_cv, local_iqr, local_mad, local_mean, local_std
)
from yeast_xrf.io.xrf_scan import XRFScan
from yeast_xrf.visualization.transforms import clahe_display

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "intensity" / "intensity_contrast_validation.json"


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
    rows, failures = [], []
    files = sorted((ROOT / "img.dat").glob("*.h5"))
    print("Intensity + Contrast Real-Data Validation")
    print("=" * 88)

    for path in files:
        try:
            with XRFScan.open(path) as scan:
                channels = {}
                for name in ("P", "Fe", "Zn"):
                    raw = scan.map(name, "Fitted")
                    unit = scan.channel_info(name, "Fitted").unit or "unknown"
                    kw = {"input_label": name, "input_unit": unit}
                    features = {
                        "local_mean": local_mean(raw, window=7, **kw),
                        "local_std": local_std(raw, window=7, **kw),
                        "local_cv": local_cv(raw, window=7, **kw),
                        "local_mad": local_mad(raw, window=7, **kw),
                        "local_iqr": local_iqr(raw, window=7, **kw),
                        "local_contrast_z": local_contrast_z(raw, window=7, **kw),
                        "background": gaussian_background(raw, sigma_pixels=3.0, **kw),
                        "background_subtracted": background_subtracted(
                            raw, sigma_pixels=3.0, **kw
                        ),
                        "asinh": asinh_feature(raw, **kw),
                        "robust_zscore": robust_zscore_feature(raw, **kw),
                    }
                    for feature_name, result in features.items():
                        if result.data.shape != raw.shape:
                            raise ValueError(
                                f"{name}/{feature_name}: shape changed "
                                f"{result.data.shape} != {raw.shape}"
                            )
                    clahe = clahe_display(raw)
                    if clahe.shape != raw.shape:
                        raise ValueError(f"{name}/CLAHE shape changed")

                    channels[name] = {
                        key: {
                            "summary": summarize(value.data),
                            "unit": value.metadata.output_unit,
                            "kind": value.metadata.kind,
                        }
                        for key, value in features.items()
                    }
                    channels[name]["clahe"] = {
                        "summary": summarize(clahe),
                        "kind": "display",
                    }

                rows.append(
                    {"scan": path.name, "shape_yx": list(scan.shape), "channels": channels}
                )
                print(
                    f"PASS  {path.name:<23} shape={scan.shape!s:<12} "
                    "P/Fe/Zn feature families validated"
                )
        except Exception as exc:
            failures.append({"scan": path.name, "error": repr(exc)})
            print(f"FAIL  {path.name:<23} {exc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scan_count": len(files),
        "pass_count": len(rows),
        "failure_count": len(failures),
        "spatial_scale_note": "windows and sigma are pixel units only",
        "results": rows,
        "failures": failures,
    }, indent=2, sort_keys=True))

    print("-" * 88)
    print(f"{len(rows)} / {len(files)} scans passed; {len(failures)} failure(s).")
    print(f"Wrote: {OUT.relative_to(ROOT)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
