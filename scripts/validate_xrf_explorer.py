#!/usr/bin/env python3
"""Exercise Change-04 explorer primitives against all 12 real scans.

This does not launch Streamlit. It validates the exact IO and display operations the UI uses:
selected maps, display transforms, descriptive stats, a scaler map, and one center spectrum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yeast_xrf.io.xrf_scan import XRFScan
from yeast_xrf.visualization.statistics import map_statistics
from yeast_xrf.visualization.transforms import display_transform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUTPUT = ROOT / "analysis" / "explorer" / "explorer_smoke_validation.json"


def main() -> None:
    files = sorted(DATA.glob("*.h5"))
    results = []
    failures = []

    print("XRF Explorer Real-Data Smoke Validation")
    print("=" * 82)

    for path in files:
        try:
            with XRFScan.open(path) as scan:
                channels = scan.channel_names("Fitted")
                test_channels = [
                    name for name in ("P", "Fe", "Zn", "Fit_Residual") if name in channels
                ]
                channel_results = {}

                for name in test_channels:
                    raw = scan.map(name, "Fitted")
                    if raw.shape != scan.shape:
                        raise ValueError(f"{name} shape {raw.shape} != {scan.shape}")

                    pct = display_transform(raw, "Percentile stretch").data
                    slog = display_transform(raw, "Signed log1p").data
                    if pct.shape != raw.shape or slog.shape != raw.shape:
                        raise ValueError(f"display transform changed {name} shape")

                    channel_results[name] = map_statistics(raw)

                scaler = scan.scaler("US_IC", "primary")
                if scaler.shape != scan.shape:
                    raise ValueError(f"US_IC shape {scaler.shape} != {scan.shape}")

                yi = scan.shape[0] // 2
                xi = scan.shape[1] // 2
                spectrum = scan.spectrum(y=yi, x=xi)
                if spectrum.shape != (scan.energy.size,):
                    raise ValueError(
                        f"spectrum shape {spectrum.shape} != {(scan.energy.size,)}"
                    )

                row = {
                    "scan": path.name,
                    "status": "PASS",
                    "shape_yx": list(scan.shape),
                    "channels_tested": test_channels,
                    "center_pixel": [yi, xi],
                    "spectrum_points": int(spectrum.size),
                    "us_ic_finite_fraction": float(np.isfinite(scaler).mean()),
                    "channel_statistics": channel_results,
                }
                results.append(row)
                print(
                    f"PASS  {path.name:<23} shape={scan.shape!s:<12} "
                    f"maps={len(test_channels):<2} spectrum={spectrum.size:<5} scaler=US_IC"
                )
        except Exception as exc:
            failures.append({"scan": path.name, "error": repr(exc)})
            print(f"FAIL  {path.name:<23} {exc}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "scan_count": len(files),
                "pass_count": len(results),
                "failure_count": len(failures),
                "results": results,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("-" * 82)
    print(
        f"{len(results)} / {len(files)} scans passed; "
        f"{len(failures)} failure(s)."
    )
    print(f"Wrote: {OUTPUT.relative_to(ROOT)}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
