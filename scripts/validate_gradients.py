#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from yeast_xrf.features.gradients import (
    coordinate_gradient, directional_derivative, edge_mask, pixel_operator_gradient
)
from yeast_xrf.io.xrf_scan import XRFScan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "gradients" / "gradient_validation.json"

def summarize(a):
    a = np.asarray(a, float)
    f = a[np.isfinite(a)]
    return {
        "finite_fraction": float(f.size/a.size) if a.size else 0.0,
        "min": float(np.min(f)) if f.size else None,
        "max": float(np.max(f)) if f.size else None,
        "median": float(np.median(f)) if f.size else None,
    }

def main():
    files = sorted((ROOT/"img.dat").glob("*.h5"))
    rows, failures = [], []
    print("Gradient + Edge Real-Data Validation")
    print("="*92)
    for path in files:
        try:
            with XRFScan.open(path) as scan:
                ch = {}
                for name in ("P","Fe","Zn"):
                    raw = scan.map(name,"Fitted")
                    unit = scan.channel_info(name,"Fitted").unit or "unknown"
                    cg = coordinate_gradient(
                        raw, scan.x, scan.y, input_label=name,
                        input_unit=unit, sigma_pixels=1.0
                    )
                    dd = directional_derivative(cg, angle_deg=45, input_label=name)
                    em = edge_mask(cg.magnitude, percentile=90, input_label=name)
                    sob = pixel_operator_gradient(
                        raw, operator="sobel", input_label=name, input_unit=unit
                    )
                    sch = pixel_operator_gradient(
                        raw, operator="scharr", input_label=name, input_unit=unit
                    )
                    pre = pixel_operator_gradient(
                        raw, operator="prewitt", input_label=name, input_unit=unit
                    )
                    products = {
                        "ix": cg.ix, "iy": cg.iy, "magnitude": cg.magnitude,
                        "orientation": cg.orientation_deg,
                        "directional45": dd.data, "edge_mask90": em.data,
                        "sobel": sob.magnitude, "scharr": sch.magnitude,
                        "prewitt": pre.magnitude,
                    }
                    for key,val in products.items():
                        if val.shape != raw.shape:
                            raise ValueError(f"{name}/{key} changed shape")
                    ch[name] = {k:summarize(v) for k,v in products.items()}
                rows.append({"scan":path.name,"shape_yx":list(scan.shape),"channels":ch})
                print(f"PASS  {path.name:<23} shape={scan.shape!s:<12} P/Fe/Zn")
        except Exception as exc:
            failures.append({"scan":path.name,"error":repr(exc)})
            print(f"FAIL  {path.name:<23} {exc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scan_count":len(files), "pass_count":len(rows),
        "failure_count":len(failures), "coordinate_unit":None,
        "note":"coordinate gradients use exact stored X/Y vectors",
        "results":rows, "failures":failures
    }, indent=2, sort_keys=True))
    print("-"*92)
    print(f"{len(rows)} / {len(files)} scans passed; {len(failures)} failure(s).")
    print(f"Wrote: {OUT.relative_to(ROOT)}")
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
