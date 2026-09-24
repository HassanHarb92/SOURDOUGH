#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yeast_xrf.features.gradients import coordinate_gradient
from yeast_xrf.features.hessian import coordinate_hessian
from yeast_xrf.io.xrf_scan import XRFScan
from yeast_xrf.visualization.landscape import (
    build_landscape_height,
    display_surface_normals,
    landscape_figure,
)
from yeast_xrf.visualization.transforms import percentile_stretch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "landscape" / "cell_landscape_validation.json"


def main():
    files = sorted((ROOT / "img.dat").glob("*.h5"))
    rows, failures = [], []

    print("2.5D Yeast Cell Landscape Real-Data Validation")
    print("=" * 100)

    for path in files:
        try:
            with XRFScan.open(path) as scan:
                channels = set(scan.channel_names("Fitted"))
                height_name = (
                    "Total_Fluorescence_Yield"
                    if "Total_Fluorescence_Yield" in channels
                    else "P"
                )
                color_name = "Zn" if "Zn" in channels else height_name

                height = scan.map(height_name, "Fitted")
                color = scan.map(color_name, "Fitted")
                unit = scan.channel_info(height_name, "Fitted").unit or "unknown"

                grad = coordinate_gradient(
                    height,
                    scan.x,
                    scan.y,
                    input_label=height_name,
                    input_unit=unit,
                    sigma_pixels=1.0,
                )
                hess = coordinate_hessian(
                    height,
                    scan.x,
                    scan.y,
                    input_label=height_name,
                    input_unit=unit,
                    sigma_pixels=1.0,
                    quadratic_radius=2,
                )

                surface = build_landscape_height(
                    height,
                    low_percentile=1.0,
                    high_percentile=99.0,
                    vertical_exaggeration=4.0,
                    background_mask_percentile=20.0,
                    curvature=hess.curvedness,
                    curvature_weight=0.75,
                )

                surface_grad = coordinate_gradient(
                    surface.z_unmasked,
                    scan.x,
                    scan.y,
                    input_label="display height",
                    input_unit="display-height",
                    sigma_pixels=0.0,
                )
                nx, ny, nz = display_surface_normals(
                    surface_grad.ix,
                    surface_grad.iy,
                )

                color_display = percentile_stretch(color, low=1.0, high=99.0)
                fig = landscape_figure(
                    x=scan.x,
                    y=scan.y,
                    surface=surface,
                    surface_color=color_display,
                    raw_height=height,
                    raw_color=color,
                    title="validation",
                    colorbar_title=color_name,
                    ridge_response=hess.bright_ridge,
                    blob_response=hess.bright_blob,
                )

                if surface.z.shape != scan.shape:
                    raise ValueError("landscape surface changed raster shape")
                if color_display.shape != scan.shape:
                    raise ValueError("landscape color changed raster shape")
                if nx.shape != scan.shape or ny.shape != scan.shape or nz.shape != scan.shape:
                    raise ValueError("surface normal changed raster shape")
                if not fig.data:
                    raise ValueError("Plotly landscape figure contains no traces")

                finite_z = surface.z[np.isfinite(surface.z)]
                if finite_z.size == 0:
                    raise ValueError("landscape contains no visible finite surface")

                rows.append(
                    {
                        "scan": path.name,
                        "shape_yx": list(scan.shape),
                        "height_channel": height_name,
                        "color_channel": color_name,
                        "visible_fraction": float(np.mean(surface.visible_mask)),
                        "gradient_finite_fraction": float(
                            np.mean(np.isfinite(grad.magnitude))
                        ),
                        "curvedness_finite_fraction": float(
                            np.mean(np.isfinite(hess.curvedness))
                        ),
                        "figure_trace_count": len(fig.data),
                        "physical_z": surface.metadata["physical_z"],
                        "background_mask_is_segmentation": surface.metadata[
                            "background_mask_is_segmentation"
                        ],
                    }
                )
                print(
                    f"PASS  {path.name:<23} shape={scan.shape!s:<12} "
                    f"height={height_name:<26} color={color_name:<4} "
                    f"visible={np.mean(surface.visible_mask):.3f}"
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
                "physical_z": False,
                "interpretation": (
                    "2.5D visualization geometry derived from 2D XRF maps; "
                    "not volumetric reconstruction"
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
