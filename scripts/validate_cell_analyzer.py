#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yeast_xrf.features.cells import segment_cells_tfy
from yeast_xrf.io.xrf_scan import XRFScan
from yeast_xrf.visualization.cell_model import (
    build_mask_conforming_envelope,
    cell_model_figure,
    internal_slice,
    projection_conserving_density,
)
from yeast_xrf.visualization.transforms import percentile_stretch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "cells" / "cell_analyzer_validation.json"


def main():
    files = sorted((ROOT / "img.dat").glob("*.h5"))
    rows, failures = [], []

    print("TFY Cell Analyzer + Inferred 3D Model Validation")
    print("=" * 108)

    for path in files:
        try:
            with XRFScan.open(path) as scan:
                tfy = scan.map("Total_Fluorescence_Yield", "Fitted")
                result = segment_cells_tfy(
                    tfy,
                    scan.x,
                    scan.y,
                    threshold_method="otsu",
                    smooth_sigma_pixels=1.0,
                    min_area_pixels=25,
                    closing_radius_pixels=1,
                    fill_holes=True,
                    split_touching=True,
                    watershed_min_distance_pixels=5,
                )

                model_status = "no-complete-cell"
                model_cell_id = None
                if result.complete_cells:
                    record = next(r for r in result.records if not r.cropped)
                    cell_mask = result.labels == record.cell_id
                    env = build_mask_conforming_envelope(
                        cell_mask,
                        scan.x,
                        scan.y,
                        depth_ratio_to_minor_radius=1.0,
                        minor_axis_coordinate=record.minor_axis_coordinate_approx,
                    )

                    zn = scan.map("Zn", "Fitted")
                    zn_cell = np.where(cell_mask, zn, np.nan)
                    density, _ = projection_conserving_density(zn_cell, env)
                    slice_data, slice_z, _ = internal_slice(
                        density, env, z_fraction=0.0
                    )
                    fig = cell_model_figure(
                        x=scan.x,
                        y=scan.y,
                        envelope=env,
                        surface_color=percentile_stretch(zn_cell),
                        raw_surface_color=zn_cell,
                        slice_data=slice_data,
                        slice_z=slice_z,
                        title="validation",
                    )
                    if not fig.data:
                        raise ValueError("3D model figure has no traces")
                    model_status = "PASS"
                    model_cell_id = record.cell_id

                rows.append(
                    {
                        "scan": path.name,
                        "shape_yx": list(scan.shape),
                        "detected_cells": result.total_cells,
                        "complete_cells": result.complete_cells,
                        "cropped_cells": result.cropped_cells,
                        "modeled_cell_id": model_cell_id,
                        "model_status": model_status,
                        "physical_unit_verified": result.metadata[
                            "physical_unit_verified"
                        ],
                    }
                )

                print(
                    f"PASS  {path.name:<23} "
                    f"cells={result.total_cells:<3} "
                    f"complete={result.complete_cells:<3} "
                    f"cropped={result.cropped_cells:<3} "
                    f"3D-model={model_status}"
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
                "ground_truth_cell_counts_available": False,
                "interpretation": (
                    "automatic TFY segmentation candidates; border-touching components "
                    "flagged as cropped; depth remains model-based"
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
