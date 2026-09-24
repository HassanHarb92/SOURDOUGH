#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from yeast_xrf.analysis.batch import analyze_directory


def parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze every MAPS HDF5 scan in a directory with SOURDOUGH. "
            "Full cells are analyzed; border-cropped cells are reported and excluded."
        )
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("analysis/batch"))
    parser.add_argument(
        "--methods",
        type=parse_csv,
        default=("Fitted", "NNLS", "ROI"),
    )
    parser.add_argument("--tfy-method", default="Fitted")
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "yen", "li", "percentile"),
        default="otsu",
    )
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--no-split-touching", action="store_true")
    parser.add_argument("--watershed-min-distance", type=int, default=5)
    parser.add_argument(
        "--structural-channels",
        type=parse_csv,
        default=("P", "Fe", "Zn"),
    )
    parser.add_argument("--structural-sigma", type=float, default=1.0)
    parser.add_argument("--hessian-radius", type=int, default=2)
    parser.add_argument("--no-centroid-spectra", action="store_true")
    args = parser.parse_args()

    result = analyze_directory(
        args.input_dir,
        output_root=args.output_root,
        methods=args.methods,
        tfy_method=args.tfy_method,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        smooth_sigma_pixels=args.smooth_sigma,
        min_area_pixels=args.min_area,
        closing_radius_pixels=args.closing_radius,
        split_touching=not args.no_split_touching,
        watershed_min_distance_pixels=args.watershed_min_distance,
        structural_channels=args.structural_channels,
        structural_sigma_pixels=args.structural_sigma,
        hessian_radius=args.hessian_radius,
        centroid_spectra=not args.no_centroid_spectra,
    )

    print("\nSOURDOUGH directory analysis complete")
    print("=" * 72)
    for key, value in result["summary"].items():
        print(f"{key:32} {value}")


if __name__ == "__main__":
    main()
