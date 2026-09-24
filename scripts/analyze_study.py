#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from yeast_xrf.analysis.study_analysis import analyze_study


ROOT = Path(__file__).resolve().parents[1]


def csv_tuple(value):
    return tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SOURDOUGH's automated study analysis on a directory "
            "of MAPS HDF5 scans."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(ROOT / "img.dat"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "analysis" / "studies"),
    )
    parser.add_argument("--metadata-csv")
    parser.add_argument(
        "--methods",
        default="Fitted,NNLS,ROI",
    )
    parser.add_argument(
        "--references",
        default="US_IC,DS_IC",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--no-arrays",
        action="store_true",
    )
    parser.add_argument(
        "--no-pairs",
        action="store_true",
    )
    parser.add_argument(
        "--no-artifact-screening",
        action="store_true",
        help=(
            "Disable flag-only artifact/contamination candidate screening."
        ),
    )
    parser.add_argument(
        "--no-cell-consensus",
        action="store_true",
        help="Disable automated TFY/P/S/K cell consensus.",
    )
    args = parser.parse_args()

    result = analyze_study(
        args.input_dir,
        output_root=args.output_root,
        metadata_csv=args.metadata_csv,
        methods=csv_tuple(args.methods),
        references=csv_tuple(args.references),
        save_concentration_arrays=not args.no_arrays,
        compute_element_pairs=not args.no_pairs,
        artifact_screening=not args.no_artifact_screening,
        multichannel_consensus=not args.no_cell_consensus,
        max_scans=args.max_scans,
    )

    s = result["summary"]
    print()
    print("SOURDOUGH · Automated Study Analysis")
    print("=" * 72)
    print(f"Run:                  {s['study_run_id']}")
    print(f"Scans discovered:     {s['scans_discovered']}")
    print(f"Scans analyzed:       {s['scans_analyzed']}")
    print(f"Failures:             {s['scan_failures']}")
    print(f"Cells detected:       {s['detected_cells']}")
    print(f"Full cells:           {s['full_cells']}")
    print(f"Cropped cells:        {s['cropped_cells']}")
    print(f"Cell-element records: {s['cell_element_records']}")
    print(f"Element-pair records: {s['element_pair_records']}")
    print(f"QC flags:             {s['qc_flags']}")
    print()
    print("Cell consensus: TFY/P/S/K; human review pending")
    print("Concentrations: validated MAPS µg/cm²")
    print(f"Output: {result['run_dir']}")


if __name__ == "__main__":
    main()
