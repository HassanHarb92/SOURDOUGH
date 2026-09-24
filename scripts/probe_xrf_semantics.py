#!/usr/bin/env python3
from pathlib import Path

from yeast_xrf.io.semantic_probe import probe_directory, summarize_probes, write_semantic_reports

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUTPUT = ROOT / "analysis" / "semantic"


def main() -> None:
    probes = probe_directory(DATA)
    summary = summarize_probes(probes)
    written = write_semantic_reports(probes, summary, OUTPUT)

    print("Yeast XRF Semantic Probe")
    print("=" * 56)
    print(f"Scans:                {summary['scan_count']}")
    print(f"Theta values:         {summary['theta']['count']}")
    print(f"Unique theta values:  {summary['theta']['unique_count']}")
    print(f"Theta span:           {summary['theta']['span']}")
    print()
    print("Tomography assessment:")
    print(f"  {summary['tomography_assessment']}")
    print()
    for method in ("Fitted", "NNLS", "ROI"):
        variants = summary["channel_name_variants"][method]
        print(
            f"{method:<8} channel variants={len(variants):<3} "
            f"most-common scans={(variants[0]['scan_count'] if variants else 0):<3} "
            f"channels={(len(variants[0]['names']) if variants else 0)}"
        )
    print()
    print("Wrote:")
    for key, path in written.items():
        print(f"  {key:<26} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
