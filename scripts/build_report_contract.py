#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

from yeast_xrf.reporting.report_contract import (
    refresh_report_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def resolve_run(value):
    if value:
        return Path(value).expanduser().resolve()

    pointer = ROOT / "analysis" / "studies" / "latest_run.txt"
    if not pointer.is_file():
        raise SystemExit(
            "No run path supplied and latest_run.txt does not exist."
        )
    return Path(pointer.read_text().strip()).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Study run directory; defaults to latest_run.txt",
    )
    args = parser.parse_args()

    run = resolve_run(args.run_dir)
    manifest = refresh_report_contract(run)

    print("SOURDOUGH Report Contract")
    print("=" * 72)
    print(f"Run: {run}")
    print(
        f"Final report ready: "
        f"{manifest['final_report_ready']}"
    )
    for section in manifest["sections"]:
        print(
            f"{section['status'].upper():8s} "
            f"{section['title']}"
        )
    print()
    print(
        "Manifest:",
        run / "report" / "report_manifest.json",
    )


if __name__ == "__main__":
    main()
