#!/usr/bin/env python3
from pathlib import Path
import argparse

from yeast_xrf.analysis.canonical_cells import (
    CanonicalizationBlockedError,
    canonicalization_status,
    finalize_canonical_cells,
    preview_canonicalization,
)


ROOT = Path(__file__).resolve().parents[1]


def resolve_run(value):
    if value:
        return Path(value).expanduser().resolve()

    pointer = ROOT / "analysis" / "studies" / "latest_run.txt"
    if not pointer.is_file():
        raise SystemExit(
            "No run supplied and latest_run.txt is missing."
        )
    return Path(pointer.read_text().strip()).resolve()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Finalize SOURDOUGH reviewed consensus candidates "
            "into the canonical cell dataset."
        )
    )
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status only; do not finalize.",
    )
    args = parser.parse_args()

    run = resolve_run(args.run_dir)
    preview = preview_canonicalization(run)

    print("SOURDOUGH · Canonical Cells")
    print("=" * 72)
    print(f"Run:       {run}")
    print(f"Accept:    {preview['accepted_count']}")
    print(f"Reject:    {preview['rejected_count']}")
    print(f"Ambiguous: {preview['ambiguous_count']}")
    print(f"Pending:   {preview['pending_count']}")
    print()

    if args.status:
        status = canonicalization_status(run)
        print(f"Finalized: {status.finalized}")
        print(f"Fresh:     {status.fresh}")
        print(f"Stale:     {status.stale}")
        print(f"Reason:    {status.reason}")
        return

    try:
        result = finalize_canonical_cells(run)
    except CanonicalizationBlockedError as exc:
        raise SystemExit(str(exc))

    print("Canonical cells:", len(result["canonical_cells"]))
    print("Wrote:", run / "canonical_cells.csv")


if __name__ == "__main__":
    main()
