#!/usr/bin/env python3
from pathlib import Path
import tempfile

import pandas as pd

from yeast_xrf.analysis.cell_review import (
    ensure_cell_review,
    review_summary,
    save_cell_review,
)


def main():
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        consensus = pd.DataFrame(
            [
                {
                    "scan": "synthetic.h5",
                    "consensus_candidate_id": 1,
                    "consensus_class": "high_confidence_cell",
                    "consensus_score": 0.9,
                    "automated_consensus_accept": True,
                },
                {
                    "scan": "synthetic.h5",
                    "consensus_candidate_id": 2,
                    "consensus_class": (
                        "ambiguous_cell_candidate"
                    ),
                    "consensus_score": 0.5,
                    "automated_consensus_accept": False,
                },
            ]
        )
        table = ensure_cell_review(run, consensus)
        save_cell_review(
            run,
            scan="synthetic.h5",
            consensus_candidate_id=1,
            decision="accept",
            reason="clear_multichannel_cell",
            reviewer="validator",
        )
        table = ensure_cell_review(run, consensus)
        summary = review_summary(table)

        checks = {
            "table_written": (
                run / "cell_review.csv"
            ).is_file(),
            "one_reviewed": summary["reviewed"] == 1,
            "one_pending": summary["pending"] == 1,
            "accept_persisted": summary["accept"] == 1,
            "not_canonicalized": (
                summary["canonical_cells_finalized"] is False
            ),
        }

        for name, passed in checks.items():
            print(
                f"{'PASS' if passed else 'FAIL'}  {name}"
            )

        if not all(checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
