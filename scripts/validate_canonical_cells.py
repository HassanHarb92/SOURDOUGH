#!/usr/bin/env python3
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from yeast_xrf.analysis.canonical_cells import (
    canonicalization_status,
    finalize_canonical_cells,
)


def main():
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        scan = "synthetic.mda.h5"
        scan_dir = run / "scans" / "synthetic"
        scan_dir.mkdir(parents=True)

        labels = np.zeros((30, 30), dtype=np.int32)
        labels[3:10, 3:10] = 1
        labels[15:24, 15:24] = 2
        np.savez_compressed(
            scan_dir / "cell_masks_multichannel_consensus.npz",
            candidate_labels=labels,
            accepted_labels=labels,
            support_count_map=(labels > 0).astype(np.int32),
        )

        pd.DataFrame(
            [
                {
                    "scan": scan,
                    "consensus_candidate_id": 1,
                    "consensus_class": "high_confidence_cell",
                    "consensus_score": 0.9,
                    "automated_consensus_accept": True,
                    "touches_scan_edge": False,
                },
                {
                    "scan": scan,
                    "consensus_candidate_id": 2,
                    "consensus_class": "accepted_cell",
                    "consensus_score": 0.7,
                    "automated_consensus_accept": True,
                    "touches_scan_edge": False,
                },
            ]
        ).to_csv(run / "cell_consensus.csv", index=False)

        pd.DataFrame(
            [
                {"scan": scan, "consensus_candidate_id": 1, "channel": "P"},
                {"scan": scan, "consensus_candidate_id": 2, "channel": "P"},
            ]
        ).to_csv(run / "cell_channel_support.csv", index=False)

        pd.DataFrame(
            [
                {
                    "scan": scan,
                    "consensus_candidate_id": 1,
                    "review_decision": "accept",
                    "review_reason": "clear_multichannel_cell",
                    "review_notes": "",
                    "reviewer": "validator",
                    "reviewed_at_utc": "2026-01-01T00:00:00Z",
                },
                {
                    "scan": scan,
                    "consensus_candidate_id": 2,
                    "review_decision": "reject",
                    "review_reason": "artifact_like",
                    "review_notes": "",
                    "reviewer": "validator",
                    "reviewed_at_utc": "2026-01-01T00:00:00Z",
                },
            ]
        ).to_csv(run / "cell_review.csv", index=False)

        result = finalize_canonical_cells(run)
        status = canonicalization_status(run)

        checks = {
            "canonical_csv": (run / "canonical_cells.csv").is_file(),
            "lineage_csv": (run / "canonical_cell_lineage.csv").is_file(),
            "study_masks": (run / "canonical_cell_masks.npz").is_file(),
            "scan_masks": (
                scan_dir / "canonical_cell_masks.npz"
            ).is_file(),
            "one_canonical": len(result["canonical_cells"]) == 1,
            "accepted_only": (
                result["canonical_cells"][
                    "source_consensus_candidate_id"
                ].astype(int).tolist()
                == [1]
            ),
            "status_finalized": status.finalized,
            "status_fresh": status.fresh,
        }

        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}  {name}")

        if not all(checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
