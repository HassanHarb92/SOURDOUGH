import numpy as np
import pandas as pd

from yeast_xrf.analysis.canonical_cells import (
    finalize_canonical_cells,
)
from yeast_xrf.reporting.report_contract import (
    refresh_report_contract,
)


def seed(run):
    scan = "a.mda.h5"
    scan_dir = run / "scans" / "a"
    scan_dir.mkdir(parents=True)

    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:8, 2:8] = 1
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
            }
        ]
    ).to_csv(run / "cell_consensus.csv", index=False)

    pd.DataFrame(
        [{"scan": scan, "consensus_candidate_id": 1, "channel": "P"}]
    ).to_csv(run / "cell_channel_support.csv", index=False)

    pd.DataFrame(
        [
            {
                "scan": scan,
                "consensus_candidate_id": 1,
                "review_decision": "accept",
                "review_reason": "clear_multichannel_cell",
                "review_notes": "",
                "reviewer": "tester",
                "reviewed_at_utc": "2026-01-01T00:00:00Z",
            }
        ]
    ).to_csv(run / "cell_review.csv", index=False)

    pd.DataFrame(
        [{"scan": scan, "include": True}]
    ).to_csv(run / "study_manifest.csv", index=False)
    pd.DataFrame(
        [{"scan": scan, "status": "analyzed"}]
    ).to_csv(run / "scans.csv", index=False)
    (run / "study_summary.json").write_text("{}")
    (run / "study_config.json").write_text("{}")
    (run / "study_provenance.json").write_text("{}")


def section(manifest, sid):
    return next(
        item
        for item in manifest["sections"]
        if item["id"] == sid
    )


def test_cell_identification_waits_for_canonicalization(tmp_path):
    seed(tmp_path)
    before = refresh_report_contract(tmp_path)
    assert section(before, "cell_identification")["status"] == "partial"

    finalize_canonical_cells(tmp_path)
    after = refresh_report_contract(tmp_path)
    assert section(after, "cell_identification")["status"] == "ready"


def test_review_edit_returns_report_to_partial(tmp_path):
    seed(tmp_path)
    finalize_canonical_cells(tmp_path)
    assert section(
        refresh_report_contract(tmp_path),
        "cell_identification",
    )["status"] == "ready"

    review = pd.read_csv(tmp_path / "cell_review.csv")
    review.loc[0, "review_reason"] = "automated_call_confirmed"
    review.to_csv(tmp_path / "cell_review.csv", index=False)

    sec = section(
        refresh_report_contract(tmp_path),
        "cell_identification",
    )
    assert sec["status"] == "partial"
    assert "stale" in sec["status_note"].lower()
