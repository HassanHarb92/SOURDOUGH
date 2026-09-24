import numpy as np
import pandas as pd
import pytest

from yeast_xrf.analysis.canonical_cells import (
    CanonicalizationBlockedError,
    canonicalization_status,
    finalize_canonical_cells,
)


def seed_run(
    run,
    decisions=("accept", "reject"),
    touches=(False, False),
):
    scan = "a.mda.h5"
    scan_dir = run / "scans" / "a"
    scan_dir.mkdir(parents=True)

    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[11:18, 11:18] = 2

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
                "consensus_score": 0.91,
                "automated_consensus_accept": True,
                "touches_scan_edge": touches[0],
            },
            {
                "scan": scan,
                "consensus_candidate_id": 2,
                "consensus_class": "accepted_cell",
                "consensus_score": 0.71,
                "automated_consensus_accept": True,
                "touches_scan_edge": touches[1],
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
                "review_decision": decisions[0],
                "review_reason": "clear_multichannel_cell",
                "review_notes": "",
                "reviewer": "tester",
                "reviewed_at_utc": "2026-01-01T00:00:00Z",
            },
            {
                "scan": scan,
                "consensus_candidate_id": 2,
                "review_decision": decisions[1],
                "review_reason": "artifact_like",
                "review_notes": "",
                "reviewer": "tester",
                "reviewed_at_utc": "2026-01-01T00:00:00Z",
            },
        ]
    ).to_csv(run / "cell_review.csv", index=False)


def test_only_review_accept_becomes_canonical(tmp_path):
    seed_run(tmp_path)
    result = finalize_canonical_cells(tmp_path)
    cells = result["canonical_cells"]

    assert len(cells) == 1
    assert cells.iloc[0]["canonical_cell_id"] == "CELL_000001"
    assert int(
        cells.iloc[0]["source_consensus_candidate_id"]
    ) == 1

    lineage = result["lineage"]
    rejected = lineage[
        lineage["consensus_candidate_id"].astype(int) == 2
    ].iloc[0]
    assert rejected["canonical_disposition"] == (
        "excluded_review_reject"
    )
    assert rejected["canonical_cell_id"] == ""


@pytest.mark.parametrize("unresolved", ["pending", "ambiguous"])
def test_unresolved_review_blocks_finalization(tmp_path, unresolved):
    seed_run(
        tmp_path,
        decisions=("accept", unresolved),
    )
    with pytest.raises(CanonicalizationBlockedError):
        finalize_canonical_cells(tmp_path)

    assert not (tmp_path / "canonical_cells.csv").exists()


def test_mask_lineage_is_exact(tmp_path):
    seed_run(tmp_path)
    result = finalize_canonical_cells(tmp_path)
    scan_npz = (
        tmp_path / "scans" / "a" / "canonical_cell_masks.npz"
    )
    with np.load(scan_npz, allow_pickle=False) as payload:
        labels = payload["canonical_labels"]

    assert set(np.unique(labels)) == {0, 1}
    assert int(np.count_nonzero(labels == 1)) == 36

    row = result["canonical_cells"].iloc[0]
    assert int(row["pixel_count"]) == 36
    assert row["canonical_mask_source"] == (
        "change13A_consensus_candidate_mask"
    )
    assert not bool(row["mask_edited_during_canonicalization"])


def test_border_cell_is_canonical_but_flagged(tmp_path):
    seed_run(
        tmp_path,
        decisions=("accept", "reject"),
        touches=(True, False),
    )
    result = finalize_canonical_cells(tmp_path)
    row = result["canonical_cells"].iloc[0]
    assert bool(row["touches_scan_edge"])
    assert not bool(row["default_complete_cell_population"])


def test_review_change_makes_snapshot_stale(tmp_path):
    seed_run(tmp_path)
    finalize_canonical_cells(tmp_path)
    assert canonicalization_status(tmp_path).finalized

    review = pd.read_csv(tmp_path / "cell_review.csv")
    review.loc[
        review["consensus_candidate_id"] == 1,
        "review_reason",
    ] = "automated_call_confirmed"
    review.to_csv(tmp_path / "cell_review.csv", index=False)

    status = canonicalization_status(tmp_path)
    assert status.stale
    assert not status.finalized
    assert not status.fresh


def test_global_and_scan_masks_are_written(tmp_path):
    seed_run(tmp_path)
    finalize_canonical_cells(tmp_path)
    assert (tmp_path / "canonical_cell_masks.npz").is_file()
    assert (
        tmp_path / "scans" / "a" / "canonical_cell_masks.npz"
    ).is_file()
    assert (tmp_path / "canonicalization_summary.json").is_file()
