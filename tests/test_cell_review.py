from pathlib import Path

import pandas as pd

from yeast_xrf.analysis.cell_review import (
    ensure_cell_review,
    review_summary,
    save_cell_review,
)


def consensus():
    return pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "consensus_candidate_id": 1,
                "consensus_class": "high_confidence_cell",
                "consensus_score": 0.91,
                "automated_consensus_accept": True,
            },
            {
                "scan": "a.h5",
                "consensus_candidate_id": 2,
                "consensus_class": (
                    "rejected_channel_specific_object"
                ),
                "consensus_score": 0.22,
                "automated_consensus_accept": False,
            },
        ]
    )


def test_review_initializes_pending_rows(tmp_path):
    table = ensure_cell_review(
        tmp_path,
        consensus(),
    )
    assert len(table) == 2
    assert set(table["review_decision"]) == {"pending"}
    assert not table["canonical_cell"].astype(bool).any()
    assert (tmp_path / "cell_review.csv").is_file()


def test_saved_decision_persists(tmp_path):
    # Initialize from an in-memory consensus table only. save_cell_review()
    # must then use the persisted review table rather than requiring an
    # on-disk cell_consensus.csv.
    ensure_cell_review(tmp_path, consensus())
    assert not (tmp_path / "cell_consensus.csv").exists()

    save_cell_review(
        tmp_path,
        scan="a.h5",
        consensus_candidate_id=1,
        decision="accept",
        reason="clear_multichannel_cell",
        notes="round cell, strong agreement",
        reviewer="HH",
    )

    again = ensure_cell_review(
        tmp_path,
        consensus(),
    )
    row = again[
        again["consensus_candidate_id"].astype(int) == 1
    ].iloc[0]

    assert row["review_decision"] == "accept"
    assert row["review_reason"] == "clear_multichannel_cell"
    assert row["review_notes"] == (
        "round cell, strong agreement"
    )
    assert row["reviewer"] == "HH"
    assert row["reviewed_at_utc"]
    assert not bool(row["canonical_cell"])


def test_repeated_save_does_not_duplicate_candidate(tmp_path):
    ensure_cell_review(tmp_path, consensus())
    for decision in ("ambiguous", "accept"):
        save_cell_review(
            tmp_path,
            scan="a.h5",
            consensus_candidate_id=1,
            decision=decision,
            reason=(
                "uncertain"
                if decision == "ambiguous"
                else "automated_call_confirmed"
            ),
        )

    table = ensure_cell_review(tmp_path, consensus())
    assert len(table) == 2
    row = table[
        table["consensus_candidate_id"].astype(int) == 1
    ].iloc[0]
    assert row["review_decision"] == "accept"


def test_review_summary(tmp_path):
    table = ensure_cell_review(tmp_path, consensus())
    save_cell_review(
        tmp_path,
        scan="a.h5",
        consensus_candidate_id=1,
        decision="accept",
        reason="clear_multichannel_cell",
    )
    table = ensure_cell_review(tmp_path, consensus())
    s = review_summary(table)
    assert s["candidate_count"] == 2
    assert s["reviewed"] == 1
    assert s["accept"] == 1
    assert s["pending"] == 1
    assert s["review_fraction"] == 0.5
    assert s["canonical_cells_finalized"] is False
