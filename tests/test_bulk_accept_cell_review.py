import pandas as pd

from yeast_xrf.analysis.cell_review import (
    bulk_accept_pending_reviews,
    ensure_cell_review,
)


def consensus():
    return pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "consensus_candidate_id": 1,
                "consensus_class": "high_confidence_cell",
                "consensus_score": 0.9,
                "automated_consensus_accept": True,
            },
            {
                "scan": "a.h5",
                "consensus_candidate_id": 2,
                "consensus_class": "accepted_cell",
                "consensus_score": 0.7,
                "automated_consensus_accept": True,
            },
            {
                "scan": "a.h5",
                "consensus_candidate_id": 3,
                "consensus_class": "ambiguous_cell_candidate",
                "consensus_score": 0.5,
                "automated_consensus_accept": False,
            },
            {
                "scan": "a.h5",
                "consensus_candidate_id": 4,
                "consensus_class": "rejected_channel_specific_object",
                "consensus_score": 0.2,
                "automated_consensus_accept": False,
            },
        ]
    )


def seed(tmp_path):
    pd.DataFrame(consensus()).to_csv(
        tmp_path / "cell_consensus.csv",
        index=False,
    )
    review = ensure_cell_review(
        tmp_path,
        consensus(),
    )
    review.loc[
        review["consensus_candidate_id"].astype(int) == 2,
        "review_decision",
    ] = "accept"
    review.loc[
        review["consensus_candidate_id"].astype(int) == 3,
        "review_decision",
    ] = "ambiguous"
    review.loc[
        review["consensus_candidate_id"].astype(int) == 4,
        "review_decision",
    ] = "reject"
    review.to_csv(
        tmp_path / "cell_review.csv",
        index=False,
    )


def test_bulk_accept_changes_only_pending(tmp_path):
    seed(tmp_path)

    result = bulk_accept_pending_reviews(
        tmp_path,
        reviewer="HH",
        notes="bulk review",
    )
    review = result["review"].copy()

    assert result["updated_count"] == 1

    decisions = {
        int(row.consensus_candidate_id): row.review_decision
        for row in review.itertuples()
    }
    assert decisions == {
        1: "accept",
        2: "accept",
        3: "ambiguous",
        4: "reject",
    }


def test_bulk_accept_records_provenance(tmp_path):
    seed(tmp_path)
    result = bulk_accept_pending_reviews(
        tmp_path,
        reviewer="HH",
        notes="accept remaining pending",
    )
    row = result["review"][
        result["review"][
            "consensus_candidate_id"
        ].astype(int)
        == 1
    ].iloc[0]

    assert row["review_reason"] == "bulk_accept_pending"
    assert row["review_source"] == (
        "manual_ui_bulk_accept_pending"
    )
    assert row["reviewer"] == "HH"
    assert row["review_notes"] == (
        "accept remaining pending"
    )
    assert row["reviewed_at_utc"]
    assert not bool(row["canonical_cell"])


def test_bulk_accept_noop_when_no_pending(tmp_path):
    seed(tmp_path)
    bulk_accept_pending_reviews(tmp_path)
    second = bulk_accept_pending_reviews(tmp_path)
    assert second["updated_count"] == 0


def test_bulk_accept_is_persistent(tmp_path):
    seed(tmp_path)
    bulk_accept_pending_reviews(tmp_path)
    persisted = pd.read_csv(
        tmp_path / "cell_review.csv",
        keep_default_na=False,
    )
    row = persisted[
        persisted["consensus_candidate_id"].astype(int)
        == 1
    ].iloc[0]
    assert row["review_decision"] == "accept"
