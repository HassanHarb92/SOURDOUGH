from pathlib import Path
import json

import pandas as pd

from yeast_xrf.reporting.report_contract import (
    REPORT_SECTIONS,
    refresh_report_contract,
)


def seed_overview(run):
    pd.DataFrame(
        [{"scan": "a.h5", "include": True}]
    ).to_csv(run / "study_manifest.csv", index=False)
    pd.DataFrame(
        [{"scan": "a.h5", "status": "analyzed"}]
    ).to_csv(run / "scans.csv", index=False)
    (run / "study_summary.json").write_text("{}")
    (run / "study_config.json").write_text("{}")
    (run / "study_provenance.json").write_text("{}")


def test_report_workspace_created(tmp_path):
    seed_overview(tmp_path)
    manifest = refresh_report_contract(tmp_path)

    report = tmp_path / "report"
    assert (report / "report_manifest.json").is_file()
    assert (report / "report_status.csv").is_file()
    assert (report / "REPORT_SKELETON.md").is_file()
    for name in (
        "figures",
        "tables",
        "cells",
        "samples",
        "conditions",
        "appendix",
    ):
        assert (report / name).is_dir()

    assert len(manifest["sections"]) == 12
    assert manifest["final_report_ready"] is False


def test_cell_identification_stays_partial_until_canonicalization(tmp_path):
    seed_overview(tmp_path)
    pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "consensus_candidate_id": 1,
            }
        ]
    ).to_csv(tmp_path / "cell_consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "consensus_candidate_id": 1,
                "channel": "P",
            }
        ]
    ).to_csv(
        tmp_path / "cell_channel_support.csv",
        index=False,
    )

    manifest = refresh_report_contract(tmp_path)
    section = next(
        s
        for s in manifest["sections"]
        if s["id"] == "cell_identification"
    )
    assert section["status"] == "partial"

    pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "consensus_candidate_id": 1,
                "review_decision": "accept",
            }
        ]
    ).to_csv(tmp_path / "cell_review.csv", index=False)

    manifest = refresh_report_contract(tmp_path)
    section = next(
        s
        for s in manifest["sections"]
        if s["id"] == "cell_identification"
    )
    assert section["status"] == "partial"
    assert "not finalized" in section["status_note"].lower()


def test_report_contract_finalizes_no_downstream_science_by_itself(tmp_path):
    seed_overview(tmp_path)
    manifest = refresh_report_contract(tmp_path)
    downstream = {
        "cellular_composition",
        "element_localization",
        "chemical_domains",
        "cell_phenotypes",
        "condition_comparison",
        "organelle_likeness",
        "biological_findings",
    }
    statuses = {
        s["id"]: s["status"]
        for s in manifest["sections"]
    }
    assert all(
        statuses[name] == "planned"
        for name in downstream
    )


def test_policy_prevents_overclaiming(tmp_path):
    seed_overview(tmp_path)
    manifest = refresh_report_contract(tmp_path)
    policy = manifest["report_policy"]

    assert policy["artifact_grounded_claims_only"]
    assert policy[
        "canonical_cells_required_for_final_cell_analysis"
    ]
    assert policy["unverified_physical_units_forbidden"]
    assert policy[
        "2d_xrf_volume_claims_require_model_label"
    ]
    assert policy[
        "organelle_output_is_likeness_not_identity"
    ]


def test_current_provisional_pairs_are_only_partial(tmp_path):
    seed_overview(tmp_path)
    pd.DataFrame(
        [{"element_a": "P", "element_b": "Zn"}]
    ).to_csv(tmp_path / "element_pairs.csv", index=False)

    manifest = refresh_report_contract(tmp_path)
    section = next(
        s
        for s in manifest["sections"]
        if s["id"] == "element_relationships"
    )
    assert section["status"] == "partial"
    assert "recomputed on canonical cells" in (
        section["status_note"]
    )
