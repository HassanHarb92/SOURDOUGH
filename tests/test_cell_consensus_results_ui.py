from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_results_has_cell_consensus_tab():
    text = APP.read_text()
    assert '"Cell Consensus"' in text
    assert "with _sd_tabs[6]:" in text
    assert "Multichannel Cell Consensus" in text


def test_results_loads_consensus_outputs():
    text = APP.read_text()
    for term in (
        '_sd_read_table("cell_consensus")',
        '"cell_channel_support"',
        '"cell_consensus_scans"',
        "cell_masks_multichannel_consensus.npz",
    ):
        assert term in text


def test_consensus_view_is_review_pending_not_canonical_claim():
    text = APP.read_text()
    assert "not final canonical truth" in text
    assert "Automated acceptance is not the same" in text
    assert "Review" in text
    assert "Pending" in text


def test_consensus_visuals_are_present():
    text = APP.read_text()
    for term in (
        "Support count",
        "All candidates",
        "Automated accepted",
        "Cell-support masks",
        "Candidate coverage by",
        "artifact_overlap_fraction",
    ):
        assert term in text


def test_old_runs_are_handled_gracefully():
    text = APP.read_text()
    assert "predates Change 13A" in text


def test_results_ui_does_not_access_hdf5_directly():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
