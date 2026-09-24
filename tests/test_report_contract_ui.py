from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_report_contract_tab_exists():
    text = APP.read_text()
    assert '"Report Contract"' in text
    assert "with _sd_tabs[7]:" in text
    assert "Final Report Contract" in text


def test_report_contract_exposes_scientific_boundaries():
    text = APP.read_text()
    for term in (
        "Every claim must trace",
        "Canonical cells are required",
        "Unverified physical units",
        "Organelle output is organelle-likeness",
        "Heuristic scores are not called probabilities",
    ):
        assert term in text


def test_report_workspace_is_visible():
    text = APP.read_text()
    for term in (
        "report_manifest.json",
        "report_status.csv",
        "REPORT_SKELETON.md",
        "figures/",
        "cells/",
        "conditions/",
    ):
        assert term in text


def test_ui_hdf5_boundary_remains_intact():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
