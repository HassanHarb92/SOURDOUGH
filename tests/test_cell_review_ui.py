from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_cell_review_controls_exist():
    text = APP.read_text()
    for term in (
        "Human cell review",
        "Save cell review",
        "cell_review.csv",
        "_sd_save_cell_review",
        "_sd_ensure_cell_review",
        "Review decision",
        "Review notes",
    ):
        assert term in text


def test_review_is_not_called_canonicalization():
    text = APP.read_text()
    assert "does not yet create" in text
    assert "Change 15 will consume accepted" in text


def test_review_appears_in_files_provenance():
    text = APP.read_text()
    assert '"cell_review.csv"' in text


def test_ui_hdf5_boundary_is_preserved():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
