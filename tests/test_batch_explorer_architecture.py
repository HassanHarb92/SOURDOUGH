from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_batch_study_tab_is_present():
    text = APP.read_text()
    for term in (
        "Batch Study",
        "Analyze an entire directory",
        "Full cells analyzed",
        "Cropped cells excluded",
        "sample snapshot",
    ):
        assert term in text


def test_batch_runner_is_imported():
    text = APP.read_text()
    assert "from yeast_xrf.analysis.batch import analyze_directory" in text


def test_batch_tab_has_no_direct_hdf5_access():
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text
