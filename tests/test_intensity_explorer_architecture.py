from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_intensity_tab_exists():
    text = APP.read_text()
    assert "Intensity & Contrast" in text
    assert "Derived scientific feature" in text
    assert "Display-only" in text


def test_core_features_are_exposed():
    text = APP.read_text()
    for term in (
        "Local mean",
        "Local std",
        "Local CV",
        "Local median",
        "Local MAD",
        "Local IQR",
        "Local contrast z",
        "Background estimate",
        "Background-subtracted",
        "Asinh intensity",
        "Robust z-score",
    ):
        assert term in text


def test_ui_has_no_direct_hdf5_paths():
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text
