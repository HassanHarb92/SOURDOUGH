from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_ui_does_not_access_hdf5_storage_paths_directly() -> None:
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "h5py" not in text
    assert "XRFScan" in text


def test_ui_exposes_core_explorer_tabs() -> None:
    text = APP.read_text()
    for label in (
        "Map Explorer",
        "Spectrum Inspector",
        "Scaler & QC",
        "Compare",
        "Acquisition",
    ):
        assert label in text


def test_2p5d_is_explicitly_not_called_a_volume() -> None:
    text = APP.read_text()
    assert "2.5D intensity surface" in text
    assert "not a reconstructed 3D yeast cell" in text



def test_map_explorer_exposes_original_raw_reference() -> None:
    text = APP.read_text()
    assert "Show original raw map beside transformed view" in text
    assert "Original / raw" in text
    assert "Original raw map" in text


def test_scan_overview_uses_total_fluorescence_yield_when_available() -> None:
    text = APP.read_text()
    assert "Scan overview / original appearance" in text
    assert "Total_Fluorescence_Yield" in text
    assert "not an optical microscopy photograph" in text
