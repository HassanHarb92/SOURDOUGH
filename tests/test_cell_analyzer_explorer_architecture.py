from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_cell_analyzer_tab_exists():
    text = APP.read_text()
    for term in (
        "Cell Analyzer & 3D Model",
        "Total_Fluorescence_Yield",
        "Complete cells",
        "Cropped cells",
        "Exclude cropped cells",
        "Inferred 3D cell model",
        "Projection-conserving interior",
        "Cutaway axis",
        "Internal Z slice",
    ):
        assert term in text


def test_3d_model_boundary_is_explicit():
    text = APP.read_text()
    assert "measured X/Y footprint" in text
    assert "inferred depth" in text
    assert "not an experimental 3D reconstruction" in text


def test_cell_model_plot_has_explicit_key():
    text = APP.read_text()
    assert 'key="inferred_cell_model_plot"' in text
    assert text.count('key="inferred_cell_model_plot"') == 1


def test_no_direct_hdf5():
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text
