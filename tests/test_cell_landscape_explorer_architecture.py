from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_landscape_tab_and_scientific_boundary():
    text = APP.read_text()
    for term in (
        "2.5D Cell Landscape",
        "2.5D chemical landscape",
        "not a reconstructed physical 3D yeast cell",
        "Derived/display height",
        "Curvature emphasis",
        "Bright-ridge overlay",
        "Bright-blob overlay",
        "Point landscape inspector",
    ):
        assert term in text


def test_background_mask_is_not_called_segmentation():
    text = APP.read_text()
    assert "visualization mask, not cell segmentation" in text


def test_landscape_plotly_charts_have_explicit_unique_keys():
    text = APP.read_text()
    keys = (
        'key="landscape_height_reference_plot"',
        'key="landscape_color_reference_plot"',
        'key="cell_landscape_3d_plot"',
    )
    for key in keys:
        assert key in text
        assert text.count(key) == 1


def test_no_direct_hdf5():
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text
