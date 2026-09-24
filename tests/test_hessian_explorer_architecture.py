from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_hessian_tab_and_core_features():
    text = APP.read_text()
    for term in (
        "Hessian & Curvature",
        "Ixx",
        "Iyy",
        "Ixy",
        "Mixed derivative disagreement",
        "Laplacian",
        "Hessian determinant",
        "Lambda min",
        "Lambda max",
        "Principal direction",
        "Curvedness",
        "Shape index",
        "Bright ridge",
        "Dark valley",
        "Bright blob",
        "Dark blob",
        "Point Hessian inspector",
    ):
        assert term in text


def test_hessian_ui_explains_mixed_derivative_qc():
    text = APP.read_text()
    assert "Ixy and Iyx" in text
    assert "numerical QC" in text


def test_no_direct_hdf5():
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text



def test_hessian_plotly_charts_have_unique_keys():
    text = APP.read_text()
    assert 'key="hessian_source_plot"' in text
    assert 'key="hessian_feature_plot"' in text
    assert text.count('key="hessian_source_plot"') == 1
    assert text.count('key="hessian_feature_plot"') == 1
