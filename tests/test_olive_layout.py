from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
SHELL = ROOT / "src" / "yeast_xrf" / "ui" / "olive_shell.py"


def test_olive_masthead_is_present():
    app = APP.read_text()
    shell = SHELL.read_text()
    assert "SHELL_HTML" in app
    assert 'class="olive-head"' in shell
    assert "XRF CHEMICAL CELL ANALYSIS PLATFORM" in shell
    assert "MEASURED · X/Y + XRF" in shell
    assert "DERIVED · gradients + Hessian + features" in shell
    assert "INFERRED · model-based Z geometry" in shell


def test_global_sidebar_was_replaced_by_main_workspace_controls():
    text = APP.read_text()
    assert "with st.sidebar:" not in text
    assert 'st.markdown("#### Study controls")' in text
    assert 'key="global_scan_selector"' in text
    assert 'key="global_analysis_method"' in text


def test_olive_dashboard_summary_is_present():
    text = APP.read_text()
    assert 'st.markdown("#### Dataset at a glance")' in text
    assert "Analyzed channels" in text
    assert "Energy channels" in text
    assert "raw data remain read-only" in text


def test_scientific_workspace_and_existing_capabilities_remain():
    text = APP.read_text()
    assert 'st.markdown("#### Scientific workspace")' in text
    for term in (
        "Map Explorer",
        "Spectrum Inspector",
        "QC & Normalize",
        "Intensity & Contrast",
        "Gradients & Edges",
        "Hessian & Curvature",
        "2.5D Cell Landscape",
        "Cell Analyzer & 3D Model",
    ):
        assert term in text


def test_olive_shell_does_not_change_hdf5_boundary():
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text
