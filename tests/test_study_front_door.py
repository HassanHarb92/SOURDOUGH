from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_study_first_modes_are_present():
    text = APP.read_text()
    assert '["ANALYZE", "RESULTS", "EXPLORE"]' in text
    assert 'key="sourdough_primary_workspace"' in text
    assert "Analyze entire study" in text
    assert "Directory in. Complete cellular XRF study out." in text


def test_analyze_is_default():
    text = APP.read_text()
    anchor = '["ANALYZE", "RESULTS", "EXPLORE"]'
    pos = text.index(anchor)
    nearby = text[pos: pos + 300]
    assert "index=0" in nearby


def test_existing_explorer_is_preserved_after_front_door():
    text = APP.read_text()
    front = text.index("# SOURDOUGH study-first front door")
    controls = text.index(
        'with st.container(border=True):\n'
        '    st.markdown("#### Study controls")'
    )
    assert front < controls
    for term in (
        "Scientific workspace",
        "Map Explorer",
        "Spectrum Inspector",
        "QC & Normalize",
        "Intensity & Contrast",
        "Gradients & Edges",
        "Hessian & Curvature",
        "2.5D Cell Landscape",
        "Cell Analyzer",
        "Beam & Quantification",
    ):
        assert term in text


def test_results_surface_study_tables():
    text = APP.read_text()
    for term in (
        "cell_elements",
        "element_pairs",
        "qc_flags",
        "Study provenance",
        "Cell × Element",
        "QC & Failures",
    ):
        assert term in text


def test_ui_keeps_hdf5_boundary():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
