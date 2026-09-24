from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_u1_guided_sections_exist():
    text = APP.read_text()
    for term in (
        "Step 1 · Inspect and normalize the signal",
        "Step 2 · Check calibration readiness",
        "Step 3 · Concentration conversion",
        "Step 4 · Full-cell quantitative summary",
        "Advanced calibration metadata",
        "Advanced signal diagnostics",
        "Manual concentration factor",
        "Quantification provenance",
    ):
        assert term in text


def test_u1_removes_old_blank_placeholder():
    text = APP.read_text()
    assert "Areal-density conversion not applied" not in text


def test_u1_preserves_scientific_io_boundary():
    text = APP.read_text()
    assert "from yeast_xrf.io.maps_quantification import" in text
    assert "/MAPS/" not in text
    assert "import h5py" not in text


def test_u1_uses_fixed_scaler_argument_order():
    text = APP.read_text()
    assert (
        'load_scaler(path_text, family="primary", name="US_IC")'
        in text
    )
    assert (
        'load_scaler(path_text, family="primary", name="DS_IC")'
        in text
    )
