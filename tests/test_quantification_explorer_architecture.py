from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_beam_quantification_tab_exists():
    text = APP.read_text()
    for term in (
        "Beam & Quantification",
        "US_IC-normalized",
        "DS_IC-normalized",
        "Beam ratio diagnostic",
        "Check calibration readiness",
        "Manual concentration factor",
        "Full-cell quantitative summary",
    ):
        assert term in text


def test_quantification_io_boundary_is_preserved():
    text = APP.read_text()
    assert "from yeast_xrf.io.maps_quantification import" in text
    assert "/MAPS/" not in text
    assert "import h5py" not in text


def test_plotly_keys_are_explicit():
    text = APP.read_text()
    for key in (
        'key="u1_quant_raw_plot"',
        'key="u1_quant_us_plot"',
        'key="u1_quant_ds_plot"',
        'key="u1_quant_beam_ratio_plot"',
        'key="u1_quant_areal_density_plot"',
    ):
        assert key in text
