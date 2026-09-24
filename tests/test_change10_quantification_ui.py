from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_change10_automatic_concentration_is_visible():
    text = APP.read_text()
    for term in (
        "MAPS concentration ready",
        "Automatic MAPS concentration",
        "MAPS calibration verified",
        "maps_concentration(",
        "quantifiable_channels(",
        "µg/cm²",
        "Full-cell quantitative summary",
    ):
        assert term in text


def test_change10_preserves_quantification_ui_contract():
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

    for key in (
        'key="u1_quant_raw_plot"',
        'key="u1_quant_us_plot"',
        'key="u1_quant_ds_plot"',
        'key="u1_quant_beam_ratio_plot"',
        'key="u1_quant_areal_density_plot"',
    ):
        assert key in text


def test_ui_respects_hdf5_boundary():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
