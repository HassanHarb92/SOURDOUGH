from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_u11_visual_terms_exist():
    text = APP.read_text()
    for term in (
        "SOURDOUGH · Beam & Quantification",
        "Counts/s → beam normalization → validated MAPS concentration",
        "1 · Signal & beam normalization",
        "2 · Calibration",
        "3 · Concentration",
        "4 · Cell results",
        "Check calibration readiness",
        "Automatic MAPS concentration",
        "sd-hero-card",
        "sd-success",
        "sd-pill",
    ):
        assert term in text


def test_u11_preserves_legacy_quantification_contract_terms():
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


def test_u11_preserves_legacy_plotly_keys():
    text = APP.read_text()
    for key in (
        'key="u1_quant_raw_plot"',
        'key="u1_quant_us_plot"',
        'key="u1_quant_ds_plot"',
        'key="u1_quant_beam_ratio_plot"',
        'key="u1_quant_areal_density_plot"',
    ):
        assert key in text
