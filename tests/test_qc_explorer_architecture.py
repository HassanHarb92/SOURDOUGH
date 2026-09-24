from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_qc_normalization_tab_exists() -> None:
    text = APP.read_text()
    assert "QC & Normalize" in text
    assert "Explicit reference normalization" in text


def test_normalization_is_not_hidden_default() -> None:
    text = APP.read_text()
    assert "Normalization is opt-in" in text
    assert "Denominator floor policy" in text
    assert "Explicit scale factor" in text


def test_negative_values_are_not_silently_removed() -> None:
    text = APP.read_text()
    assert "Mask negative XRF numerator" in text
    assert "value=False" in text
    assert "Negative fitted values are flagged, not discarded" in text


def test_detector_diagnostics_are_labeled_as_diagnostics() -> None:
    text = APP.read_text()
    assert "Implied dead-time fraction = 1 - OCR/ICR" in text
    assert "Live-time diagnostic = ELT/ERT" in text
    assert "not automatically applied" in text


def test_ui_still_has_no_direct_hdf5_paths() -> None:
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text
