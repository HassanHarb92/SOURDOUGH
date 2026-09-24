from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_accept_all_pending_button_exists():
    text = APP.read_text()
    for term in (
        "Bulk review actions",
        "Accept all pending",
        "_sd_bulk_accept_pending_reviews",
        "results_bulk_accept_confirm",
        "currently `pending` as `accept`",
    ):
        assert term in text


def test_bulk_accept_ui_preserves_non_pending_decisions_message():
    text = APP.read_text()
    assert (
        "Existing accepted, rejected, and ambiguous decisions"
        in text
    )
    assert "are not changed" in text


def test_bulk_accept_ui_requires_confirmation():
    text = APP.read_text()
    assert "or not _sd_bulk_confirm" in text


def test_ui_hdf5_boundary_preserved():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
