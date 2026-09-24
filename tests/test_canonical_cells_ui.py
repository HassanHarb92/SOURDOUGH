from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_canonical_cells_tab_exists():
    text = APP.read_text()
    assert '"Canonical Cells"' in text
    assert "with _sd_tabs[8]:" in text
    assert "Canonical Cell Dataset" in text


def test_canonicalization_is_review_gated():
    text = APP.read_text()
    for term in (
        "Finalize canonical cells",
        "pending",
        "ambiguous",
        "Canonicalization is blocked",
        "_sd_preview_canonicalization",
    ):
        assert term in text


def test_canonical_outputs_visible():
    text = APP.read_text()
    for term in (
        "canonical_cells.csv",
        "canonical_cell_lineage.csv",
        "canonicalization_summary.json",
        "canonical_cell_masks.npz",
        "Canonical lineage",
    ):
        assert term in text


def test_staleness_is_visible():
    text = APP.read_text()
    assert "canonical dataset is stale" in text
    assert "Re-finalize" in text


def test_ui_hdf5_boundary_preserved():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
