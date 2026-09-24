from pathlib import Path
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

def test_gradient_tab_and_features():
    text = APP.read_text()
    for term in (
        "Gradients & Edges", "Coordinate-aware gradient", "Pixel-grid edge operator",
        "Gradient magnitude", "Gradient orientation", "Directional derivative",
        "Edge mask", "Sobel", "Scharr", "Prewitt",
        "exact stored X/Y coordinate vectors", "stored-coordinate-unit",
    ):
        assert term in text

def test_no_direct_hdf5():
    text = APP.read_text()
    assert "/MAPS/" not in text
    assert "import h5py" not in text
