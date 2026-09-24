from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
SHELL = ROOT / "src" / "yeast_xrf" / "ui" / "olive_shell.py"


def test_sourdough_primary_branding():
    shell = SHELL.read_text()
    assert '<div class="olive-title">SOURDOUGH</div>' in shell
    assert (
        '<div class="olive-kicker">XRF CHEMICAL CELL ANALYSIS PLATFORM</div>'
        in shell
    )


def test_sourdough_acronym_expansion():
    shell = SHELL.read_text()
    assert (
        "Synchrotron Observations Using Resolved Distributions Of Uptake,"
        in shell
    )
    assert "Geometry &amp; Heterogeneity" in shell


def test_product_line():
    shell = SHELL.read_text()
    assert "XRF chemical imaging and cellular analysis." in shell


def test_scientific_state_badges_survive_rebrand():
    shell = SHELL.read_text()
    assert "MEASURED · X/Y + XRF" in shell
    assert "DERIVED · gradients + Hessian + features" in shell
    assert "INFERRED · model-based Z geometry" in shell


def test_old_product_title_is_not_the_primary_masthead():
    shell = SHELL.read_text()
    assert '<div class="olive-title">Yeast XRF Explorer</div>' not in shell


def test_browser_page_title_if_configured():
    app = APP.read_text()
    if "page_title=" in app:
        assert 'page_title="SOURDOUGH"' in app
