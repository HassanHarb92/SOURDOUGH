from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
ENGINE = (
    ROOT
    / "src"
    / "yeast_xrf"
    / "analysis"
    / "study_analysis.py"
)


def test_artifact_screening_is_default_and_non_destructive():
    text = APP.read_text()
    assert (
        "Screen artifact / contamination candidates (retain + flag)"
        in text
    )
    assert "study_analysis_artifact_screening" in text
    assert "artifact_screening=bool(" in text
    assert (
        "Suspicion scores are heuristic review priorities"
        in text
    )


def test_results_show_artifact_review_table():
    text = APP.read_text()
    for term in (
        "artifact_candidates",
        "Artifact / contamination review",
        "results_artifact_priority",
        "results_artifact_element",
        "retain + flag",
    ):
        assert term in text


def test_study_engine_writes_artifact_layer():
    text = ENGINE.read_text()
    for term in (
        "screen_artifact_candidates",
        "artifact_candidate_masks.npz",
        "artifact_screening.json",
        '"artifact_layer_ready": bool(artifact_screening)',
        '"artifact_policy": "retain_and_flag"',
    ):
        assert term in text


def test_ui_still_respects_hdf5_boundary():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
