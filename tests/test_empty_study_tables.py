from pathlib import Path

import pandas as pd

from yeast_xrf.analysis.study_analysis import _write_table


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_zero_by_zero_table_is_not_written_as_invalid_csv(tmp_path):
    result = _write_table(
        pd.DataFrame(),
        tmp_path / "failures",
    )
    assert result["empty"] is True
    assert result["row_count"] == 0
    assert result["column_count"] == 0
    assert result["csv"] is None
    assert not (tmp_path / "failures.csv").exists()


def test_empty_table_with_schema_is_readable_csv(tmp_path):
    df = pd.DataFrame(columns=["scan", "error"])
    result = _write_table(
        df,
        tmp_path / "failures",
    )
    path = tmp_path / "failures.csv"

    assert result["empty"] is True
    assert path.exists()
    loaded = pd.read_csv(path)
    assert loaded.empty
    assert list(loaded.columns) == ["scan", "error"]


def test_results_reader_has_empty_file_guard():
    text = APP.read_text()
    assert "if _path.stat().st_size == 0:" in text
    assert "except _sd_pd.errors.EmptyDataError:" in text
