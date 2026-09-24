from pathlib import Path

import numpy as np

from yeast_xrf.analysis.batch import (
    _cell_bounds,
    _energy_axis,
    _stats,
    _write_html_report,
)


class FakeScan:
    energy = np.linspace(0, 10, 6)


def test_stats_are_comprehensive():
    result = _stats(np.array([-2.0, 0.0, 1.0, 3.0, np.nan]))
    assert result["finite_count"] == 4
    assert result["negative_fraction_finite"] == 0.25
    assert result["zero_fraction_finite"] == 0.25
    for key in (
        "min", "p01", "p05", "p25", "median", "mean", "std",
        "mad", "p75", "p95", "p99", "max", "sum",
    ):
        assert key in result


def test_cell_bounds_are_clipped_and_padded():
    labels = np.zeros((10, 12), dtype=int)
    labels[0:3, 0:4] = 1
    assert _cell_bounds(labels, 1, pad=4) == (0, 7, 0, 8)


def test_energy_axis_uses_scan_axis_when_available():
    assert np.allclose(_energy_axis(FakeScan(), 6), FakeScan.energy)


def test_html_report_contains_full_and_cropped_counts(tmp_path):
    row = {
        "scan": "sample.h5",
        "detected_cells": 5,
        "full_cells_analyzed": 3,
        "cropped_cells": 2,
        "sample_snapshot": "scans/sample/sample_snapshot.png",
        "scan_directory": "scans/sample",
    }
    _write_html_report(tmp_path, [row])
    html = (tmp_path / "index.html").read_text()
    assert "Full analyzed <b>3</b>" in html
    assert "Cropped excluded <b>2</b>" in html
    assert "sample_snapshot.png" in html
