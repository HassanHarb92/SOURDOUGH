from pathlib import Path

import pandas as pd
import pytest

from yeast_xrf.analysis.study_analysis import (
    _merge_metadata,
    _stats,
    discover_scans,
)


def test_stats_preserve_negative_information():
    stats = _stats([-2.0, 0.0, 2.0, 4.0])
    assert stats["finite_count"] == 4
    assert stats["negative_count"] == 1
    assert stats["negative_fraction_finite"] == pytest.approx(0.25)
    assert stats["sum_pixel_values"] == pytest.approx(4.0)


def test_discover_scans_recursive(tmp_path):
    (tmp_path / "a.h5").write_bytes(b"x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.hdf5").write_bytes(b"x")
    (nested / "ignore.txt").write_text("x")

    names = [
        p.name
        for p in discover_scans(tmp_path, recursive=True)
    ]
    assert names == ["a.h5", "b.hdf5"]


def test_metadata_merge_by_scan(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "sample_id": "a",
                "condition": "",
                "include": True,
            }
        ]
    )
    metadata = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "condition": "control",
                "biological_replicate": "1",
            }
        ]
    ).to_csv(metadata, index=False)

    merged = _merge_metadata(manifest, metadata)
    assert merged.loc[0, "condition"] == "control"
    assert merged.loc[0, "biological_replicate"] == "1"
    assert bool(merged.loc[0, "include"]) is True


def test_metadata_preserves_identifier_strings_and_parses_include(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "sample_id": "a",
                "condition": "",
                "include": True,
            }
        ]
    )
    metadata = tmp_path / "meta.csv"
    metadata.write_text(
        "scan,sample_id,biological_replicate,technical_replicate,include\n"
        "a.h5,S01,01,002,no\n"
    )

    merged = _merge_metadata(manifest, metadata)
    assert merged.loc[0, "sample_id"] == "S01"
    assert merged.loc[0, "biological_replicate"] == "01"
    assert merged.loc[0, "technical_replicate"] == "002"
    assert bool(merged.loc[0, "include"]) is False


def test_study_engine_does_not_claim_canonical_cells():
    source = Path(
        __import__(
            "yeast_xrf.analysis.study_analysis",
            fromlist=["dummy"],
        ).__file__
    ).read_text()
    assert '"canonical_cell": False' in source
    assert '"segmentation_status": (' in source
    assert "provisional_tfy_v1" in source
