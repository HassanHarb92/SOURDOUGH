import numpy as np

from yeast_xrf.analysis.artifacts import (
    ArtifactScreenConfig,
    screen_artifact_candidates,
)


def _disk(shape, cy, cx, r):
    yy, xx = np.indices(shape)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def test_isolated_channel_spike_is_flagged_without_mutation():
    shape = (48, 48)
    cell = _disk(shape, 24, 24, 9)

    maps = {}
    for name, scale in (
        ("P", 5.0),
        ("S", 4.0),
        ("K", 6.0),
        ("Zn", 1.0),
    ):
        arr = np.zeros(shape, dtype=float)
        arr[cell] = scale
        maps[name] = arr

    maps["Zn"][5, 6] = 100.0

    tfy = np.zeros(shape, dtype=float)
    tfy[cell] = 10.0
    labels = np.zeros(shape, dtype=int)
    labels[cell] = 1

    originals = {k: v.copy() for k, v in maps.items()}

    result = screen_artifact_candidates(
        maps,
        tfy=tfy,
        cell_labels=labels,
        config=ArtifactScreenConfig(
            candidate_z=5.0,
            local_spike_z=6.0,
            high_priority_score=0.60,
        ),
    )

    zn_rows = [
        row
        for row in result.candidates
        if row["channel"] == "Zn"
        and row["centroid_y_pixel"] < 10
    ]
    assert zn_rows
    row = max(zn_rows, key=lambda x: x["suspicion_score"])
    assert row["action"] == "retain_and_flag"
    assert row["score_is_probability"] is False
    assert row["cross_channel_support_count"] == 0
    assert row["tfy_support_fraction"] < 0.2
    assert row["provisional_cell_overlap_fraction"] < 0.2
    assert row["review_priority"] == "high"

    for key in maps:
        assert np.array_equal(maps[key], originals[key])


def test_cross_channel_cell_feature_is_retained():
    shape = (50, 50)
    cell = _disk(shape, 25, 25, 8)
    maps = {}
    for name in ("P", "S", "K"):
        arr = np.zeros(shape, dtype=float)
        arr[cell] = 8.0
        maps[name] = arr

    tfy = np.zeros(shape, dtype=float)
    tfy[cell] = 12.0
    labels = np.zeros(shape, dtype=int)
    labels[cell] = 1

    result = screen_artifact_candidates(
        maps,
        tfy=tfy,
        cell_labels=labels,
        config=ArtifactScreenConfig(candidate_z=2.0),
    )

    assert result.summary["raw_or_quantitative_maps_modified"] is False
    assert all(
        row["action"] == "retain_and_flag"
        for row in result.candidates
    )


def test_shape_mismatch_is_rejected():
    maps = {
        "P": np.zeros((10, 10)),
        "Zn": np.zeros((11, 10)),
    }
    try:
        screen_artifact_candidates(maps)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch ValueError")
