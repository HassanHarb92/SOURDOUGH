import numpy as np

from yeast_xrf.analysis.cell_consensus import (
    CellConsensusConfig,
    build_multichannel_cell_consensus,
)


def disk(shape, cy, cx, r):
    yy, xx = np.indices(shape)
    return (yy-cy)**2 + (xx-cx)**2 <= r**2


def test_three_channel_cell_is_accepted():
    shape = (80, 80)
    region = disk(shape, 40, 40, 10)
    maps = {
        "Total_Fluorescence_Yield": np.zeros(shape),
        "P": np.zeros(shape),
        "S": np.zeros(shape),
        "K": np.zeros(shape),
    }
    maps["Total_Fluorescence_Yield"][region] = 12
    maps["P"][region] = 8
    maps["S"][region] = 7

    result = build_multichannel_cell_consensus(
        maps,
        config=CellConsensusConfig(
            support_z=1.0,
            min_area_pixels=15,
            watershed_min_distance_pixels=4,
        ),
    )
    accepted = [
        r for r in result.consensus_rows
        if r["automated_consensus_accept"]
    ]
    assert accepted
    assert max(r["support_channel_count"] for r in accepted) >= 3
    assert all(r["canonical_cell"] is False for r in accepted)
    assert result.summary["canonical_cells_finalized"] is False


def test_single_channel_object_is_rejected():
    shape = (70, 70)
    region = disk(shape, 30, 30, 7)
    maps = {
        "Total_Fluorescence_Yield": np.zeros(shape),
        "P": np.zeros(shape),
        "S": np.zeros(shape),
        "K": np.zeros(shape),
    }
    maps["P"][region] = 20

    result = build_multichannel_cell_consensus(
        maps,
        config=CellConsensusConfig(
            support_z=1.0,
            min_area_pixels=10,
        ),
    )
    assert result.consensus_rows
    assert all(
        not r["automated_consensus_accept"]
        for r in result.consensus_rows
    )
    assert any(
        r["consensus_class"]
        == "rejected_channel_specific_object"
        for r in result.consensus_rows
    )


def test_artifact_mask_is_context_not_deletion():
    shape = (80, 80)
    region = disk(shape, 40, 40, 9)
    maps = {}
    for name in (
        "Total_Fluorescence_Yield",
        "P",
        "S",
        "K",
    ):
        arr = np.zeros(shape)
        arr[region] = 10
        maps[name] = arr

    artifact = np.zeros(shape, dtype=bool)
    artifact[region] = True
    originals = {k: v.copy() for k, v in maps.items()}

    result = build_multichannel_cell_consensus(
        maps,
        artifact_mask=artifact,
        config=CellConsensusConfig(
            support_z=1.0,
            min_area_pixels=15,
        ),
    )
    assert result.consensus_rows
    assert max(
        r["artifact_overlap_fraction"]
        for r in result.consensus_rows
    ) > 0.5
    for key in maps:
        assert np.array_equal(maps[key], originals[key])


def test_localized_elements_not_required():
    cfg = CellConsensusConfig()
    assert "Zn" not in cfg.support_channels
    assert "Fe" not in cfg.support_channels
    assert "Ca" not in cfg.support_channels
