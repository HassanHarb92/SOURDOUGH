from yeast_xrf.io.schema_compare import analyze_3d_readiness, compare_inventories


def _inventory(name: str, datasets: list[dict]) -> dict:
    return {
        "_scan_name": name,
        "source": name,
        "root_attrs": {},
        "groups": [],
        "datasets": datasets,
    }


def test_schema_compare_tracks_common_and_variable_paths() -> None:
    a = _inventory(
        "a.h5",
        [
            {"path": "/maps/Fe", "shape": [10, 20], "dtype": "float32", "ndim": 2, "attrs": {}},
            {"path": "/x", "shape": [20], "dtype": "float64", "ndim": 1, "attrs": {"axis": "x_pos"}},
        ],
    )
    b = _inventory(
        "b.h5",
        [{"path": "/maps/Fe", "shape": [12, 20], "dtype": "float32", "ndim": 2, "attrs": {}}],
    )
    comparison = compare_inventories([a, b])
    assert "/maps/Fe" in comparison["common_dataset_paths"]
    assert "/x" in comparison["variable_dataset_paths"]
    assert len(comparison["datasets"]["/maps/Fe"]["shape_signatures"]) == 2


def test_rank_three_does_not_automatically_claim_native_volume() -> None:
    spectral = _inventory(
        "spectral.h5",
        [
            {
                "path": "/mca/spectra",
                "shape": [64, 64, 2048],
                "dtype": "float32",
                "ndim": 3,
                "attrs": {"last_axis": "energy_channel"},
            }
        ],
    )
    comparison = compare_inventories([spectral])
    readiness = analyze_3d_readiness([spectral], comparison)
    assert readiness["routes"]["native_volume"]["status"] == "inspect"
    assert readiness["candidate_rank3plus_datasets"]


def test_explicit_z_hint_makes_native_volume_plausible_not_proven() -> None:
    volume = _inventory(
        "volume.h5",
        [
            {
                "path": "/reconstruction/volume",
                "shape": [8, 64, 64],
                "dtype": "float32",
                "ndim": 3,
                "attrs": {"z_axis": "depth_um"},
            }
        ],
    )
    comparison = compare_inventories([volume])
    readiness = analyze_3d_readiness([volume], comparison)
    assert readiness["routes"]["native_volume"]["status"] == "plausible"
    assert readiness["coordinate_hints"]["z"]


def test_2p5d_is_always_labeled_as_visualization() -> None:
    flat = _inventory(
        "flat.h5",
        [{"path": "/Fe", "shape": [32, 32], "dtype": "float32", "ndim": 2, "attrs": {}}],
    )
    comparison = compare_inventories([flat])
    readiness = analyze_3d_readiness([flat], comparison)
    info = readiness["routes"]["pseudo_3d_2p5d"]
    assert info["status"] == "available"
    assert "must not be described as a reconstructed cell volume" in info["reason"]
