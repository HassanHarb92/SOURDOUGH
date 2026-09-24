from pathlib import Path

import h5py
import numpy as np
import pytest

from yeast_xrf.io.semantic_probe import SMALL_DATASETS, axis_summary, probe_scan, read_small_dataset, summarize_probes


def _make_tiny_maps_file(path: Path, *, theta: float) -> None:
    with h5py.File(path, "w") as h:
        maps = h.create_group("MAPS")
        maps.create_dataset("version", data=np.array([b"test"]))
        scan = maps.create_group("Scan")
        scan.create_dataset("name", data=np.array([b"tiny"]))
        scan.create_dataset("scan_type", data=np.array([b"fly"]))
        scan.create_dataset("scan_time_stamp", data=np.array([b"now"]))
        scan.create_dataset("theta", data=np.array([theta]))
        scan.create_dataset("requested_rows", data=np.array([3]))
        scan.create_dataset("requested_cols", data=np.array([4]))
        scan.create_dataset("x_axis", data=np.array([0.0, 0.5, 1.0, 1.5]))
        scan.create_dataset("y_axis", data=np.array([0.0, 1.0, 2.0]))
        pv = scan.create_group("Extra_PVs")
        pv.create_dataset("Names", data=np.array([b"sample_z", b"beam_current"]))
        pv.create_dataset("Description", data=np.array([b"depth stage", b"current"]))
        pv.create_dataset("Unit", data=np.array([b"um", b"mA"]))
        pv.create_dataset("Values", data=np.array([b"2.0", b"100"]))

        analyzed = maps.create_group("XRF_Analyzed")
        for method in ("Fitted", "NNLS", "ROI"):
            g = analyzed.create_group(method)
            g.create_dataset("Channel_Names", data=np.array([b"Fe", b"Zn"]))
            g.create_dataset("Channel_Units", data=np.array([b"counts/s", b"counts/s"]))
            g.create_dataset("Counts_Per_Sec", data=np.zeros((2, 3, 4), dtype=np.float32))

        scalers = maps.create_group("Scalers")
        scalers.create_dataset("Names", data=np.array([b"I0", b"dwell"]))
        scalers.create_dataset("Units", data=np.array([b"counts", b"s"]))
        scalers.create_dataset("Values", data=np.zeros((2, 3, 4), dtype=np.float32))
        maps.create_dataset("scaler_names", data=np.array([b"I0"]))
        maps.create_dataset("scaler_units", data=np.array([b"counts"]))
        maps.create_dataset("scalers", data=np.zeros((1, 3, 4), dtype=np.float32))

        spectra = maps.create_group("Spectra")
        spectra.create_dataset("Energy", data=np.linspace(0.0, 20.47, 2048))
        spectra.create_dataset("Energy_Calibration", data=np.array([0.0, 0.01, 0.0]))
        spectra.create_dataset("mca_arr", data=np.zeros((2048, 3, 4), dtype=np.float32))
        for name in ("Elapsed_Livetime", "Elapsed_Realtime", "Input_Counts", "Output_Counts"):
            spectra.create_dataset(name, data=np.zeros((3, 4)))

        quant = maps.create_group("Quantification")
        quant.create_dataset("Number_Of_Standards", data=np.array([1]))
        std = quant.create_group("Standard0")
        std.create_dataset("Standard_Name", data=np.array([b"std"]))
        std.create_dataset("Element_Weights", data=np.array([1.0, 2.0]))
        std.create_dataset("Element_Weights_Names", data=np.array([b"Fe", b"Zn"]))
        maps.create_dataset("XRF_fits_quant_names", data=np.array([b"fits"]))
        maps.create_dataset("XRF_roi_plus_quant_names", data=np.array([b"roi_plus"]))
        maps.create_dataset("XRF_roi_quant_names", data=np.array([b"roi"]))


def test_axis_summary_recovers_spacing() -> None:
    info = axis_summary([0.0, 0.5, 1.0, 1.5])
    assert info["count"] == 4
    assert info["median_step_abs"] == pytest.approx(0.5)
    assert info["approximately_uniform"] is True


def test_small_dataset_guard_rejects_non_allowlisted_path(tmp_path: Path) -> None:
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("huge", data=np.zeros((100, 100)))
    with h5py.File(path, "r") as h:
        with pytest.raises(ValueError, match="not allowlisted"):
            read_small_dataset(h, "/huge")
    assert "/MAPS/Scan/theta" in SMALL_DATASETS


def test_probe_recovers_channel_axis_geometry_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _make_tiny_maps_file(path, theta=15.0)
    probe = probe_scan(path)
    assert probe["scan"]["theta"] == pytest.approx(15.0)
    assert probe["geometry"]["fitted_raster_shape_yx"] == [3, 4]
    assert probe["geometry"]["x_axis_matches_raster"] is True
    assert probe["geometry"]["y_axis_matches_raster"] is True
    assert probe["channels"]["Fitted"]["names"] == ["Fe", "Zn"]
    assert probe["spectra"]["mca_shape"] == [2048, 3, 4]
    assert probe["z_related_extra_pvs"]


def test_theta_variation_is_only_a_candidate_not_proof(tmp_path: Path) -> None:
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _make_tiny_maps_file(a, theta=0.0)
    _make_tiny_maps_file(b, theta=15.0)
    summary = summarize_probes([probe_scan(a), probe_scan(b)])
    assert summary["theta"]["unique_count"] == 2
    assert "worth investigating" in summary["tomography_assessment"]
    assert "does not prove tomography" in summary["tomography_assessment"]


def test_semantic_probe_preserves_blank_channel_unit_slots(tmp_path: Path) -> None:
    path = tmp_path / "blank-units.h5"
    _make_tiny_maps_file(path, theta=0.0)
    with h5py.File(path, "r+") as h:
        del h["/MAPS/XRF_Analyzed/Fitted/Channel_Units"]
        h["/MAPS/XRF_Analyzed/Fitted"].create_dataset(
            "Channel_Units", data=np.array([b"cts/s", b""])
        )

    probe = probe_scan(path)
    assert probe["channels"]["Fitted"]["names"] == ["Fe", "Zn"]
    assert probe["channels"]["Fitted"]["units"] == ["cts/s", ""]
