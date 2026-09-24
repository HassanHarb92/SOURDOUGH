from pathlib import Path

import h5py
import numpy as np
import pytest

from yeast_xrf.io.channel_roles import ChannelRole, channel_role
from yeast_xrf.io.xrf_scan import XRFScan, normalize_method


def _tiny_scan(path: Path) -> None:
    y_size, x_size = 3, 4
    channels = [b"Fe", b"Zn", b"COMPTON_AMPLITUDE", b"Num_Iter", b"Fit_Residual"]
    units = [b"cts/s", b"cts/s", b"cts/s", b"", b""]

    with h5py.File(path, "w") as h:
        maps = h.create_group("MAPS")

        scan = maps.create_group("Scan")
        scan.create_dataset("name", data=np.array([b"tiny"]))
        scan.create_dataset("scan_type", data=np.array([b"fly"]))
        scan.create_dataset("scan_time_stamp", data=np.array([b"now"]))
        scan.create_dataset("theta", data=np.array([0.0]))
        scan.create_dataset("requested_rows", data=np.array([y_size]))
        scan.create_dataset("requested_cols", data=np.array([x_size]))
        scan.create_dataset("x_axis", data=np.array([0.0, 0.08, 0.16, 0.24]))
        scan.create_dataset("y_axis", data=np.array([0.0, 0.08, 0.16]))

        pv = scan.create_group("Extra_PVs")
        pv.create_dataset("Names", data=np.array([b"sample_z"]))
        pv.create_dataset("Description", data=np.array([b"stage"]))
        pv.create_dataset("Unit", data=np.array([b"um"]))
        pv.create_dataset("Values", data=np.array([b"2.0"]))

        analyzed = maps.create_group("XRF_Analyzed")
        base = np.arange(len(channels) * y_size * x_size).reshape(len(channels), y_size, x_size)
        for offset, method in enumerate(("Fitted", "NNLS", "ROI")):
            g = analyzed.create_group(method)
            g.create_dataset("Channel_Names", data=np.array(channels))
            g.create_dataset("Channel_Units", data=np.array(units))
            g.create_dataset("Counts_Per_Sec", data=(base + offset * 100).astype(np.float32))

        spectra = maps.create_group("Spectra")
        spectra.create_dataset("Energy", data=np.array([1.0, 2.0, 3.0, 4.0]))
        spectra.create_dataset("Energy_Calibration", data=np.array([0.0, 1.0, 0.0]))
        mca = np.arange(4 * y_size * x_size).reshape(4, y_size, x_size)
        spectra.create_dataset("mca_arr", data=mca.astype(np.float32))
        integ = spectra.create_group("Integrated_Spectra")
        integ.create_dataset("Spectra", data=mca.sum(axis=(1, 2)))

        primary = maps.create_group("Scalers")
        primary.create_dataset("Names", data=np.array([b"US_IC", b"Dead_Time"]))
        primary.create_dataset("Units", data=np.array([b"cts", b"%"]))
        primary.create_dataset(
            "Values",
            data=np.stack([np.ones((y_size, x_size)), np.full((y_size, x_size), 5.0)]),
        )

        maps.create_dataset("scaler_names", data=np.array([b"US_IC"]))
        maps.create_dataset("scaler_units", data=np.array([b"cts"]))
        maps.create_dataset("scalers", data=np.ones((1, y_size, x_size), dtype=np.float32))


def test_channel_roles() -> None:
    assert channel_role("Fe") == ChannelRole.ELEMENTAL_OR_LINE
    assert channel_role("COMPTON_AMPLITUDE") == ChannelRole.SCATTER
    assert channel_role("Fit_Residual") == ChannelRole.FIT_DIAGNOSTIC
    assert channel_role("future") == ChannelRole.AUXILIARY


def test_method_normalization() -> None:
    assert normalize_method("fit") == "Fitted"
    assert normalize_method("NNLS") == "NNLS"
    with pytest.raises(ValueError):
        normalize_method("magic")


def test_blank_units_stay_index_aligned(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    with XRFScan.open(path) as scan:
        channels = scan.channels()
        assert len(channels) == 5
        assert channels[3].name == "Num_Iter"
        assert channels[3].unit == ""
        assert channels[4].name == "Fit_Residual"
        assert channels[4].unit == ""


def test_named_map_and_method(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    with XRFScan.open(path) as scan:
        fe = scan.map("Fe", "Fitted")
        zn = scan.map("Zn", "NNLS")
        assert fe.shape == (3, 4)
        assert np.allclose(zn, np.arange(12, 24).reshape(3, 4) + 100)


def test_xarray_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    with XRFScan.open(path) as scan:
        data = scan.map_xarray("Fe")
        assert data.dims == ("y", "x")
        assert np.allclose(data.coords["x"], [0.0, 0.08, 0.16, 0.24])
        assert np.allclose(data.coords["y"], [0.0, 0.08, 0.16])
        assert data.attrs["unit"] == "cts/s"


def test_spectrum_is_one_energy_vector(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    with XRFScan.open(path) as scan:
        spec = scan.spectrum_xarray(y=1, x=2)
        assert spec.shape == (4,)
        assert np.allclose(spec.coords["energy"], [1.0, 2.0, 3.0, 4.0])


def test_nearest_coordinate(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    with XRFScan.open(path) as scan:
        assert scan.nearest_pixel(x=0.17, y=0.07) == (1, 2)


def test_scaler_by_name(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    with XRFScan.open(path) as scan:
        dead = scan.scaler_xarray("Dead_Time")
        assert np.allclose(dead, 5.0)
        assert dead.attrs["unit"] == "%"


def test_validation_passes(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    with XRFScan.open(path) as scan:
        assert scan.validate() == ()


def test_context_manager_closes(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _tiny_scan(path)
    scan = XRFScan.open(path)
    with scan:
        assert scan.is_open
    assert not scan.is_open
