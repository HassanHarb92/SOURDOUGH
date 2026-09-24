import h5py
import numpy as np
import pytest

from yeast_xrf.io.maps_concentration import (
    quantifiable_channels,
    read_maps_calibration_factor,
)


def build_file(path):
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "/MAPS/XRF_Analyzed/Fitted/Channel_Names",
            data=np.array([b"P", b"Zn", b"Total_Fluorescence_Yield"]),
        )
        labels = np.array(
            [
                [b"P", b"Zn", b"X"],
                [b"P_L", b"Zn_L", b"X_L"],
                [b"P_M", b"Zn_M", b"X_M"],
            ]
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/"
            "Calibration_Curve_Labels",
            data=labels,
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/"
            "Calibration_Curve_US_IC",
            data=np.array(
                [
                    [0.1, 0.2, 0.0],
                    [0.01, 0.02, 0.0],
                    [0.001, 0.002, 0.0],
                ]
            ),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/"
            "Calibration_Curve_DS_IC",
            data=np.array(
                [
                    [0.05, 0.1, 0.0],
                    [0.005, 0.01, 0.0],
                    [0.0005, 0.001, 0.0],
                ]
            ),
        )
        quant = np.zeros((3, 1, 3), dtype=float)
        quant[0, 0, :] = [10.0, 20.0, 0.0]
        quant[1, 0, :] = [0.1, 0.2, 0.0]
        quant[2, 0, :] = [0.05, 0.1, 0.0]
        h5.create_dataset("/MAPS/XRF_fits_quant", data=quant)


def test_factor_matches_curve(tmp_path):
    path = tmp_path / "sample.h5"
    build_file(path)
    factor = read_maps_calibration_factor(
        path,
        method="Fitted",
        reference="US_IC",
        channel="Zn",
    )
    assert factor.factor == pytest.approx(0.2)
    assert factor.curve_value == pytest.approx(0.2)
    assert factor.quant_row == 1
    assert factor.relative_error == pytest.approx(0.0)
    assert factor.read_only is True


def test_quantifiable_channels_excludes_non_element_map(tmp_path):
    path = tmp_path / "sample.h5"
    build_file(path)
    channels = quantifiable_channels(
        path,
        method="Fitted",
        reference="US_IC",
    )
    assert channels == ["P", "Zn"]


def test_reader_rejects_factor_curve_disagreement(tmp_path):
    path = tmp_path / "sample.h5"
    build_file(path)
    with h5py.File(path, "r+") as h5:
        h5["/MAPS/XRF_fits_quant"][1, 0, 1] = 999.0
    with pytest.raises(ValueError, match="disagrees"):
        read_maps_calibration_factor(
            path,
            method="Fitted",
            reference="US_IC",
            channel="Zn",
        )
