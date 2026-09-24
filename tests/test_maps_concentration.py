import h5py
import numpy as np
import pytest

from yeast_xrf.analysis.maps_concentration import maps_concentration


def build_file(path):
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "/MAPS/XRF_Analyzed/Fitted/Channel_Names",
            data=np.array([b"Zn"]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/"
            "Calibration_Curve_Labels",
            data=np.array([[b"Zn"], [b"Zn_L"], [b"Zn_M"]]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/"
            "Calibration_Curve_US_IC",
            data=np.array([[0.2], [0.02], [0.002]]),
        )
        h5.create_dataset(
            "/MAPS/XRF_fits_quant",
            data=np.array([[[20.0]], [[0.2]], [[0.1]]]),
        )


def test_maps_concentration_formula_and_negative_preservation(tmp_path):
    path = tmp_path / "sample.h5"
    build_file(path)

    raw = np.array([[40.0, -20.0, 20.0]])
    scaler = np.array([[100.0, 100.0, 0.0]])

    product = maps_concentration(
        path,
        raw,
        scaler,
        method="Fitted",
        reference="US_IC",
        channel="Zn",
        floor_percentile=0.0,
    )

    # Strict denominator rule masks scaler <= floor. Positive floor is 100,
    # so both 100-valued pixels are masked at percentile 0. Use a second
    # test below for finite values.
    assert np.isnan(product.data[0, 0])
    assert np.isnan(product.data[0, 1])
    assert np.isnan(product.data[0, 2])


def test_maps_concentration_numeric_conversion(tmp_path):
    path = tmp_path / "sample.h5"
    build_file(path)

    raw = np.array([[40.0, -20.0, 20.0]])
    scaler = np.array([[200.0, 200.0, 100.0]])

    product = maps_concentration(
        path,
        raw,
        scaler,
        method="Fitted",
        reference="US_IC",
        channel="Zn",
        floor_percentile=0.0,
    )

    # floor=100, scaler=200 survives:
    # (40/200)/0.2 = 1 ug/cm2
    # (-20/200)/0.2 = -0.5 ug/cm2
    assert product.data[0, 0] == pytest.approx(1.0)
    assert product.data[0, 1] == pytest.approx(-0.5)
    assert np.isnan(product.data[0, 2])
    assert product.unit == "µg/cm²"
    assert product.provenance["negative_values_preserved"] is True
    assert product.provenance["raw_data_modified"] is False
