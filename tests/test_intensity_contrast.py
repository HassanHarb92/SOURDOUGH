import numpy as np
import pytest

from yeast_xrf.features.background import background_subtracted, gaussian_background
from yeast_xrf.features.intensity import asinh_feature, robust_zscore_feature
from yeast_xrf.features.local_stats import (
    local_contrast_z,
    local_cv,
    local_iqr,
    local_mad,
    local_mean,
    local_median,
    local_std,
)
from yeast_xrf.visualization.transforms import clahe_display, display_transform


def test_local_mean_and_std_constant():
    data = np.full((9, 9), 5.0)
    assert np.allclose(local_mean(data, window=3, input_label="Fe", input_unit="cts/s").data, 5.0)
    assert np.allclose(local_std(data, window=5, input_label="Fe", input_unit="cts/s").data, 0.0)


def test_local_features_units_and_shapes():
    data = np.arange(81, dtype=float).reshape(9, 9) + 1
    for fn in (local_median, local_mad, local_iqr):
        result = fn(data, window=3, input_label="P", input_unit="cts/s")
        assert result.data.shape == data.shape
        assert result.metadata.output_unit == "cts/s"
    assert local_cv(data, window=3, input_label="Zn", input_unit="cts/s").metadata.output_unit == "dimensionless"
    assert local_contrast_z(data, window=3, input_label="Zn", input_unit="cts/s").metadata.output_unit == "dimensionless"


def test_window_must_be_odd():
    with pytest.raises(ValueError):
        local_mean(np.ones((7, 7)), window=4, input_label="Fe", input_unit="cts/s")


def test_asinh_preserves_sign_and_is_dimensionless():
    result = asinh_feature(
        np.array([[-10.0, 0.0, 10.0]]),
        input_label="Fe",
        input_unit="cts/s",
        scale=2.0,
    )
    assert result.data[0, 0] < 0 < result.data[0, 2]
    assert result.metadata.output_unit == "dimensionless"


def test_robust_zscore_does_not_modify_input():
    data = np.arange(25, dtype=float).reshape(5, 5)
    original = data.copy()
    result = robust_zscore_feature(data, input_label="Zn", input_unit="cts/s")
    assert np.array_equal(data, original)
    assert result.metadata.output_unit == "dimensionless"


def test_background_subtraction_can_be_negative():
    data = np.zeros((21, 21))
    data[10, 10] = 10.0
    result = background_subtracted(
        data,
        sigma_pixels=2.0,
        input_label="Zn",
        input_unit="cts/s",
    )
    assert result.data[10, 10] > 0
    assert np.nanmin(result.data) < 0
    assert result.metadata.output_unit == "cts/s"


def test_bad_sigma_rejected():
    with pytest.raises(ValueError):
        gaussian_background(
            np.ones((5, 5)),
            sigma_pixels=0,
            input_label="Fe",
            input_unit="cts/s",
        )


def test_clahe_is_display_only_and_bounded():
    data = np.arange(100, dtype=float).reshape(10, 10)
    result = clahe_display(data)
    assert np.nanmin(result) >= 0
    assert np.nanmax(result) <= 1
    wrapped = display_transform(data, "CLAHE")
    assert wrapped.parameters["mode"] == "clahe"
