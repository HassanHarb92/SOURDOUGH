import numpy as np
import pytest

from yeast_xrf.visualization.statistics import finite_shared_range, map_statistics
from yeast_xrf.visualization.transforms import (
    display_transform,
    gamma_display,
    percentile_stretch,
    signed_log1p,
)


def test_raw_display_is_copy_not_alias() -> None:
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = display_transform(source, "Raw")
    assert np.allclose(result.data, source)
    result.data[0, 0] = 999.0
    assert source[0, 0] == 1.0


def test_percentile_stretch_is_bounded() -> None:
    source = np.arange(100, dtype=float).reshape(10, 10)
    result = percentile_stretch(source, low=5, high=95)
    assert np.nanmin(result) >= 0.0
    assert np.nanmax(result) <= 1.0


def test_percentile_stretch_validates_percentiles() -> None:
    with pytest.raises(ValueError):
        percentile_stretch(np.arange(5), low=99, high=1)


def test_signed_log_preserves_sign() -> None:
    source = np.array([-9.0, -1.0, 0.0, 1.0, 9.0])
    result = signed_log1p(source)
    assert result[0] < 0
    assert result[1] < 0
    assert result[2] == 0
    assert result[3] > 0
    assert result[4] > 0


def test_gamma_display_is_bounded() -> None:
    source = np.arange(25, dtype=float).reshape(5, 5)
    result = gamma_display(source, gamma=0.7)
    assert np.nanmin(result) >= 0.0
    assert np.nanmax(result) <= 1.0


def test_map_statistics_tracks_negative_and_nonfinite_values() -> None:
    source = np.array([[-2.0, 0.0], [2.0, np.nan]])
    stats = map_statistics(source)
    assert stats["pixel_count"] == 4
    assert stats["finite_count"] == 3
    assert stats["nan_or_inf_count"] == 1
    assert stats["negative_count"] == 1
    assert stats["zero_count"] == 1


def test_shared_range_ignores_nonfinite() -> None:
    shared = finite_shared_range(
        [
            np.array([[1.0, 2.0], [np.nan, 3.0]]),
            np.array([[-4.0, 8.0]]),
        ]
    )
    assert shared == (-4.0, 8.0)
