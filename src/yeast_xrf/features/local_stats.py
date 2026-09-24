'''Local statistical feature maps for 2D XRF images.'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from yeast_xrf.features.feature_metadata import FeatureMetadata, make_feature_metadata


@dataclass(frozen=True)
class FeatureResult:
    data: np.ndarray
    metadata: FeatureMetadata


def _window(value: int) -> int:
    value = int(value)
    if value < 3 or value % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    return value


def _filled(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)
    fill = float(np.median(arr[finite])) if finite.any() else 0.0
    return np.where(finite, arr, fill), finite


def _restore(data: np.ndarray, finite: np.ndarray) -> np.ndarray:
    out = np.asarray(data, dtype=float).copy()
    out[~finite] = np.nan
    return out


def _meta(name: str, description: str, input_label: str, output_unit: str, **params):
    return make_feature_metadata(
        name=name,
        kind="derived_feature",
        description=description,
        input_label=input_label,
        output_unit=output_unit,
        parameters=params,
        provenance={"raw_data_modified": False},
    )


def local_mean(data, *, window, input_label, input_unit):
    window = _window(window)
    work, finite = _filled(data)
    out = ndi.uniform_filter(work, size=window, mode="nearest")
    return FeatureResult(
        _restore(out, finite),
        _meta("local_mean", "Local arithmetic mean.", input_label, input_unit,
              window_pixels=window),
    )


def local_std(data, *, window, input_label, input_unit):
    window = _window(window)
    work, finite = _filled(data)
    mean = ndi.uniform_filter(work, size=window, mode="nearest")
    mean2 = ndi.uniform_filter(work * work, size=window, mode="nearest")
    out = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    return FeatureResult(
        _restore(out, finite),
        _meta("local_std", "Local standard deviation.", input_label, input_unit,
              window_pixels=window),
    )


def local_cv(data, *, window, input_label, input_unit, mean_floor=0.0):
    window = _window(window)
    work, finite = _filled(data)
    mean = ndi.uniform_filter(work, size=window, mode="nearest")
    mean2 = ndi.uniform_filter(work * work, size=window, mode="nearest")
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    valid = finite & (np.abs(mean) > float(mean_floor))
    out = np.full(work.shape, np.nan)
    np.divide(std, np.abs(mean), out=out, where=valid)
    return FeatureResult(
        out,
        _meta("local_cv", "Local coefficient of variation.", input_label,
              "dimensionless", window_pixels=window,
              mean_floor=float(mean_floor), input_unit=input_unit),
    )


def local_median(data, *, window, input_label, input_unit):
    window = _window(window)
    work, finite = _filled(data)
    out = ndi.median_filter(work, size=window, mode="nearest")
    return FeatureResult(
        _restore(out, finite),
        _meta("local_median", "Local median.", input_label, input_unit,
              window_pixels=window),
    )


def local_mad(data, *, window, input_label, input_unit):
    window = _window(window)
    work, finite = _filled(data)
    med = ndi.median_filter(work, size=window, mode="nearest")
    out = ndi.median_filter(np.abs(work - med), size=window, mode="nearest")
    return FeatureResult(
        _restore(out, finite),
        _meta("local_mad", "Local median absolute deviation.", input_label,
              input_unit, window_pixels=window),
    )


def local_iqr(data, *, window, input_label, input_unit):
    window = _window(window)
    work, finite = _filled(data)
    q25 = ndi.percentile_filter(work, 25, size=window, mode="nearest")
    q75 = ndi.percentile_filter(work, 75, size=window, mode="nearest")
    return FeatureResult(
        _restore(q75 - q25, finite),
        _meta("local_iqr", "Local interquartile range.", input_label,
              input_unit, window_pixels=window),
    )


def local_contrast_z(
    data,
    *,
    window,
    input_label,
    input_unit,
    std_floor=0.0,
):
    window = _window(window)
    work, finite = _filled(data)
    mean = ndi.uniform_filter(work, size=window, mode="nearest")
    mean2 = ndi.uniform_filter(work * work, size=window, mode="nearest")
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    valid = finite & (std > float(std_floor))
    out = np.full(work.shape, np.nan)
    np.divide(work - mean, std, out=out, where=valid)
    return FeatureResult(
        out,
        _meta("local_contrast_z", "Local standardized contrast.", input_label,
              "dimensionless", window_pixels=window,
              std_floor=float(std_floor), input_unit=input_unit),
    )
