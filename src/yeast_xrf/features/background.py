'''Background estimation and subtraction for 2D XRF maps.'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from yeast_xrf.features.feature_metadata import FeatureMetadata, make_feature_metadata


@dataclass(frozen=True)
class FeatureResult:
    data: np.ndarray
    metadata: FeatureMetadata


def _filled(data):
    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)
    fill = float(np.median(arr[finite])) if finite.any() else 0.0
    return arr, np.where(finite, arr, fill), finite


def gaussian_background(data, *, sigma_pixels, input_label, input_unit):
    sigma = float(sigma_pixels)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_pixels must be finite and > 0")
    _, work, finite = _filled(data)
    out = ndi.gaussian_filter(work, sigma=sigma, mode="nearest")
    out[~finite] = np.nan
    return FeatureResult(
        out,
        make_feature_metadata(
            name="gaussian_background",
            kind="derived_feature",
            description="Smooth Gaussian background estimate.",
            input_label=input_label,
            output_unit=input_unit,
            parameters={"sigma_pixels": sigma},
            provenance={"operation": "gaussian_filter", "raw_data_modified": False},
        ),
    )


def background_subtracted(data, *, sigma_pixels, input_label, input_unit):
    arr = np.asarray(data, dtype=float)
    bg = gaussian_background(
        arr,
        sigma_pixels=sigma_pixels,
        input_label=input_label,
        input_unit=input_unit,
    )
    out = arr - bg.data
    return FeatureResult(
        out,
        make_feature_metadata(
            name="background_subtracted",
            kind="derived_feature",
            description="Input minus Gaussian background estimate.",
            input_label=input_label,
            output_unit=input_unit,
            parameters={"sigma_pixels": float(sigma_pixels)},
            provenance={
                "formula": "input - gaussian_background",
                "raw_data_modified": False,
            },
        ),
    )
