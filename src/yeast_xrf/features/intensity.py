'''Global intensity-derived feature maps.'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yeast_xrf.features.feature_metadata import FeatureMetadata, make_feature_metadata


@dataclass(frozen=True)
class FeatureResult:
    data: np.ndarray
    metadata: FeatureMetadata


def asinh_feature(
    data: np.ndarray,
    *,
    input_label: str,
    input_unit: str,
    scale: float | None = None,
) -> FeatureResult:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if scale is None:
        robust = float(np.median(np.abs(finite))) if finite.size else 1.0
        scale = robust if robust > 0 else 1.0
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("asinh scale must be finite and > 0")
    out = np.arcsinh(arr / scale)
    return FeatureResult(
        out,
        make_feature_metadata(
            name="asinh_intensity",
            kind="derived_feature",
            description="Dimensionless asinh intensity transform.",
            input_label=input_label,
            output_unit="dimensionless",
            parameters={"scale": scale, "input_unit": input_unit},
            provenance={"formula": "asinh(input / scale)", "raw_data_modified": False},
        ),
    )


def robust_zscore_feature(
    data: np.ndarray,
    *,
    input_label: str,
    input_unit: str,
) -> FeatureResult:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        center = float(np.median(finite))
        mad = float(np.median(np.abs(finite - center)))
        denom = 1.4826 * mad
        if not np.isfinite(denom) or denom <= 0:
            denom = 1.0
    else:
        center, denom = 0.0, 1.0
    out = (arr - center) / denom
    return FeatureResult(
        out,
        make_feature_metadata(
            name="robust_zscore",
            kind="derived_feature",
            description="Median/MAD robust standardized intensity.",
            input_label=input_label,
            output_unit="dimensionless",
            parameters={"median": center, "scaled_mad": denom, "input_unit": input_unit},
            provenance={
                "formula": "(input - median) / (1.4826 * MAD)",
                "raw_data_modified": False,
            },
        ),
    )
