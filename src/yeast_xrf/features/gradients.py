"""First-derivative and edge features for 2D XRF maps.

Coordinate-aware derivatives use the stored coordinate values. Strictly monotonic axes use
coordinate finite differences. Axes containing repeated positions use a local least-squares
slope against the real coordinate values rather than inventing a uniform grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage as ndi

from yeast_xrf.features.feature_metadata import FeatureMetadata, make_feature_metadata

GradientOperator = Literal["sobel", "scharr", "prewitt"]


@dataclass(frozen=True)
class GradientField:
    ix: np.ndarray
    iy: np.ndarray
    magnitude: np.ndarray
    orientation_deg: np.ndarray
    metadata: dict[str, FeatureMetadata]


@dataclass(frozen=True)
class FeatureResult:
    data: np.ndarray
    metadata: FeatureMetadata


def _prepare(data):
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D data, got {arr.shape}")
    finite = np.isfinite(arr)
    fill = float(np.median(arr[finite])) if finite.any() else 0.0
    return np.where(finite, arr, fill), finite


def _axis(axis, expected, name):
    arr = np.asarray(axis, dtype=float).reshape(-1)
    if arr.size != expected:
        raise ValueError(f"{name} length {arr.size} != {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has nonfinite values")
    if np.unique(arr).size < 2:
        raise ValueError(f"{name} has fewer than two distinct coordinates")
    return arr


def _diag(axis):
    d = np.diff(axis)
    abs_d = np.abs(d)
    positive = abs_d[abs_d > 0]
    mean = float(np.mean(positive)) if positive.size else 0.0
    cv = float(np.std(positive) / mean) if mean else 0.0

    strict_inc = bool(np.all(d > 0))
    strict_dec = bool(np.all(d < 0))
    nondec = bool(np.all(d >= 0))
    noninc = bool(np.all(d <= 0))

    if strict_inc:
        direction = "strictly_increasing"
    elif strict_dec:
        direction = "strictly_decreasing"
    elif nondec:
        direction = "nondecreasing_with_repeats"
    elif noninc:
        direction = "nonincreasing_with_repeats"
    else:
        direction = "mixed_order"

    return {
        "median_nonzero_step": float(np.median(positive)) if positive.size else 0.0,
        "nonzero_step_cv": cv,
        "repeated_adjacent_count": int(np.count_nonzero(d == 0)),
        "unique_coordinate_count": int(np.unique(axis).size),
        "direction": direction,
        "strict_monotonic": bool(strict_inc or strict_dec),
    }


def _local_coordinate_slope(work, coords, *, axis, radius=2):
    """Estimate dI/dcoord using local least-squares fits.

    Duplicate coordinates are allowed. For each row/column location we fit a local line
    against the actual coordinate values. The window expands only if the initial local
    neighborhood does not contain enough coordinate variation.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.size
    out = np.full(work.shape, np.nan, dtype=float)

    for i in range(n):
        r = int(radius)
        denom = 0.0
        while True:
            lo = max(0, i - r)
            hi = min(n, i + r + 1)
            c = coords[lo:hi]
            centered = c - np.mean(c)
            denom = float(np.dot(centered, centered))
            if denom > 0 or (lo == 0 and hi == n):
                break
            r += 1

        if denom <= 0:
            raise ValueError("local coordinate window has no coordinate variation")

        weights = centered / denom
        if axis == 0:
            local = work[lo:hi, :]
            out[i, :] = np.tensordot(weights, local, axes=(0, 0))
        elif axis == 1:
            local = work[:, lo:hi]
            out[:, i] = np.tensordot(local, weights, axes=(1, 0))
        else:
            raise ValueError("axis must be 0 or 1")

    return out


def _coordinate_derivative(work, coords, *, axis):
    diag = _diag(coords)
    if diag["strict_monotonic"]:
        edge_order = 2 if work.shape[axis] >= 3 else 1
        derivative = np.gradient(work, coords, axis=axis, edge_order=edge_order)
        method = "coordinate_finite_difference"
    else:
        derivative = _local_coordinate_slope(work, coords, axis=axis, radius=2)
        method = "local_coordinate_least_squares"
    return np.asarray(derivative, dtype=float), method, diag


def coordinate_gradient(
    data,
    x,
    y,
    *,
    input_label,
    input_unit,
    sigma_pixels=1.0,
):
    work, finite = _prepare(data)
    x = _axis(x, work.shape[1], "x")
    y = _axis(y, work.shape[0], "y")

    sigma = float(sigma_pixels)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma_pixels must be finite and >= 0")
    if sigma > 0:
        work = ndi.gaussian_filter(work, sigma=sigma, mode="nearest")

    ix, x_method, x_diag = _coordinate_derivative(work, x, axis=1)
    iy, y_method, y_diag = _coordinate_derivative(work, y, axis=0)

    magnitude = np.hypot(ix, iy)
    orientation = np.degrees(np.arctan2(iy, ix))

    for arr in (ix, iy, magnitude, orientation):
        arr[~finite] = np.nan

    unit = f"{input_unit or 'value'} per stored-coordinate-unit"
    prov = {
        "sigma_pixels": sigma,
        "spacing_mode": "actual_coordinate_values",
        "x_derivative_method": x_method,
        "y_derivative_method": y_method,
        "x_axis": x_diag,
        "y_axis": y_diag,
        "coordinate_unit": None,
        "raw_data_modified": False,
    }

    meta = {
        "ix": make_feature_metadata(
            name="gradient_ix",
            kind="derived_feature",
            description="dI/dx using stored X coordinate values.",
            input_label=input_label,
            output_unit=unit,
            parameters={"sigma_pixels": sigma, "axis": "x"},
            provenance={**prov, "formula": "dI/dx"},
        ),
        "iy": make_feature_metadata(
            name="gradient_iy",
            kind="derived_feature",
            description="dI/dy using stored Y coordinate values.",
            input_label=input_label,
            output_unit=unit,
            parameters={"sigma_pixels": sigma, "axis": "y"},
            provenance={**prov, "formula": "dI/dy"},
        ),
        "magnitude": make_feature_metadata(
            name="gradient_magnitude",
            kind="derived_feature",
            description="sqrt(Ix^2 + Iy^2).",
            input_label=input_label,
            output_unit=unit,
            parameters={"sigma_pixels": sigma},
            provenance={**prov, "formula": "sqrt(Ix^2 + Iy^2)"},
        ),
        "orientation_deg": make_feature_metadata(
            name="gradient_orientation",
            kind="derived_feature",
            description="atan2(Iy, Ix) in degrees.",
            input_label=input_label,
            output_unit="degrees",
            parameters={"range": "[-180,180]", "zero": "+X"},
            provenance={**prov, "formula": "degrees(atan2(Iy,Ix))"},
        ),
    }
    return GradientField(ix, iy, magnitude, orientation, meta)


def directional_derivative(field, *, angle_deg, input_label):
    theta = np.deg2rad(float(angle_deg))
    out = field.ix * np.cos(theta) + field.iy * np.sin(theta)
    return FeatureResult(
        out,
        make_feature_metadata(
            name="directional_derivative",
            kind="derived_feature",
            description="Gradient projection onto selected direction.",
            input_label=input_label,
            output_unit=field.metadata["ix"].output_unit,
            parameters={"angle_deg": float(angle_deg)},
            provenance={
                "formula": "Ix*cos(theta)+Iy*sin(theta)",
                "convention": "0 deg=+X, 90 deg=+Y",
                "raw_data_modified": False,
            },
        ),
    )


def _kernels(operator):
    if operator == "sobel":
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float) / 8.0
    elif operator == "scharr":
        kx = np.array([[-3,0,3],[-10,0,10],[-3,0,3]], float) / 32.0
    elif operator == "prewitt":
        kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], float) / 6.0
    else:
        raise ValueError("operator must be sobel, scharr, or prewitt")
    return kx, kx.T


def pixel_operator_gradient(
    data, *, operator: GradientOperator, input_label, input_unit
):
    work, finite = _prepare(data)
    kx, ky = _kernels(operator)
    ix = ndi.correlate(work, kx, mode="nearest")
    iy = ndi.correlate(work, ky, mode="nearest")
    magnitude = np.hypot(ix, iy)
    orientation = np.degrees(np.arctan2(iy, ix))
    for arr in (ix, iy, magnitude, orientation):
        arr[~finite] = np.nan

    unit = f"{input_unit or 'value'} per pixel"
    common = {
        "operator": operator,
        "spacing_mode": "pixel_grid",
        "raw_data_modified": False,
    }
    meta = {
        "ix": make_feature_metadata(
            name=f"{operator}_gx",
            kind="derived_feature",
            description=f"{operator} X edge response.",
            input_label=input_label,
            output_unit=unit,
            parameters={"operator": operator, "axis": "x"},
            provenance=common,
        ),
        "iy": make_feature_metadata(
            name=f"{operator}_gy",
            kind="derived_feature",
            description=f"{operator} Y edge response.",
            input_label=input_label,
            output_unit=unit,
            parameters={"operator": operator, "axis": "y"},
            provenance=common,
        ),
        "magnitude": make_feature_metadata(
            name=f"{operator}_magnitude",
            kind="derived_feature",
            description=f"{operator} edge magnitude.",
            input_label=input_label,
            output_unit=unit,
            parameters={"operator": operator},
            provenance={**common, "formula": "sqrt(Gx^2+Gy^2)"},
        ),
        "orientation_deg": make_feature_metadata(
            name=f"{operator}_orientation",
            kind="derived_feature",
            description=f"{operator} response orientation.",
            input_label=input_label,
            output_unit="degrees",
            parameters={"operator": operator},
            provenance={**common, "formula": "degrees(atan2(Gy,Gx))"},
        ),
    }
    return GradientField(ix, iy, magnitude, orientation, meta)


def edge_mask(strength, *, percentile, input_label):
    p = float(percentile)
    if not 0 < p < 100:
        raise ValueError("percentile must be between 0 and 100")
    arr = np.asarray(strength, dtype=float)
    finite = arr[np.isfinite(arr)]
    threshold = float(np.percentile(finite, p)) if finite.size else np.nan
    mask = (
        np.isfinite(arr) & (arr >= threshold)
        if finite.size
        else np.zeros(arr.shape, bool)
    )
    return FeatureResult(
        mask,
        make_feature_metadata(
            name="edge_mask",
            kind="derived_feature",
            description="Binary percentile-threshold edge mask.",
            input_label=input_label,
            output_unit="boolean",
            parameters={"percentile": p, "threshold": threshold},
            provenance={
                "formula": "strength>=percentile_threshold",
                "raw_data_modified": False,
            },
        ),
    )

