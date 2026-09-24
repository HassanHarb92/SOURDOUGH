'''Coordinate-aware Hessian, curvature descriptors, ridges, valleys, and blobs.

Second derivatives are evaluated against the actual stored coordinate values.

For pure second derivatives Ixx and Iyy, each row/column uses a local quadratic least-squares
fit in coordinate space. This handles nonuniform spacing and repeated coordinates without
inventing a uniform grid.

The mixed derivative is symmetrized from d(Ix)/dy and d(Iy)/dx. Their disagreement is retained
as a QC feature rather than hidden.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from yeast_xrf.features.feature_metadata import FeatureMetadata, make_feature_metadata
from yeast_xrf.features.gradients import coordinate_gradient


@dataclass(frozen=True)
class HessianField:
    ixx: np.ndarray
    iyy: np.ndarray
    ixy: np.ndarray
    mixed_disagreement: np.ndarray
    trace: np.ndarray
    determinant: np.ndarray
    lambda_min: np.ndarray
    lambda_max: np.ndarray
    laplacian: np.ndarray
    principal_orientation_deg: np.ndarray
    curvedness: np.ndarray
    shape_index: np.ndarray
    bright_ridge: np.ndarray
    dark_valley: np.ndarray
    bright_blob: np.ndarray
    dark_blob: np.ndarray
    metadata: dict[str, FeatureMetadata]


def _prepare(data):
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D data, got {arr.shape}")
    finite = np.isfinite(arr)
    fill = float(np.median(arr[finite])) if finite.any() else 0.0
    return arr, np.where(finite, arr, fill), finite


def _axis(axis, expected, name):
    arr = np.asarray(axis, dtype=float).reshape(-1)
    if arr.size != expected:
        raise ValueError(f"{name} length {arr.size} != {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has nonfinite coordinates")
    if np.unique(arr).size < 3:
        raise ValueError(
            f"{name} requires at least three distinct coordinates for second derivatives"
        )
    return arr


def _axis_summary(axis):
    d = np.diff(axis)
    abs_d = np.abs(d)
    nonzero = abs_d[abs_d > 0]
    return {
        "unique_coordinate_count": int(np.unique(axis).size),
        "repeated_adjacent_count": int(np.count_nonzero(d == 0)),
        "median_nonzero_step": float(np.median(nonzero)) if nonzero.size else 0.0,
        "strict_monotonic": bool(np.all(d > 0) or np.all(d < 0)),
    }


def _local_second_derivative(work, coords, *, axis, radius):
    coords = np.asarray(coords, dtype=float)
    n = coords.size
    if np.unique(coords).size < 3:
        raise ValueError("second derivative requires at least three distinct coordinates")

    radius = int(radius)
    if radius < 1:
        raise ValueError("quadratic_radius must be >= 1")

    out = np.full(work.shape, np.nan, dtype=float)
    conds = []
    expanded = 0

    for i in range(n):
        r = radius
        while True:
            lo = max(0, i - r)
            hi = min(n, i + r + 1)
            c = coords[lo:hi] - coords[i]
            design = np.column_stack([np.ones_like(c), c, c * c])
            rank = int(np.linalg.matrix_rank(design))
            if rank >= 3:
                break
            if lo == 0 and hi == n:
                raise ValueError(
                    "could not obtain three independent coordinate positions "
                    "for a local quadratic fit"
                )
            r += 1
            expanded += 1

        conds.append(float(np.linalg.cond(design)))

        if axis == 0:
            coeff = np.linalg.lstsq(design, work[lo:hi, :], rcond=None)[0]
            out[i, :] = 2.0 * coeff[2, :]
        elif axis == 1:
            coeff = np.linalg.lstsq(design, work[:, lo:hi].T, rcond=None)[0]
            out[:, i] = 2.0 * coeff[2, :]
        else:
            raise ValueError("axis must be 0 or 1")

    diagnostics = {
        "method": "local_quadratic_least_squares",
        "requested_radius": radius,
        "window_expansion_count": expanded,
        "median_design_condition": float(np.median(conds)) if conds else None,
        "max_design_condition": float(np.max(conds)) if conds else None,
    }
    return out, diagnostics


def _restore_nonfinite(arrays, finite):
    for arr in arrays:
        arr[~finite] = np.nan


def coordinate_hessian(
    data,
    x,
    y,
    *,
    input_label,
    input_unit,
    sigma_pixels=1.0,
    quadratic_radius=2,
):
    raw, work, finite = _prepare(data)
    x = _axis(x, work.shape[1], "x")
    y = _axis(y, work.shape[0], "y")

    sigma = float(sigma_pixels)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma_pixels must be finite and >= 0")
    if sigma > 0:
        work = ndi.gaussian_filter(work, sigma=sigma, mode="nearest")

    first = coordinate_gradient(
        work,
        x,
        y,
        input_label=input_label,
        input_unit=input_unit,
        sigma_pixels=0.0,
    )

    ixx, x_second_diag = _local_second_derivative(
        work, x, axis=1, radius=quadratic_radius
    )
    iyy, y_second_diag = _local_second_derivative(
        work, y, axis=0, radius=quadratic_radius
    )

    d_ix = coordinate_gradient(
        first.ix,
        x,
        y,
        input_label=f"{input_label}:Ix",
        input_unit=f"{input_unit or 'value'} per stored-coordinate-unit",
        sigma_pixels=0.0,
    )
    d_iy = coordinate_gradient(
        first.iy,
        x,
        y,
        input_label=f"{input_label}:Iy",
        input_unit=f"{input_unit or 'value'} per stored-coordinate-unit",
        sigma_pixels=0.0,
    )

    iyx = d_ix.iy
    ixy_direct = d_iy.ix
    ixy = 0.5 * (iyx + ixy_direct)
    mixed_disagreement = np.abs(ixy_direct - iyx)

    trace = ixx + iyy
    determinant = ixx * iyy - ixy * ixy
    discriminant = np.sqrt(np.maximum((ixx - iyy) ** 2 + 4.0 * ixy * ixy, 0.0))

    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)
    laplacian = trace.copy()

    principal_orientation = np.degrees(
        0.5 * np.arctan2(2.0 * ixy, ixx - iyy)
    )

    curvedness = np.sqrt(0.5 * (lambda_min ** 2 + lambda_max ** 2))
    shape_index = (2.0 / np.pi) * np.arctan2(
        trace,
        np.maximum(lambda_max - lambda_min, 0.0),
    )

    bright_ridge = np.maximum(-lambda_min, 0.0)
    dark_valley = np.maximum(lambda_max, 0.0)

    positive_det_strength = np.sqrt(np.maximum(determinant, 0.0))
    bright_blob = np.where(trace < 0, positive_det_strength, 0.0)
    dark_blob = np.where(trace > 0, positive_det_strength, 0.0)

    arrays = [
        ixx, iyy, ixy, mixed_disagreement, trace, determinant,
        lambda_min, lambda_max, laplacian, principal_orientation,
        curvedness, shape_index, bright_ridge, dark_valley,
        bright_blob, dark_blob,
    ]
    _restore_nonfinite(arrays, finite)

    second_unit = f"{input_unit or 'value'} per stored-coordinate-unit^2"
    det_unit = f"({second_unit})^2"

    mixed_finite = mixed_disagreement[np.isfinite(mixed_disagreement)]
    common = {
        "sigma_pixels": sigma,
        "quadratic_radius": int(quadratic_radius),
        "spacing_mode": "actual_coordinate_values",
        "x_axis": _axis_summary(x),
        "y_axis": _axis_summary(y),
        "x_second_derivative": x_second_diag,
        "y_second_derivative": y_second_diag,
        "mixed_derivative": {
            "xy_path": "d(Iy)/dx",
            "yx_path": "d(Ix)/dy",
            "symmetric_value": "0.5 * (Ixy_path + Iyx_path)",
            "median_absolute_disagreement": (
                float(np.median(mixed_finite)) if mixed_finite.size else None
            ),
            "max_absolute_disagreement": (
                float(np.max(mixed_finite)) if mixed_finite.size else None
            ),
        },
        "coordinate_unit": None,
        "raw_data_modified": False,
    }

    def meta(name, description, unit, formula, **params):
        return make_feature_metadata(
            name=name,
            kind="derived_feature",
            description=description,
            input_label=input_label,
            output_unit=unit,
            parameters={
                "sigma_pixels": sigma,
                "quadratic_radius": int(quadratic_radius),
                **params,
            },
            provenance={**common, "formula": formula},
        )

    metadata = {
        "ixx": meta(
            "hessian_ixx",
            "Second derivative d2I/dx2.",
            second_unit,
            "d2I/dx2",
        ),
        "iyy": meta(
            "hessian_iyy",
            "Second derivative d2I/dy2.",
            second_unit,
            "d2I/dy2",
        ),
        "ixy": meta(
            "hessian_ixy",
            "Symmetrized mixed derivative.",
            second_unit,
            "0.5 * (d(Iy)/dx + d(Ix)/dy)",
        ),
        "mixed_disagreement": meta(
            "mixed_derivative_disagreement",
            "Absolute disagreement between the two mixed-derivative paths.",
            second_unit,
            "abs(d(Iy)/dx - d(Ix)/dy)",
        ),
        "trace": meta(
            "hessian_trace",
            "Trace of the 2x2 Hessian.",
            second_unit,
            "Ixx + Iyy",
        ),
        "determinant": meta(
            "hessian_determinant",
            "Determinant of the 2x2 Hessian.",
            det_unit,
            "Ixx*Iyy - Ixy^2",
        ),
        "lambda_min": meta(
            "hessian_lambda_min",
            "Algebraically smaller Hessian eigenvalue.",
            second_unit,
            "0.5*(trace - sqrt((Ixx-Iyy)^2 + 4*Ixy^2))",
        ),
        "lambda_max": meta(
            "hessian_lambda_max",
            "Algebraically larger Hessian eigenvalue.",
            second_unit,
            "0.5*(trace + sqrt((Ixx-Iyy)^2 + 4*Ixy^2))",
        ),
        "laplacian": meta(
            "laplacian",
            "Coordinate-aware Laplacian.",
            second_unit,
            "Ixx + Iyy",
        ),
        "principal_orientation_deg": meta(
            "principal_curvature_orientation",
            "Orientation of the eigenvector associated with lambda_max.",
            "degrees",
            "0.5*atan2(2*Ixy, Ixx-Iyy)",
            orientation_range="[-90,90] modulo 180",
        ),
        "curvedness": meta(
            "hessian_curvedness",
            "Magnitude of second-order curvature content.",
            second_unit,
            "sqrt((lambda_min^2 + lambda_max^2)/2)",
        ),
        "shape_index": meta(
            "hessian_shape_index",
            "Dimensionless local Hessian shape descriptor.",
            "dimensionless",
            "(2/pi)*atan2(trace, lambda_max-lambda_min)",
            nominal_range="[-1,1]",
        ),
        "bright_ridge": meta(
            "bright_ridge_response",
            "Simple bright-ridge curvature response from negative lambda_min.",
            second_unit,
            "max(-lambda_min, 0)",
        ),
        "dark_valley": meta(
            "dark_valley_response",
            "Simple dark-valley curvature response from positive lambda_max.",
            second_unit,
            "max(lambda_max, 0)",
        ),
        "bright_blob": meta(
            "bright_blob_response",
            "Same-sign negative-curvature blob response.",
            second_unit,
            "sqrt(max(det(H),0)) where trace(H)<0",
        ),
        "dark_blob": meta(
            "dark_blob_response",
            "Same-sign positive-curvature blob response.",
            second_unit,
            "sqrt(max(det(H),0)) where trace(H)>0",
        ),
    }

    return HessianField(
        ixx=ixx,
        iyy=iyy,
        ixy=ixy,
        mixed_disagreement=mixed_disagreement,
        trace=trace,
        determinant=determinant,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        laplacian=laplacian,
        principal_orientation_deg=principal_orientation,
        curvedness=curvedness,
        shape_index=shape_index,
        bright_ridge=bright_ridge,
        dark_valley=dark_valley,
        bright_blob=bright_blob,
        dark_blob=dark_blob,
        metadata=metadata,
    )
