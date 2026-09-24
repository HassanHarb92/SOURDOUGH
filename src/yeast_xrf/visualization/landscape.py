'''Interactive 2.5D chemical-landscape helpers.

These functions create visualization geometry from 2D XRF data. They do not reconstruct a
physical Z coordinate and must not be interpreted as volumetric/tomographic reconstruction.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from yeast_xrf.visualization.transforms import percentile_stretch


@dataclass(frozen=True)
class LandscapeSurface:
    z: np.ndarray
    z_unmasked: np.ndarray
    base_normalized: np.ndarray
    visible_mask: np.ndarray
    metadata: dict[str, Any]


def build_landscape_height(
    data: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    vertical_exaggeration: float = 4.0,
    background_mask_percentile: float | None = None,
    curvature: np.ndarray | None = None,
    curvature_weight: float = 0.0,
) -> LandscapeSurface:
    '''Build display height from a 2D scalar map.

    Base height is robustly scaled to [0,1] and multiplied by vertical_exaggeration.

    Optional curvature emphasis is visualization-only:
        z <- z * (1 + curvature_weight * normalized_abs_curvature)

    Optional background masking sets low-signal display geometry to NaN. It is explicitly not
    a cell segmentation algorithm.
    '''
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("landscape source must be a 2D map")

    exaggeration = float(vertical_exaggeration)
    if not np.isfinite(exaggeration) or exaggeration <= 0:
        raise ValueError("vertical_exaggeration must be finite and > 0")

    weight = float(curvature_weight)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("curvature_weight must be finite and >= 0")

    base = percentile_stretch(
        arr,
        low=float(low_percentile),
        high=float(high_percentile),
    )
    z_unmasked = base * exaggeration

    if curvature is not None and weight > 0:
        curvature_arr = np.asarray(curvature, dtype=float)
        if curvature_arr.shape != arr.shape:
            raise ValueError("curvature shape must match source data")
        c = percentile_stretch(
            np.abs(curvature_arr),
            low=1.0,
            high=99.0,
        )
        z_unmasked = z_unmasked * (1.0 + weight * c)

    visible = np.isfinite(arr)
    mask_threshold = None

    if background_mask_percentile is not None:
        p = float(background_mask_percentile)
        if not 0 <= p < 100:
            raise ValueError("background_mask_percentile must satisfy 0 <= p < 100")
        finite = arr[np.isfinite(arr)]
        if finite.size:
            mask_threshold = float(np.percentile(finite, p))
            visible &= arr >= mask_threshold
        else:
            visible[:] = False

    z = np.asarray(z_unmasked, dtype=float).copy()
    z[~visible] = np.nan

    return LandscapeSurface(
        z=z,
        z_unmasked=np.asarray(z_unmasked, dtype=float),
        base_normalized=np.asarray(base, dtype=float),
        visible_mask=visible,
        metadata={
            "representation": "2.5D chemical landscape",
            "physical_z": False,
            "height_source": "robust percentile-scaled source intensity",
            "low_percentile": float(low_percentile),
            "high_percentile": float(high_percentile),
            "vertical_exaggeration": exaggeration,
            "background_mask_percentile": background_mask_percentile,
            "background_mask_threshold": mask_threshold,
            "background_mask_is_segmentation": False,
            "curvature_emphasis": bool(curvature is not None and weight > 0),
            "curvature_weight": weight,
            "raw_data_modified": False,
        },
    )


def display_surface_normals(
    dzdx: np.ndarray,
    dzdy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Return normalized display-surface normals (-dz/dx, -dz/dy, 1).'''
    gx = np.asarray(dzdx, dtype=float)
    gy = np.asarray(dzdy, dtype=float)
    if gx.shape != gy.shape:
        raise ValueError("dzdx and dzdy must have the same shape")

    nx = -gx
    ny = -gy
    nz = np.ones(gx.shape, dtype=float)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    valid = np.isfinite(norm) & (norm > 0)

    out_x = np.full(gx.shape, np.nan, dtype=float)
    out_y = np.full(gx.shape, np.nan, dtype=float)
    out_z = np.full(gx.shape, np.nan, dtype=float)

    np.divide(nx, norm, out=out_x, where=valid)
    np.divide(ny, norm, out=out_y, where=valid)
    np.divide(nz, norm, out=out_z, where=valid)

    return out_x, out_y, out_z


def hessian_point_class(trace: float, determinant: float, tolerance: float = 0.0) -> str:
    '''Simple deterministic Hessian-sign classification for local inspection.'''
    tr = float(trace)
    det = float(determinant)
    tol = abs(float(tolerance))

    if not np.isfinite(tr) or not np.isfinite(det):
        return "nonfinite"
    if det < -tol:
        return "saddle-like"
    if det > tol and tr < -tol:
        return "peak / bright-blob-like curvature"
    if det > tol and tr > tol:
        return "bowl / dark-blob-like curvature"
    return "near-flat or degenerate second-order structure"


def _overlay_trace(
    *,
    response: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    visible_mask: np.ndarray,
    percentile: float,
    name: str,
    max_points: int,
) -> go.Scatter3d | None:
    response = np.asarray(response, dtype=float)
    finite = response[np.isfinite(response) & visible_mask]
    if finite.size == 0:
        return None

    threshold = float(np.percentile(finite, float(percentile)))
    mask = (
        np.isfinite(response)
        & visible_mask
        & np.isfinite(z)
        & (response >= threshold)
    )
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None

    if yy.size > max_points:
        stride = int(np.ceil(yy.size / max_points))
        yy = yy[::stride]
        xx = xx[::stride]

    finite_z = z[np.isfinite(z)]
    lift = 0.02 * float(np.ptp(finite_z)) if finite_z.size else 0.0

    return go.Scatter3d(
        x=x[xx],
        y=y[yy],
        z=z[yy, xx] + lift,
        mode="markers",
        marker={"size": 3},
        name=name,
        customdata=response[yy, xx],
        hovertemplate=(
            f"{name}<br>"
            "x=%{x:.6g}<br>"
            "y=%{y:.6g}<br>"
            "display z=%{z:.6g}<br>"
            "response=%{customdata:.6g}<extra></extra>"
        ),
    )


def landscape_figure(
    *,
    x: np.ndarray,
    y: np.ndarray,
    surface: LandscapeSurface,
    surface_color: np.ndarray,
    raw_height: np.ndarray,
    raw_color: np.ndarray,
    title: str,
    colorbar_title: str,
    colorscale: str = "Viridis",
    opacity: float = 1.0,
    show_z_contours: bool = False,
    ridge_response: np.ndarray | None = None,
    ridge_percentile: float = 97.5,
    blob_response: np.ndarray | None = None,
    blob_percentile: float = 97.5,
) -> go.Figure:
    '''Build the interactive Plotly 2.5D landscape.'''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    color = np.asarray(surface_color, dtype=float)
    raw_height = np.asarray(raw_height, dtype=float)
    raw_color = np.asarray(raw_color, dtype=float)

    shape = surface.z.shape
    for name, arr in (
        ("surface_color", color),
        ("raw_height", raw_height),
        ("raw_color", raw_color),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} != surface shape {shape}")

    custom = np.dstack([raw_height, raw_color])

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=surface.z,
                surfacecolor=color,
                colorscale=colorscale,
                opacity=float(opacity),
                colorbar={"title": colorbar_title},
                customdata=custom,
                hovertemplate=(
                    "x=%{x:.6g}<br>"
                    "y=%{y:.6g}<br>"
                    "display height=%{z:.6g}<br>"
                    "raw height source=%{customdata[0]:.6g}<br>"
                    "raw color source=%{customdata[1]:.6g}<extra></extra>"
                ),
                contours={
                    "z": {
                        "show": bool(show_z_contours),
                        "usecolormap": True,
                        "project_z": bool(show_z_contours),
                    }
                },
                lighting={
                    "ambient": 0.55,
                    "diffuse": 0.75,
                    "specular": 0.25,
                    "roughness": 0.65,
                    "fresnel": 0.15,
                },
            )
        ]
    )

    if ridge_response is not None:
        trace = _overlay_trace(
            response=ridge_response,
            z=surface.z,
            x=x,
            y=y,
            visible_mask=surface.visible_mask,
            percentile=ridge_percentile,
            name="bright ridge",
            max_points=1800,
        )
        if trace is not None:
            fig.add_trace(trace)

    if blob_response is not None:
        trace = _overlay_trace(
            response=blob_response,
            z=surface.z,
            x=x,
            y=y,
            visible_mask=surface.visible_mask,
            percentile=blob_percentile,
            name="bright blob",
            max_points=1800,
        )
        if trace is not None:
            fig.add_trace(trace)

    fig.update_layout(
        title=title,
        height=780,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "X coordinate",
            "yaxis_title": "Y coordinate",
            "zaxis_title": "Derived/display height",
            "aspectmode": "data",
        },
        uirevision="yeast-cell-landscape",
        legend={"orientation": "h"},
    )
    return fig
