'''Model-based 3D cell geometry from a measured 2D cell footprint.

The model preserves the measured X/Y mask but infers depth. It is therefore not an
experimental 3D reconstruction.

Interior XRF slices can use a projection-conserving uniform-depth model:

    rho(x,y) = measured_2D_XRF(x,y) / inferred_thickness(x,y)

so integrating rho through the inferred thickness returns the original measured 2D map.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
import plotly.graph_objects as go


@dataclass(frozen=True)
class CellEnvelope:
    top: np.ndarray
    bottom: np.ndarray
    half_thickness: np.ndarray
    mask: np.ndarray
    max_half_depth: float
    metadata: dict


def _median_step(axis):
    d = np.abs(np.diff(np.asarray(axis, dtype=float)))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 1.0


def build_mask_conforming_envelope(
    mask,
    x,
    y,
    *,
    depth_ratio_to_minor_radius=1.0,
    minor_axis_coordinate=None,
    dome_power=0.75,
):
    mask = np.asarray(mask, dtype=bool)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if mask.ndim != 2:
        raise ValueError("cell mask must be 2D")
    if x.size != mask.shape[1] or y.size != mask.shape[0]:
        raise ValueError("coordinate lengths do not match mask")
    if not np.any(mask):
        raise ValueError("cell mask is empty")

    dx = _median_step(x)
    dy = _median_step(y)
    distance = ndi.distance_transform_edt(mask, sampling=(dy, dx))
    max_distance = float(np.max(distance))
    if max_distance <= 0:
        raise ValueError("cell mask has no interior distance")

    if minor_axis_coordinate is None or not np.isfinite(minor_axis_coordinate):
        inferred_minor_radius = max_distance
    else:
        inferred_minor_radius = max(float(minor_axis_coordinate) / 2.0, min(dx, dy))

    ratio = float(depth_ratio_to_minor_radius)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("depth_ratio_to_minor_radius must be finite and > 0")

    power = float(dome_power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError("dome_power must be finite and > 0")

    max_half_depth = inferred_minor_radius * ratio
    normalized_distance = np.clip(distance / max_distance, 0.0, 1.0)
    half = max_half_depth * np.power(
        np.sin(0.5 * np.pi * normalized_distance),
        power,
    )
    half[~mask] = np.nan

    return CellEnvelope(
        top=half.copy(),
        bottom=-half,
        half_thickness=half,
        mask=mask,
        max_half_depth=max_half_depth,
        metadata={
            "representation": "inferred mask-conforming cell envelope",
            "physical_z_measured": False,
            "xy_footprint_measured": True,
            "depth_inferred": True,
            "depth_ratio_to_minor_radius": ratio,
            "minor_axis_coordinate_approx": (
                float(minor_axis_coordinate)
                if minor_axis_coordinate is not None
                else None
            ),
            "dome_power": power,
            "distance_sampling_dx": dx,
            "distance_sampling_dy": dy,
            "z_unit": "stored-coordinate-unit (inferred)",
        },
    )


def projection_conserving_density(
    measured_map,
    envelope: CellEnvelope,
    *,
    thickness_floor_fraction=0.10,
):
    measured = np.asarray(measured_map, dtype=float)
    if measured.shape != envelope.mask.shape:
        raise ValueError("measured_map shape does not match envelope")

    half = envelope.half_thickness
    positive = half[np.isfinite(half) & (half > 0)]
    if positive.size == 0:
        return np.full(measured.shape, np.nan), np.nan

    floor = float(np.percentile(positive, float(thickness_floor_fraction) * 100.0))
    effective_half = np.where(np.isfinite(half), np.maximum(half, floor), np.nan)
    thickness = 2.0 * effective_half

    density = np.full(measured.shape, np.nan, dtype=float)
    valid = envelope.mask & np.isfinite(measured) & np.isfinite(thickness) & (thickness > 0)
    np.divide(measured, thickness, out=density, where=valid)

    return density, floor


def internal_slice(
    density,
    envelope: CellEnvelope,
    *,
    z_fraction=0.0,
):
    fraction = float(z_fraction)
    if not -1.0 <= fraction <= 1.0:
        raise ValueError("z_fraction must be between -1 and 1")

    z_value = fraction * envelope.max_half_depth
    visible = (
        envelope.mask
        & np.isfinite(envelope.half_thickness)
        & (np.abs(z_value) <= envelope.half_thickness)
    )
    slice_data = np.asarray(density, dtype=float).copy()
    slice_data[~visible] = np.nan
    return slice_data, float(z_value), visible


def cell_model_figure(
    *,
    x,
    y,
    envelope: CellEnvelope,
    surface_color,
    raw_surface_color,
    slice_data=None,
    slice_z=None,
    title,
    colorscale="Viridis",
    surface_opacity=0.45,
    show_bottom=True,
    cutaway_axis="none",
    cutaway_fraction=0.50,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    surface_color = np.asarray(surface_color, dtype=float)
    raw_surface_color = np.asarray(raw_surface_color, dtype=float)

    top = envelope.top.copy()
    bottom = envelope.bottom.copy()
    mask = envelope.mask.copy()

    axis = str(cutaway_axis).lower()
    fraction = float(cutaway_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("cutaway_fraction must be between 0 and 1")

    if axis == "x":
        cutoff = float(np.min(x) + fraction * (np.max(x) - np.min(x)))
        mask &= x[None, :] <= cutoff
    elif axis == "y":
        cutoff = float(np.min(y) + fraction * (np.max(y) - np.min(y)))
        mask &= y[:, None] <= cutoff
    elif axis != "none":
        raise ValueError("cutaway_axis must be none, x, or y")

    top[~mask] = np.nan
    bottom[~mask] = np.nan

    custom = np.dstack([raw_surface_color, envelope.half_thickness])

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=top,
            surfacecolor=surface_color,
            colorscale=colorscale,
            opacity=float(surface_opacity),
            customdata=custom,
            name="cell envelope top",
            colorbar={"title": "surface color"},
            hovertemplate=(
                "x=%{x:.6g}<br>"
                "y=%{y:.6g}<br>"
                "inferred z=%{z:.6g}<br>"
                "measured XRF=%{customdata[0]:.6g}<br>"
                "inferred half-thickness=%{customdata[1]:.6g}<extra></extra>"
            ),
            showscale=True,
        )
    )

    if show_bottom:
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=bottom,
                surfacecolor=surface_color,
                colorscale=colorscale,
                opacity=max(0.10, float(surface_opacity) * 0.65),
                customdata=custom,
                name="cell envelope bottom",
                showscale=False,
                hovertemplate=(
                    "x=%{x:.6g}<br>"
                    "y=%{y:.6g}<br>"
                    "inferred z=%{z:.6g}<br>"
                    "measured XRF=%{customdata[0]:.6g}<extra></extra>"
                ),
            )
        )

    if slice_data is not None and slice_z is not None:
        slice_data = np.asarray(slice_data, dtype=float)
        slice_plane = np.full(envelope.mask.shape, float(slice_z))
        slice_plane[~np.isfinite(slice_data)] = np.nan
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=slice_plane,
                surfacecolor=slice_data,
                colorscale="Turbo",
                opacity=0.95,
                name="internal XRF slice",
                showscale=False,
                hovertemplate=(
                    "internal model slice<br>"
                    "x=%{x:.6g}<br>"
                    "y=%{y:.6g}<br>"
                    "inferred z=%{z:.6g}<br>"
                    "modeled density=%{surfacecolor:.6g}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        height=800,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "X coordinate",
            "yaxis_title": "Y coordinate",
            "zaxis_title": "Inferred depth coordinate",
            "aspectmode": "data",
        },
        uirevision="yeast-inferred-cell-model",
        legend={"orientation": "h"},
    )
    return fig
