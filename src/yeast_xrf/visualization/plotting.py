"""Plotly figure builders used by the XRF Explorer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def heatmap_figure(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    colorscale: str = "Viridis",
    colorbar_title: str = "",
    zmin: float | None = None,
    zmax: float | None = None,
    marker_xy: tuple[float, float] | None = None,
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=np.asarray(data),
            x=np.asarray(x),
            y=np.asarray(y),
            colorscale=colorscale,
            colorbar={"title": colorbar_title},
            zmin=zmin,
            zmax=zmax,
            hovertemplate=(
                "x=%{x:.6g}<br>y=%{y:.6g}<br>value=%{z:.6g}<extra></extra>"
            ),
        )
    )
    if marker_xy is not None:
        fig.add_trace(
            go.Scatter(
                x=[marker_xy[0]],
                y=[marker_xy[1]],
                mode="markers",
                marker={"size": 11, "symbol": "x"},
                name="selected pixel",
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        margin={"l": 50, "r": 30, "t": 55, "b": 50},
        height=580,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def spectrum_figure(
    energy: np.ndarray,
    intensity: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(
            x=np.asarray(energy),
            y=np.asarray(intensity),
            mode="lines",
            name="pixel spectrum",
            hovertemplate="E=%{x:.6g}<br>I=%{y:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Energy",
        yaxis_title="Intensity",
        height=430,
        margin={"l": 55, "r": 25, "t": 55, "b": 50},
    )
    return fig


def histogram_figure(
    data: np.ndarray,
    *,
    title: str,
    bins: int = 80,
) -> go.Figure:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    fig = go.Figure(
        data=go.Histogram(
            x=finite,
            nbinsx=int(bins),
            hovertemplate="value=%{x:.6g}<br>count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Pixels",
        height=320,
        margin={"l": 55, "r": 25, "t": 50, "b": 45},
    )
    return fig


def surface_figure(
    display_data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Interactive 2.5D surface. Height is display intensity, not physical depth."""
    fig = go.Figure(
        data=go.Surface(
            z=np.asarray(display_data),
            x=np.asarray(x),
            y=np.asarray(y),
            surfacecolor=np.asarray(display_data),
            colorscale=colorscale,
            colorbar={"title": "display value"},
        )
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "X coordinate",
            "yaxis_title": "Y coordinate",
            "zaxis_title": "Display intensity",
            "aspectmode": "auto",
        },
        height=650,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    return fig
