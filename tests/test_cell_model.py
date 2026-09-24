import numpy as np
import plotly.graph_objects as go

from yeast_xrf.visualization.cell_model import (
    build_mask_conforming_envelope,
    cell_model_figure,
    internal_slice,
    projection_conserving_density,
)


def circular_mask(n=61, radius=22):
    yy, xx = np.mgrid[:n, :n]
    return (xx - n // 2) ** 2 + (yy - n // 2) ** 2 <= radius**2


def test_envelope_preserves_measured_xy_footprint():
    mask = circular_mask()
    x = np.linspace(-3, 3, mask.shape[1])
    y = np.linspace(-3, 3, mask.shape[0])

    env = build_mask_conforming_envelope(
        mask,
        x,
        y,
        depth_ratio_to_minor_radius=1.0,
        minor_axis_coordinate=4.0,
    )

    assert np.array_equal(np.isfinite(env.top), mask)
    assert np.allclose(env.top[mask], -env.bottom[mask])
    assert env.metadata["physical_z_measured"] is False
    assert env.metadata["xy_footprint_measured"] is True


def test_projection_conserving_density_integrates_back_to_2d_map():
    mask = circular_mask()
    x = np.linspace(-3, 3, mask.shape[1])
    y = np.linspace(-3, 3, mask.shape[0])
    measured = np.where(mask, 12.0, np.nan)

    env = build_mask_conforming_envelope(
        mask, x, y,
        depth_ratio_to_minor_radius=1.0,
        minor_axis_coordinate=4.0,
    )
    density, floor = projection_conserving_density(
        measured, env, thickness_floor_fraction=0.10
    )
    effective_half = np.maximum(env.half_thickness, floor)
    recovered = density * 2.0 * effective_half
    valid = mask & np.isfinite(recovered)
    assert np.allclose(recovered[valid], measured[valid], atol=1e-10)


def test_internal_slice_is_inside_envelope():
    mask = circular_mask()
    x = np.linspace(-3, 3, mask.shape[1])
    y = np.linspace(-3, 3, mask.shape[0])
    measured = np.where(mask, 5.0, np.nan)
    env = build_mask_conforming_envelope(mask, x, y, minor_axis_coordinate=4.0)
    density, _ = projection_conserving_density(measured, env)
    slice_data, z, visible = internal_slice(density, env, z_fraction=0.0)
    assert z == 0.0
    assert np.array_equal(np.isfinite(slice_data), visible)
    assert np.count_nonzero(visible) > 0


def test_cell_model_figure_has_surfaces():
    mask = circular_mask()
    x = np.linspace(-3, 3, mask.shape[1])
    y = np.linspace(-3, 3, mask.shape[0])
    measured = np.where(mask, 5.0, np.nan)
    env = build_mask_conforming_envelope(mask, x, y, minor_axis_coordinate=4.0)
    density, _ = projection_conserving_density(measured, env)
    slice_data, z, _ = internal_slice(density, env, z_fraction=0.0)

    fig = cell_model_figure(
        x=x,
        y=y,
        envelope=env,
        surface_color=measured,
        raw_surface_color=measured,
        slice_data=slice_data,
        slice_z=z,
        title="test",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2
