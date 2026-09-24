import numpy as np
import plotly.graph_objects as go

from yeast_xrf.visualization.landscape import (
    build_landscape_height,
    display_surface_normals,
    hessian_point_class,
    landscape_figure,
)


def test_landscape_height_does_not_modify_source():
    source = np.arange(100, dtype=float).reshape(10, 10)
    original = source.copy()
    surface = build_landscape_height(
        source,
        low_percentile=1,
        high_percentile=99,
        vertical_exaggeration=4.0,
    )
    assert np.array_equal(source, original)
    assert surface.z.shape == source.shape
    assert surface.metadata["physical_z"] is False
    assert surface.metadata["raw_data_modified"] is False


def test_background_mask_is_explicitly_not_segmentation():
    source = np.arange(100, dtype=float).reshape(10, 10)
    surface = build_landscape_height(
        source,
        background_mask_percentile=50.0,
    )
    assert np.count_nonzero(~surface.visible_mask) > 0
    assert np.isnan(surface.z[~surface.visible_mask]).all()
    assert surface.metadata["background_mask_is_segmentation"] is False


def test_curvature_emphasis_changes_display_geometry_only():
    source = np.arange(100, dtype=float).reshape(10, 10)
    curvature = np.zeros_like(source)
    curvature[4:6, 4:6] = 10
    base = build_landscape_height(source, vertical_exaggeration=2.0)
    boosted = build_landscape_height(
        source,
        vertical_exaggeration=2.0,
        curvature=curvature,
        curvature_weight=1.5,
    )
    assert not np.allclose(base.z_unmasked, boosted.z_unmasked)
    assert boosted.metadata["curvature_emphasis"] is True


def test_display_surface_normal_on_plane():
    gx = np.full((4, 5), 2.0)
    gy = np.full((4, 5), 3.0)
    nx, ny, nz = display_surface_normals(gx, gy)
    norm = np.sqrt(14.0)
    assert np.allclose(nx, -2.0 / norm)
    assert np.allclose(ny, -3.0 / norm)
    assert np.allclose(nz, 1.0 / norm)


def test_hessian_point_classification():
    assert "saddle" in hessian_point_class(0.0, -1.0)
    assert "peak" in hessian_point_class(-4.0, 3.0)
    assert "bowl" in hessian_point_class(4.0, 3.0)


def test_landscape_figure_contains_surface_and_overlays():
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 7)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    source = xx + yy
    surface = build_landscape_height(source)
    ridge = np.zeros_like(source)
    ridge[:, 3] = 10
    blob = np.zeros_like(source)
    blob[3, 4] = 20

    fig = landscape_figure(
        x=x,
        y=y,
        surface=surface,
        surface_color=source,
        raw_height=source,
        raw_color=source,
        title="test",
        colorbar_title="value",
        ridge_response=ridge,
        blob_response=blob,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2
