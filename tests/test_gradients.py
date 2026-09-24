import numpy as np
import pytest

from yeast_xrf.features.gradients import (
    coordinate_gradient,
    directional_derivative,
    edge_mask,
    pixel_operator_gradient,
)


def test_nonuniform_plane_gradient():
    x = np.array([0.0, 0.4, 1.1, 2.0, 3.5, 5.0])
    y = np.array([-1.0, -0.2, 0.6, 1.8, 3.0])
    xx, yy = np.meshgrid(x, y)
    data = 2 * xx + 3 * yy
    g = coordinate_gradient(
        data, x, y, input_label="plane", input_unit="a.u.", sigma_pixels=0
    )
    assert np.allclose(g.ix, 2.0, atol=1e-10)
    assert np.allclose(g.iy, 3.0, atol=1e-10)


def test_repeated_y_coordinate_plane_gradient():
    x = np.array([0.0, 0.5, 1.2, 2.0, 3.1])
    y = np.array([0.0, 0.8, 0.8, 1.7, 2.8, 4.2])
    xx, yy = np.meshgrid(x, y)
    data = 2 * xx + 3 * yy
    g = coordinate_gradient(
        data, x, y, input_label="plane", input_unit="a.u.", sigma_pixels=0
    )
    assert np.allclose(g.ix, 2.0, atol=1e-10)
    assert np.allclose(g.iy, 3.0, atol=1e-10)
    prov = g.metadata["magnitude"].provenance
    assert prov["y_derivative_method"] == "local_coordinate_least_squares"
    assert prov["y_axis"]["repeated_adjacent_count"] == 1


def test_mixed_order_coordinate_still_uses_actual_positions():
    x = np.linspace(0, 1, 5)
    y = np.array([0.0, 1.0, 0.9, 2.0, 3.0])
    xx, yy = np.meshgrid(x, y)
    data = 4 * xx - 2 * yy
    g = coordinate_gradient(
        data, x, y, input_label="plane", input_unit="a.u.", sigma_pixels=0
    )
    assert np.allclose(g.ix, 4.0, atol=1e-10)
    assert np.allclose(g.iy, -2.0, atol=1e-10)
    assert g.metadata["iy"].provenance["y_derivative_method"] == (
        "local_coordinate_least_squares"
    )


def test_directional_projection():
    x = np.linspace(0, 1, 7)
    y = np.linspace(0, 1, 6)
    xx, yy = np.meshgrid(x, y)
    g = coordinate_gradient(
        4 * xx + 7 * yy,
        x,
        y,
        input_label="plane",
        input_unit="a.u.",
        sigma_pixels=0,
    )
    assert np.allclose(
        directional_derivative(g, angle_deg=0, input_label="p").data, 4
    )
    assert np.allclose(
        directional_derivative(g, angle_deg=90, input_label="p").data, 7
    )


@pytest.mark.parametrize("op", ["sobel", "scharr", "prewitt"])
def test_pixel_operators_constant_zero(op):
    g = pixel_operator_gradient(
        np.full((20, 20), 5.0),
        operator=op,
        input_label="Fe",
        input_unit="cts/s",
    )
    assert np.allclose(g.magnitude, 0, atol=1e-12)
    assert g.metadata["magnitude"].output_unit == "cts/s per pixel"


def test_edge_mask_percentile():
    r = edge_mask(
        np.arange(100).reshape(10, 10),
        percentile=90,
        input_label="g",
    )
    assert r.data.dtype == bool
    assert 9 <= np.count_nonzero(r.data) <= 11


def test_degenerate_coordinate_axis_rejected():
    with pytest.raises(ValueError):
        coordinate_gradient(
            np.ones((4, 4)),
            [0, 1, 2, 3],
            [2, 2, 2, 2],
            input_label="Fe",
            input_unit="cts/s",
        )

