import numpy as np

from yeast_xrf.features.hessian import coordinate_hessian


def quadratic_surface(x, y):
    xx, yy = np.meshgrid(x, y)
    return 2.0 * xx**2 + 5.0 * xx * yy + 3.0 * yy**2 + 7.0 * xx - 4.0 * yy + 1.0


def test_quadratic_hessian_nonuniform_grid():
    x = np.array([0.0, 0.4, 1.1, 2.0, 3.5, 5.0])
    y = np.array([-1.0, -0.2, 0.6, 1.8, 3.0, 4.6])
    h = coordinate_hessian(
        quadratic_surface(x, y),
        x,
        y,
        input_label="quadratic",
        input_unit="a.u.",
        sigma_pixels=0.0,
        quadratic_radius=2,
    )
    assert np.allclose(h.ixx, 4.0, atol=1e-10)
    assert np.allclose(h.iyy, 6.0, atol=1e-10)
    assert np.allclose(h.ixy, 5.0, atol=1e-10)
    assert np.nanmax(h.mixed_disagreement) < 1e-10
    assert np.allclose(h.trace, 10.0, atol=1e-10)
    assert np.allclose(h.determinant, -1.0, atol=1e-10)


def test_quadratic_hessian_with_repeated_y():
    x = np.array([0.0, 0.4, 1.1, 2.0, 3.5, 5.0])
    y = np.array([-1.0, -0.2, -0.2, 0.6, 1.8, 3.0, 4.6])
    h = coordinate_hessian(
        quadratic_surface(x, y),
        x,
        y,
        input_label="quadratic",
        input_unit="a.u.",
        sigma_pixels=0.0,
        quadratic_radius=2,
    )
    assert np.allclose(h.ixx, 4.0, atol=1e-10)
    assert np.allclose(h.iyy, 6.0, atol=1e-10)
    assert np.allclose(h.ixy, 5.0, atol=1e-10)
    assert np.nanmax(h.mixed_disagreement) < 1e-10
    assert h.metadata["iyy"].provenance["y_axis"]["repeated_adjacent_count"] == 1


def test_eigenvalues_match_known_matrix():
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-3, 3, 11)
    xx, yy = np.meshgrid(x, y)
    data = 0.5 * (4.0 * xx**2 + 2.0 * 5.0 * xx * yy + 6.0 * yy**2)
    h = coordinate_hessian(
        data, x, y,
        input_label="known_hessian",
        input_unit="a.u.",
        sigma_pixels=0.0,
        quadratic_radius=2,
    )
    expected = np.linalg.eigvalsh(np.array([[4.0, 5.0], [5.0, 6.0]]))
    assert np.allclose(h.lambda_min, expected[0], atol=1e-10)
    assert np.allclose(h.lambda_max, expected[1], atol=1e-10)


def test_bright_blob_response_for_negative_definite_peak():
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-2, 2, 9)
    xx, yy = np.meshgrid(x, y)
    data = -(xx**2 + yy**2)
    h = coordinate_hessian(
        data, x, y,
        input_label="peak", input_unit="a.u.",
        sigma_pixels=0.0, quadratic_radius=2,
    )
    assert np.all(h.bright_blob > 0)
    assert np.allclose(h.dark_blob, 0.0)
    assert np.all(h.bright_ridge > 0)


def test_shape_index_is_dimensionless_and_bounded():
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-2, 2, 9)
    xx, yy = np.meshgrid(x, y)
    h = coordinate_hessian(
        xx**2 + 3*yy**2,
        x, y,
        input_label="bowl", input_unit="a.u.",
        sigma_pixels=0.0, quadratic_radius=2,
    )
    assert np.nanmin(h.shape_index) >= -1.0
    assert np.nanmax(h.shape_index) <= 1.0
    assert h.metadata["shape_index"].output_unit == "dimensionless"
