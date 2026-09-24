import numpy as np

from yeast_xrf.features.cells import segment_cells_tfy


def synthetic_tfy():
    yy, xx = np.mgrid[:120, :140]
    image = np.zeros((120, 140), dtype=float)

    # Two complete cells.
    image[((xx - 40) / 15) ** 2 + ((yy - 45) / 12) ** 2 <= 1] = 8.0
    image[((xx - 90) / 13) ** 2 + ((yy - 75) / 16) ** 2 <= 1] = 10.0

    # One cropped cell touching the left border.
    image[((xx - 2) / 14) ** 2 + ((yy - 90) / 12) ** 2 <= 1] = 9.0

    image += 0.05
    return image


def test_detects_complete_and_cropped_cells():
    image = synthetic_tfy()
    x = np.arange(image.shape[1], dtype=float) * 0.08
    y = np.arange(image.shape[0], dtype=float) * 0.08

    result = segment_cells_tfy(
        image,
        x,
        y,
        threshold_method="otsu",
        smooth_sigma_pixels=0.5,
        min_area_pixels=80,
        closing_radius_pixels=1,
        split_touching=False,
    )

    assert result.total_cells == 3
    assert result.complete_cells == 2
    assert result.cropped_cells == 1

    cropped = [record for record in result.records if record.cropped]
    assert len(cropped) == 1
    assert cropped[0].touches_left


def test_cell_geometry_is_reported_in_coordinate_units():
    image = synthetic_tfy()
    x = np.arange(image.shape[1], dtype=float) * 0.10
    y = np.arange(image.shape[0], dtype=float) * 0.20

    result = segment_cells_tfy(
        image,
        x,
        y,
        smooth_sigma_pixels=0.5,
        min_area_pixels=80,
        split_touching=False,
    )
    complete = [record for record in result.records if not record.cropped][0]
    assert complete.coordinate_area_approx > 0
    assert complete.major_axis_coordinate_approx > 0
    assert complete.minor_axis_coordinate_approx > 0
    assert result.metadata["physical_unit_verified"] is False


def test_repeated_coordinate_values_do_not_break_morphometry():
    image = synthetic_tfy()
    x = np.arange(image.shape[1], dtype=float) * 0.08
    y = np.arange(image.shape[0], dtype=float) * 0.08
    y[50] = y[49]

    result = segment_cells_tfy(
        image,
        x,
        y,
        smooth_sigma_pixels=0.5,
        min_area_pixels=80,
        split_touching=False,
    )
    assert result.total_cells == 3
