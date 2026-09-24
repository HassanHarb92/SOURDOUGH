import numpy as np
import pytest

from yeast_xrf.analysis.beam_quantification import (
    apply_explicit_areal_density_conversion,
    beam_normalize,
    normalization_comparison,
    summarize_full_cells,
    transmission_diagnostic,
)


class Record:
    def __init__(self, cell_id, cropped):
        self.cell_id = cell_id
        self.cropped = cropped


def test_beam_normalization_is_explicit_and_masks_low_reference():
    numerator = np.array([[4.0, 4.0, 4.0]])
    reference = np.array([[2.0, 1.0, 0.0]])
    result = beam_normalize(
        numerator,
        reference,
        numerator_label="Zn",
        reference_label="US_IC",
        floor_percentile=0.0,
    )
    assert result.data[0, 0] == pytest.approx(2.0)
    assert np.isnan(result.data[0, 2])
    assert result.provenance["product_type"] == "beam_normalized_intensity"


def test_transmission_is_ds_over_us_and_not_hidden_correction():
    # The denominator rule is intentionally strict: values at or below the
    # chosen floor are masked. With a 0th-percentile positive floor of 1.0,
    # the first pixel is therefore invalid while the higher-US_IC pixels
    # exercise the DS_IC / US_IC formula.
    us = np.array([[1.0, 2.0, 4.0]])
    ds = np.array([[1.0, 1.0, 1.0]])
    result = transmission_diagnostic(us, ds, floor_percentile=0.0)
    assert np.isnan(result["data"][0, 0])
    assert result["data"][0, 1] == pytest.approx(0.5)
    assert result["data"][0, 2] == pytest.approx(0.25)
    assert "diagnostic" in result["interpretation"].lower()


def test_normalization_comparison():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    result = normalization_comparison(a, b)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["median_ratio_b_over_a"] == pytest.approx(2.0)


def test_explicit_areal_density_requires_calibration_source():
    with pytest.raises(ValueError):
        apply_explicit_areal_density_conversion(
            np.array([[1.0]]),
            slope=2.0,
            calibration_source="",
            reference_label="US_IC",
            channel_label="Zn",
            method="Fitted",
        )


def test_explicit_areal_density_formula_and_provenance():
    result = apply_explicit_areal_density_conversion(
        np.array([[1.0, 2.0]]),
        slope=3.0,
        intercept=0.5,
        output_unit="µg/cm²",
        calibration_source="validated external MAPS factor",
        reference_label="US_IC",
        channel_label="Zn",
        method="Fitted",
    )
    assert np.allclose(result.data, [[3.5, 6.5]])
    assert result.provenance["automatic_maps_factor_inference"] is False
    assert result.provenance["raw_data_modified"] is False


def test_full_cell_summary_excludes_cropped():
    data = np.arange(9, dtype=float).reshape(3, 3)
    labels = np.array(
        [
            [1, 1, 0],
            [1, 2, 2],
            [0, 2, 2],
        ]
    )
    rows = summarize_full_cells(
        data,
        labels,
        [Record(1, False), Record(2, True)],
    )
    assert len(rows) == 1
    assert rows[0]["cell_id"] == 1
