import numpy as np
import pytest

from yeast_xrf.analysis.normalization import (
    InvalidReason,
    invalid_reason_counts,
    normalize_by_reference,
    positive_reference_percentile_floor,
)
from yeast_xrf.analysis.qc import (
    QCFlag,
    build_qc_flags,
    count_rate_diagnostics,
    flag_mask,
    live_time_diagnostics,
)


def test_qc_flags_negative_without_invalidating_data() -> None:
    data = np.array([[-2.0, 0.0], [2.0, np.nan]])
    result = build_qc_flags(data)
    assert flag_mask(result.flags, QCFlag.NEGATIVE)[0, 0]
    assert flag_mask(result.flags, QCFlag.ZERO)[0, 1]
    assert flag_mask(result.flags, QCFlag.NONFINITE)[1, 1]
    assert data[0, 0] == -2.0


def test_high_value_flag_is_candidate_not_saturation_claim() -> None:
    data = np.arange(100, dtype=float)
    result = build_qc_flags(data, high_value_percentile=95.0)
    assert result.summary["high_value_candidate_count"] > 0
    assert result.summary["high_value_candidate_threshold"] is not None


def test_normalization_preserves_negative_by_default() -> None:
    numerator = np.array([[-2.0, 2.0]])
    reference = np.array([[2.0, 2.0]])
    result = normalize_by_reference(
        numerator,
        reference,
        numerator_label="Fe",
        reference_label="US_IC",
        denominator_floor=0.0,
    )
    assert result.valid_mask.all()
    assert result.data[0, 0] == pytest.approx(-1.0)
    assert result.data[0, 1] == pytest.approx(1.0)


def test_normalization_can_explicitly_mask_negative_values() -> None:
    numerator = np.array([[-2.0, 2.0]])
    reference = np.array([[2.0, 2.0]])
    result = normalize_by_reference(
        numerator,
        reference,
        numerator_label="Fe",
        reference_label="US_IC",
        denominator_floor=0.0,
        mask_negative_numerator=True,
    )
    assert np.isnan(result.data[0, 0])
    assert result.valid_mask[0, 1]
    assert (
        result.invalid_reason[0, 0] & InvalidReason.NUMERATOR_NEGATIVE_MASKED
    ) != 0


def test_low_reference_is_masked_instead_of_exploding_ratio() -> None:
    numerator = np.array([[1.0, 1.0, 1.0]])
    reference = np.array([[1.0, 0.01, 0.0]])
    result = normalize_by_reference(
        numerator,
        reference,
        numerator_label="Zn",
        reference_label="US_IC",
        denominator_floor=0.05,
    )
    assert result.data[0, 0] == pytest.approx(1.0)
    assert np.isnan(result.data[0, 1])
    assert np.isnan(result.data[0, 2])
    counts = invalid_reason_counts(result.invalid_reason)
    assert counts["reference_at_or_below_floor"] == 2


def test_scale_factor_is_explicit() -> None:
    result = normalize_by_reference(
        np.array([[4.0]]),
        np.array([[2.0]]),
        numerator_label="Fe",
        reference_label="US_IC",
        denominator_floor=0.0,
        scale_factor=100.0,
    )
    assert result.data[0, 0] == pytest.approx(200.0)
    assert result.provenance["scale_factor"] == 100.0


def test_inputs_are_not_modified() -> None:
    numerator = np.array([[1.0, -1.0]])
    reference = np.array([[2.0, 2.0]])
    n0 = numerator.copy()
    r0 = reference.copy()
    normalize_by_reference(
        numerator,
        reference,
        numerator_label="P",
        reference_label="US_IC",
        denominator_floor=0.0,
    )
    assert np.array_equal(numerator, n0)
    assert np.array_equal(reference, r0)


def test_positive_reference_percentile_floor_ignores_nonpositive() -> None:
    ref = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, np.nan])
    floor = positive_reference_percentile_floor(ref, 50.0)
    assert floor == pytest.approx(2.0)


def test_icr_ocr_diagnostics_are_unclipped() -> None:
    icr = np.array([[10.0, 10.0]])
    ocr = np.array([[8.0, 12.0]])
    result = count_rate_diagnostics(icr, ocr)
    assert result["ocr_over_icr"][0, 0] == pytest.approx(0.8)
    assert result["implied_dead_time_fraction"][0, 0] == pytest.approx(0.2)
    assert result["implied_dead_time_fraction"][0, 1] == pytest.approx(-0.2)


def test_live_time_fraction() -> None:
    result = live_time_diagnostics(
        np.array([[8.0]]),
        np.array([[10.0]]),
    )
    assert result["live_time_fraction"][0, 0] == pytest.approx(0.8)
