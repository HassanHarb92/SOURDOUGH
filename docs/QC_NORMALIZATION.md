# QC and Normalization Framework

Change 05 introduces explicit XRF QC flags, reference/scaler normalization, detector-rate
diagnostics, and normalization provenance.

## Core principle

Raw data remain raw.

No scaler correction, dead-time correction, livetime correction, clipping, or mask is applied
unless the caller explicitly requests it.

## QC flags

The selected XRF map can be flagged for:

- nonfinite values
- negative values
- zero values
- an explicitly supplied low-signal threshold
- a user-selected upper-tail percentile

The upper-tail flag is called `high_value_candidate`, not saturation. True detector saturation
requires detector-specific limits that are not established by the current HDF5 metadata.

Negative fitted values are not automatically invalid. They are retained unless the caller
explicitly chooses to mask them.

## Reference normalization

The canonical operation is:

```text
normalized = (numerator / reference) × scale_factor
```

The caller explicitly supplies:

- XRF channel / numerator
- reference/scaler
- reference family
- denominator floor policy
- scale factor
- whether negative numerator values are masked
- whether zero numerator values are masked

Reference pixels at or below the denominator floor are masked rather than producing unstable
or infinite ratios.

## Denominator floor policies

The Explorer supports:

1. an explicit absolute floor;
2. a percentile calculated only from positive finite reference pixels.

The chosen numeric floor is always recorded in provenance.

## Provenance

Each normalization result records:

- operation
- formula
- numerator label
- reference label
- denominator rule and numeric floor
- scale factor
- negative/zero mask choices
- analysis method
- source identity
- confirmation that raw data were not modified

## Detector/acquisition diagnostics

Where available:

```text
OCR / ICR
1 - OCR / ICR
ELT / ERT
```

are exposed as diagnostic maps.

The derived `1 - OCR/ICR` values are not clipped to [0,1], because anomalous values are useful
QC evidence.

These diagnostics are not automatically applied as corrections.

## Scientific boundary

The presence of US_IC, DS_IC, ELT, ERT, ICR, OCR, Dead_Time, and other scalers does not by
itself establish which normalization is scientifically appropriate for a particular
experiment.

Change 05 makes the candidate operations inspectable and reproducible. Choosing an experimental
normalization strategy remains an explicit scientific decision.
