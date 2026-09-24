# Change 09 — Beam Normalization and Quantitative XRF

SOURDOUGH keeps fitted XRF signal, beam normalization, and physical concentration conversion
as separate scientific operations.

## Beam normalization

Change 09 provides explicit normalization to `US_IC` and `DS_IC`, with positive-reference
floors, masks, scale factors, and provenance. Raw arrays are never modified.

`DS_IC / US_IC` is exposed only as a transmission-style diagnostic.

## MAPS quantification metadata

The read-only IO layer inspects standard composition metadata, calibration curves,
per-reference element-info arrays, and legacy quantification-candidate arrays.

The current HDF5 schema contains substantial MAPS calibration metadata, but Change 09 does not
guess which stored coefficient is the physical conversion factor.

## Explicit validated factor mode

When a beamline scientist or trusted MAPS export supplies a validated linear relationship:

```text
areal_density = slope × beam_normalized_intensity + intercept
```

SOURDOUGH can apply it with explicit unit and provenance.

## Per-cell summary

When an areal-density map exists, complete cells only are summarized. Cropped cells are
excluded.

The sum of pixel values is not converted to total elemental mass until the physical X/Y area
unit is verified.

## Validation

```bash
make validate-quantification
```

writes:

```text
analysis/quantification/change09_validation.json
```
