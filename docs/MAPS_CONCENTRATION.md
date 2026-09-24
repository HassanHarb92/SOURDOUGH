# Change 10 — Automatic MAPS areal-density conversion

SOURDOUGH uses the calibration factors embedded by MAPS, but only after verifying
each factor against the exact channel label in the stored MAPS calibration curve.

For these files, the legacy quantification arrays map as:

- row 0: SR_Current
- row 1: US_IC
- row 2: DS_IC

The automatic conversion implemented for US_IC and DS_IC is:

```text
areal density (µg/cm²)
    = (analyzed counts/s / scaler) / MAPS calibration factor
```

The factor is rejected if it does not match the corresponding calibration-curve
value within numerical tolerance.

## Scientific safeguards

- Fitted, NNLS, and ROI remain separate.
- US_IC and DS_IC remain explicit choices.
- Raw HDF5 is read-only.
- Negative analyzed values are preserved.
- Low/invalid scaler pixels are masked explicitly.
- Pixel sums are not interpreted as total elemental mass until the physical
  X/Y pixel-area unit is verified.
- Non-element channels without a MAPS calibration-curve label are not
  automatically quantified.

## Validation

```bash
PYTHONPATH=$PWD/src python scripts/validate_maps_concentration.py
```

Output:

```text
analysis/quantification/change10_concentration_validation.json
```
