# Change 10 — MAPS calibration semantics

SOURDOUGH does not unlock automatic concentration conversion from array shape alone.

The Change-10 probe records, without modifying HDF5:

- analyzed channel names and units;
- MAPS calibration-curve labels and values;
- per-reference element-info index, names, values, and HDF5 attributes;
- legacy MAPS quantification arrays and names;
- row-wise numeric fingerprints;
- cross-scan semantic signatures for Fitted, NNLS, and ROI.

## Acceptance gate

Automatic areal-density conversion remains locked until the stored coefficient meaning and
formula are demonstrated from the actual files and checked against trusted MAPS semantics.

The eventual conversion must preserve:

- method separation: Fitted / NNLS / ROI;
- explicit reference identity;
- physical output unit;
- full provenance;
- read-only raw HDF5;
- no conversion of pixel sums to total elemental mass until physical pixel area is verified.

## Probe

```bash
PYTHONPATH=$PWD/src python scripts/probe_maps_calibration_semantics.py
```

Outputs:

```text
analysis/quantification/change10_semantics_probe.json
analysis/quantification/change10_semantics_probe.txt
```
