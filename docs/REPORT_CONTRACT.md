# Change 14B — Final Report Contract

Change 14B establishes the stable collaborator-facing report schema **before**
the downstream biology modules are finished.

The purpose is to make every later change populate one known report rather than
building a disconnected reporting layer at the end.

## Stable report workspace

Every study run now contains:

```text
report/
  report_manifest.json
  report_status.csv
  REPORT_SKELETON.md
  README.md
  figures/
  tables/
  cells/
  samples/
  conditions/
  appendix/
```

## Final report sections

1. Study Overview
2. Data Quality & Quantification
3. Cell Identification & Review
4. Cellular Elemental Composition
5. Element Localization
6. Chemical Domains / Granules / Blobs
7. Element ↔ Element Relationships
8. Cell Phenotypes
9. Condition / Mutant / Treatment Comparison
10. Organelle-Likeness
11. Biological Findings
12. Methods / Provenance / QC Appendix

## Scientific reporting policy

The report contract explicitly requires:

- artifact-grounded claims only;
- reviewed/canonical cells for final cell-level biology;
- no modification of raw HDF5;
- no invented physical units;
- 2D XRF volume claims only when clearly model-based;
- organelle-likeness rather than definitive organelle identity;
- no probability language for uncalibrated heuristic scores.

## Current status behavior

A current artifact can be useful without being accepted as final evidence.

For example:

- current study overview can be ready;
- current scan-level quantification can be partial;
- cell identification stays partial until review is complete;
- current pre-canonical element pairs remain partial;
- downstream composition/localization/domain/statistics sections remain planned.

This prevents the final report from silently using provisional TFY-cell
statistics as canonical biology.

## Planned final deliverables

The contract reserves:

- `SOURDOUGH_Report.html`
- `SOURDOUGH_Report.pdf`
- `SOURDOUGH_Data.xlsx`
- `SOURDOUGH_Data/`
- publication-ready figures

Actual report assembly comes later, after the scientific tables are populated.
