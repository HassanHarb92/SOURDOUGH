# SOURDOUGH Automated Study Analysis

The normal SOURDOUGH workflow is becoming:

```text
study directory
    -> scan discovery
    -> validated MAPS quantification
    -> cell detection
    -> cell × element dataset
    -> element-pair relationships
    -> QC / provenance
    -> comparison-ready study tables
```

Change 11A establishes the persistent study schema and execution engine.

## Current automatic products

For every included scan:

- provisional TFY cell mask;
- detected/full/cropped cell census;
- automatic MAPS concentration for every quantifiable element;
- Fitted, NNLS, and ROI kept separate;
- US_IC and DS_IC kept separate;
- whole-scan elemental concentration statistics;
- every detected cell × every quantified element statistics;
- within-cell element-pair Pearson correlations;
- compressed concentration arrays by method/reference;
- scan provenance and objective QC flags.

At study level:

- `study_manifest.csv`
- `study_summary.json`
- `study_provenance.json`
- `scans.csv`
- `cells.csv`
- `cell_elements.csv`
- `scan_elements.csv`
- `element_pairs.csv`
- `qc_flags.csv`
- `failures.csv`
- Parquet mirrors when a Parquet engine is installed.

## Critical scientific status

Cell masks in Change 11A are **provisional TFY-based masks**.

They are not the final canonical cells. The planned multichannel cell-consensus
engine will replace this segmentation while preserving the study data model.

Channel-specific artifact correction is also not yet applied.

No pixel sum is interpreted as total elemental mass because the physical X/Y
pixel-area unit has not yet been verified.

## Run

```bash
make analyze-study
```

or:

```bash
PYTHONPATH=$PWD/src python scripts/analyze_study.py /path/to/study/folder
```

Optional metadata CSV can contain:

```text
scan,sample_id,condition,strain_or_mutant,treatment,biological_replicate,technical_replicate,include,notes
```

## Product philosophy

Automated Study Analysis is the default scientific workflow.

The existing interactive Explorer remains the expert/manual layer for inspecting
maps, spectra, QC, normalization, gradients, Hessians, 2.5D models, calibration,
and individual quantitative products.

## Change 12 — artifact / contamination screening

Automated study runs include a non-destructive artifact review layer.
Candidates are retained and flagged using within-channel extremeness, local
spike behavior, cross-channel support, TFY support, provisional-cell overlap,
size, and edge context.

The output is `artifact_candidates.csv` plus per-scan masks and provenance.
Suspicion scores are heuristic review priorities, not probabilities.

No concentration maps are modified. The current screening product is explicitly
Fitted / US_IC; this is a screening choice and does not make that product the
scientifically preferred quantitative result.
