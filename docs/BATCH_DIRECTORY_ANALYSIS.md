# SOURDOUGH Directory-Scale Batch Analysis

The Batch Study analyzes every `.h5` / `.hdf5` MAPS file in one directory.

## Saved sample picture

Each scan receives `sample_snapshot.png` containing:

1. raw Total_Fluorescence_Yield;
2. a robust TFY display view;
3. detected cell IDs;
4. full versus cropped cell status.

This is the audit image showing what SOURDOUGH treated as the sample.

## Cell census

Three counts remain distinct:

- **Detected cells**: every TFY segmentation candidate.
- **Cropped cells excluded**: candidates touching any image boundary.
- **Full cells analyzed**: complete cells entering quantitative analysis.

Cropped cells are still saved and listed, but do not enter full-cell chemistry or
gradient/Hessian statistics.

## Per-scan artifacts

Each scan directory contains:

- `sample_snapshot.png`
- `sample_manifest.json`
- `cells.csv`
- `whole_scan_channel_stats.csv`
- `full_cell_chemistry.csv`
- `full_cell_feature_stats.csv`
- `full_cells/cell_###.png`
- `cropped_cells/cell_###.png`
- optional `centroid_spectra/cell_###_centroid_spectrum.csv`

## Quantitative outputs

Whole-scan and full-cell channel statistics include finite count/fraction, negative and
zero fractions, min/max, mean, median, standard deviation, MAD, percentiles, and sum.

Selected structural channels receive gradient magnitude/orientation and Hessian curvedness,
shape index, ridge/valley, and blob summaries inside each full-cell mask.

## Directory-level outputs

Each run contains:

- `batch_summary.json`
- `run_config.json`
- `scans.csv`
- `cells.csv`
- `full_cell_chemistry.csv`
- `whole_scan_channel_stats.csv`
- `full_cell_feature_stats.csv`
- `failures.csv`
- `index.html`
- one subdirectory per scan

## Command line

```bash
PYTHONPATH=$PWD/src python scripts/analyze_directory.py img.dat
```

## Important limitation

Automatic TFY segmentation is a candidate-cell detector, not ground truth. The saved
sample snapshots and cell crops are mandatory audit artifacts and should be reviewed before
automatic counts are treated as final biological counts.
