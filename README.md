# Yeast XRF Analysis

Working title only. The project will get a proper name later.

## Goal

Build a very extensive analysis environment for XRF images of yeast stored in HDF5 files.

The project starts **structure-first**. We will not guess that any particular HDF5 dataset is
an elemental map until the files have been inventoried and the hierarchy/metadata have been
studied.

The raw HDF5 files remain untouched under:

```text
img.dat/
```

## Current source files

- bnp_fly0006.mda.h5
- bnp_fly0019.mda.h5
- bnp_fly0025.mda.h5
- bnp_fly0050.mda.h5
- bnp_fly0053.mda.h5
- bnp_fly0057.mda.h5
- bnp_fly0072.mda.h5
- bnp_fly0076.mda.h5
- bnp_fly0079.mda.h5
- bnp_fly0095.mda.h5
- bnp_fly0097.mda.h5
- bnp_fly0105.mda.h5

## Project layout

```text
Yeast/
├── img.dat/                  # raw HDF5 inputs; never modified
├── src/yeast_xrf/
│   ├── io/                   # discovery + HDF5 inventory
│   ├── features/             # feature specs/registry/artifact keys
│   ├── analysis/             # future scientific transforms
│   ├── model.py
│   ├── paths.py
│   ├── provenance.py
│   └── cli.py
├── app/                      # Streamlit explorer
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── tests/
├── analysis/
│   ├── inventory/
│   ├── features/
│   ├── figures/
│   └── tables/
├── cache/
└── reports/
```

## First milestone

Inventory every HDF5 file:

```bash
PYTHONPATH=$PWD/src python scripts/inspect_all_h5.py
```

or:

```bash
yeast-xrf inventory-all
```

This records:

- all groups
- all datasets
- shapes
- dimensionality
- dtypes
- chunking
- compression
- group/dataset attributes
- estimated uncompressed sizes

It deliberately does **not** load large image arrays or assign scientific meaning yet.

## Environment

With Miniforge/Mamba:

```bash
mamba env create -f envs/environment-mac.yml
conda activate yeast-xrf
python -m pip install -e .
```

Or in an existing Python 3.11+ environment:

```bash
python -m pip install -e ".[dev]"
```

## Commands

```bash
make test
make inventory
make app
```

## Analysis roadmap

The intended system will eventually include:

1. HDF5 forensic inventory
2. Canonical XRF data model
3. Element-map discovery and metadata normalization
4. Raw-count / scaler / dwell-time / background QC
5. Intensity and contrast transforms
6. Thresholds and masks
7. Gradients and edge fields
8. Hessian / curvature / ridge / valley / blob maps
9. Structure tensors and orientation/coherence
10. Morphology and local statistics
11. Texture and frequency-domain features
12. Multiscale / scale-space analysis
13. Cross-element differences, ratios, normalized differences, and log-ratios
14. Local/global elemental co-localization
15. PCA / NMF / clustering / chemical domains
16. Anomaly/hotspot detection
17. Yeast-cell/object segmentation
18. Per-cell elemental chemistry and spatial organization
19. QC, uncertainty, and low-count diagnostics
20. Interactive explorer, exports, and reproducible reports

## Scientific rule

A derived map is an **observation tool**, not automatically a biological conclusion.

For example, an Fe-rich hotspot is evidence of localized Fe signal. Calling it a vacuole,
organelle, phenotype, or biological process requires additional evidence.

## Cross-file schema and 3D-readiness report

After generating `analysis/inventory/*.inventory.json`, compare all scans with:

```bash
make schema
```

Outputs:

```text
analysis/schema/schema_summary.json
analysis/schema/dataset_matrix.csv
analysis/schema/schema_report.md
analysis/schema/three_d_readiness.json
```

The 3D-readiness report is intentionally conservative. A rank-3 array is **not** automatically
called a 3D cell volume; it may instead represent channels, energy, detector axes, or another
acquisition cube. See `docs/THREE_D_ROADMAP.md`.

## Semantic XRF probe

After structural inventory/schema comparison:

```bash
make semantic
```

Outputs:

```text
analysis/semantic/semantic_report.md
analysis/semantic/semantic_summary.json
analysis/semantic/semantic_scans.json
analysis/semantic/scan_geometry_theta.csv
analysis/semantic/channel_catalog.csv
analysis/semantic/scaler_catalog.csv
analysis/semantic/extra_pvs.csv
```

The probe remains read-only and does not load the large elemental/spectral cubes.

## Canonical XRF loader

```python
from yeast_xrf.io.xrf_scan import XRFScan

with XRFScan.open("img.dat/bnp_fly0006.mda.h5") as scan:
    fe = scan.map("Fe", method="Fitted")
    zn = scan.map_xarray("Zn", method="NNLS")
    spectrum = scan.spectrum_xarray(y=50, x=50)
    us_ic = scan.scaler_xarray("US_IC")
```

Large cubes are not loaded during object construction. Exact X/Y coordinate vectors are
preserved and Fitted, NNLS, and ROI remain separate scientific products.

Validate all current scans with:

```bash
make validate-scans
```

See `docs/CANONICAL_XRF_MODEL.md`.

## Interactive XRF Explorer

Change 04 replaces the initial structure viewer with a scientific explorer built entirely on
the canonical `XRFScan` API.

Launch it with:

```bash
make app
```

The application includes:

- Map Explorer
- Spectrum Inspector
- Scaler & QC viewer
- side-by-side channel/method comparison
- acquisition metadata
- a clearly labeled 2.5D intensity-surface preview

Validate the exact map/spectrum/scaler operations used by the UI across all current scans:

```bash
make validate-explorer
```

See `docs/XRF_EXPLORER.md`.

## QC and explicit normalization

Change 05 adds provenance-tracked XRF QC and opt-in reference normalization.

Core example:

```python
from yeast_xrf.analysis.normalization import normalize_by_reference

result = normalize_by_reference(
    fe_map,
    us_ic,
    numerator_label="Fe",
    reference_label="US_IC",
    denominator_floor=chosen_floor,
)
```

The operation never overwrites the raw map.

The Explorer now includes a **QC & Normalize** tab with:

- nonfinite / negative / zero / low-signal QC
- high-value-tail candidate flags
- explicit US_IC / DS_IC / other scaler normalization
- absolute or positive-percentile denominator floors
- optional negative/zero numerator masks
- explicit scale factor
- raw vs normalized comparison
- normalization-mask accounting
- provenance JSON
- ICR/OCR and ELT/ERT diagnostic maps

Run real-data validation with:

```bash
make validate-qc
```

See `docs/QC_NORMALIZATION.md`.

## Intensity and contrast features

Change 06 adds a dedicated **Intensity & Contrast** Explorer tab plus reusable local-statistics,
background, asinh, and robust-standardization feature functions.

CLAHE is available as display-only enhancement.

Validate on all current scans:

```bash
make validate-intensity
```

See `docs/INTENSITY_CONTRAST.md`.

## Gradients and edges

Change 07 adds coordinate-aware Ix/Iy, magnitude, orientation, directional derivatives,
Sobel/Scharr/Prewitt, and percentile edge masks.

```bash
make validate-gradients
```

See `docs/GRADIENTS_EDGES.md`.

## Hessian and curvature

Change 08 adds coordinate-aware second derivatives and Hessian descriptors:

- Ixx / Iyy / symmetric Ixy
- mixed derivative QC
- Laplacian
- trace / determinant
- Hessian eigenvalues
- principal direction
- curvedness
- shape index
- bright-ridge / dark-valley responses
- bright/dark blob responses

```bash
make validate-hessian
```

See `docs/HESSIAN_CURVATURE.md`.

## Interactive 2.5D yeast-cell landscape

Change 08.5 adds an interactive Plotly chemical landscape combining:

- XRF intensity
- gradient structure
- Hessian curvature
- chemical color overlays
- ridge/blob overlays
- point-level differential-geometry inspection

The Z axis is visualization-derived and is never presented as a measured physical Z coordinate.

```bash
make validate-landscape
```

See `docs/CELL_LANDSCAPE.md`.

## TFY cell analyzer and inferred 3D cell model

Change 08.6 detects cell candidates from `Total_Fluorescence_Yield`, identifies border-cropped
cells, reports cell geometry, and constructs an interactive inferred cell envelope from the
measured 2D footprint.

The internal XRF mode is projection-conserving but model-based.

```bash
make validate-cells
```

See `docs/CELL_ANALYZER_3D_MODEL.md`.

## Batch Study

SOURDOUGH can analyze every MAPS HDF5 file in a directory and create a reproducible
sample/cell study with TFY snapshots, complete/cropped cell counts, cell crops, channel
statistics, per-cell chemistry, gradient/Hessian summaries, and optional centroid spectra.

```bash
make analyze-directory
```

See `docs/BATCH_DIRECTORY_ANALYSIS.md`.

## Beam normalization and quantitative XRF

Change 09 adds explicit US_IC / DS_IC beam normalization, transmission diagnostics,
read-only MAPS calibration inspection, and a guarded areal-density conversion path.

```bash
make validate-quantification
```

See `docs/BEAM_QUANTIFICATION.md`.

## Change 10 — Automatic MAPS concentration

SOURDOUGH now converts validated elemental MAPS products to areal density
(`µg/cm²`) using the stored MAPS calibration factor for the selected analysis
method and ion-chamber reference. Every automatic factor is cross-checked
against the exact stored calibration-curve label before use.

```bash
make validate-concentration
```
