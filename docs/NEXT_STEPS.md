# Immediate next steps

1. Create the environment.
2. Run the HDF5 inventory across all 12 files.
3. Compare dataset paths/shapes/attributes across scans.
4. Identify:
   - elemental maps
   - element/channel labels
   - scaler channels
   - scan coordinates
   - dwell-time metadata
   - beamline metadata
   - fitted-vs-raw spectra/products
   - masks or acquisition QC fields
5. Only after that, design the canonical XRF data model.
6. Then begin actual image mathematics.

## Cross-file schema step

Run:

```bash
make schema
```

Then inspect and share:

- `analysis/schema/schema_report.md`
- `analysis/schema/three_d_readiness.json`
- `analysis/schema/dataset_matrix.csv`

The report also evaluates whether native 3D, serial-section stacking, tomography, or only
2.5D visualization is currently supported by the inventory metadata.

## Semantic metadata step

Run:

```bash
make semantic
```

Then review/share:

- `analysis/semantic/semantic_report.md`
- `analysis/semantic/semantic_summary.json`
- `analysis/semantic/scan_geometry_theta.csv`

The next change will use those results to define the canonical XRF scan object and explicit
scientific roles for maps, spectra, coordinates, scalers, and acquisition metadata.

## After Change 03

Build Change 04, the first real XRF Explorer, using only the canonical `XRFScan` API:

- scan selection;
- Fitted / NNLS / ROI selection;
- analyzed-channel map display;
- exact X/Y coordinates;
- channel role and unit;
- pixel spectrum inspection;
- scaler/QC-map display;
- side-by-side channel comparison;
- no hidden scientific normalization.

## After the interactive explorer

Change 04 establishes the observation UI.

Next scientific milestone:

### Change 05 — QC and normalization framework

Build explicit, provenance-tracked operations for:

- raw vs normalized views
- livetime/realtime diagnostics
- ICR/OCR/dead-time diagnostics
- scaler normalization candidates such as US_IC / DS_IC when scientifically justified
- finite/zero/negative-value masks
- low-count and saturation diagnostics
- normalization sensitivity checks
- explicit provenance describing every correction

No correction should become the hidden default.

## After QC + normalization

Change 05 establishes explicit QC/masking/normalization primitives.

Next:

### Change 06 — intensity and contrast analysis

Add independently tested image transforms and local contrast descriptors, while preserving the
distinction between:

- display transforms;
- scientifically derived feature maps;
- normalization/correction products.

Candidate families include robust scaling, asinh, local background subtraction, local contrast,
CLAHE-style display views, local mean/std/CV/MAD, and Poisson-aware variance stabilization
only when the source quantity is appropriate for a count-data model.

## After Change 06

Next: **Change 07 — gradients and edges**

Planned operators:

- Gaussian Ix / Iy
- gradient magnitude
- gradient orientation
- Sobel
- Scharr
- directional derivatives
- edge masks
- optional LoG as a clearly classified second-derivative preview

The implementation should use stored X/Y spacing numerically where appropriate without
inventing a physical coordinate unit.

## After Change 07

Next: **Change 08 — Hessian and curvature**

Planned: Ixx, Iyy, Ixy, Laplacian, Hessian determinant/trace/eigenvalues,
principal curvature direction, ridges, valleys, blobs, and multiscale sigma controls.

## After Change 08

Next: **Change 09 — Structure tensor**

Planned maps:

- Jxx / Jyy / Jxy
- structure-tensor eigenvalues
- dominant orientation
- coherence
- anisotropy
- corner/edge organization
- tensor ellipses / orientation glyphs

This complements the Hessian: the Hessian describes second-order curvature while the
structure tensor describes local organization of first-order gradients.

## After Change 08.5

Return to the core roadmap:

### Change 09 — Structure tensor

Then continue with texture, multiscale analysis, cross-element chemistry, spatial statistics,
multivariate analysis, and cell segmentation.

After segmentation, revisit the 2.5D landscape as an **individual-cell explorer** where each
yeast cell becomes a separately selectable chemical object.

## After Change 08.6

The 3D viewer now has a measured cell footprint and an explicit inferred-depth model.

Important future upgrades:

1. validate/tune TFY segmentation against manually reviewed cells;
2. establish the physical X/Y coordinate unit;
3. add structure-tensor orientation/coherence;
4. add texture and multiscale features;
5. use chemical/morphological evidence jointly for higher-confidence cell boundaries;
6. if real Z-resolved data become available, replace inferred depth with measured 3D geometry.

## After Change 08.9

The project now has a directory-scale study layer. Before automatic cell counts are treated
as biological ground truth, review the saved TFY sample snapshots and tune segmentation
settings across representative scans.

The next mathematical roadmap item remains **Change 09: Structure Tensor**.

## After Change 09

Inspect `analysis/quantification/change09_validation.json` to decode the stored MAPS
calibration metadata across all representative scans.

### Change 10 — MAPS calibration unlock + automatic concentration products

Once the calibration-factor semantics are verified, connect stored MAPS calibration data
directly to per-pixel and per-cell areal-density products.

Then proceed to hot-pixel / artifact correction and stronger segmentation review.

## Change 10 COMPLETE — MAPS quantitative concentration

Automatic MAPS areal-density products are validated across the representative
dataset for Fitted, NNLS, and ROI using both US_IC and DS_IC references.

Next development path:

- Change 11 — Study Ingest
- Change 12 — Artifact / contamination layer
- Change 13 — Multichannel Cell Consensus
