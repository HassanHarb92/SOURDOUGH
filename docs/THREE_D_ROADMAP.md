# Long-term 3D chemical-cell explorer

The codebase should remain 3D-ready from the beginning, but it must distinguish true
volumetric reconstruction from purely visual 2.5D rendering.

## Reconstruction modes

### 1. Native 3D volume

Use when a dataset can be proven to represent physical `(z, y, x)` or an equivalent volume,
with spatial calibration.

Potential future views:

- orthogonal XY / XZ / YZ slice navigation
- arbitrary clipping planes
- multi-element volume rendering
- isosurfaces for selected elemental concentrations
- interior fly-through / cutaway views
- linked 2D and 3D cursors
- voxel-level elemental signatures

### 2. Registered serial sections

Use only when separate 2D scans are physically ordered sections of the same specimen/cell or
volume and z spacing/order is known.

Required work:

- section identity and z ordering
- physical x/y calibration
- rigid/affine/nonrigid registration as justified
- missing-section handling
- resampling onto a common physical grid
- registration uncertainty

### 3. XRF tomography

Use when projection-angle metadata and projection data are present.

Required work:

- projection geometry
- center-of-rotation handling
- reconstruction method
- attenuation/self-absorption considerations where scientifically relevant
- reconstructed-volume QC

### 4. 2.5D intensity surface

Always possible for a 2D scalar map:

`z_display = f(intensity)`

This is a visualization only. It is not a reconstructed biological cell volume and must be
labeled accordingly.

## Future interactive experience

A mature explorer could support:

- mouse orbit/pan/zoom
- clipping box and arbitrary slice planes
- transparent volume rendering
- element toggles and opacity sliders
- per-element transfer functions
- element-ratio volumes
- segmented cell/cytoplasm/organelle-candidate meshes
- click-to-inspect voxel chemistry
- ROI statistics in physical 3D
- linked histogram/scatter/PCA views
- surface and interior measurements
- 3D gradients, Hessians, structure tensors, blob/ridge/sheet responses
- export of meshes, volumes, screenshots, and quantitative tables

## Scientific rule

Array rank does not establish physical dimensionality.

A rank-3 XRF dataset may be:

- z × y × x
- channels × y × x
- y × x × energy
- detector × y × x
- another acquisition cube

Physical axes and metadata must establish the interpretation before a dataset is called a
volume.
