# Semantic XRF Probe

This step reads a small allowlisted subset of metadata/vector datasets from the structurally
consistent MAPS HDF5 files.

## Safety model

- raw `.h5` files are opened read-only;
- nothing is written under `img.dat/`;
- only explicitly allowlisted small datasets are read;
- a small-data read above 20,000 elements is refused;
- large XRF/scaler cubes are inspected by metadata only.

## Questions answered

- Fitted / NNLS / ROI channel names and units
- channel-list consistency across scans
- raster dimensions and x/y coordinate consistency
- coordinate ranges and median steps
- actual `theta` values and angular span
- 46-channel and 17-channel scaler identities
- energy range, spacing, and calibration coefficients
- quantification-standard metadata
- Extra PVs related to angle/rotation/tomography or z/depth/sections

## Interpretation boundary

A varying `theta` series is not proof of tomography. A real tomographic interpretation also
requires evidence that the scans are projections of the same specimen/volume under known
geometry with suitable angular coverage. Likewise, z/depth-like metadata does not establish a
serial-section volume by itself.
