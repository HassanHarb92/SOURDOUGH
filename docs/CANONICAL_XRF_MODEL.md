# Canonical XRF Data Model

Change 03 introduces the scientific IO boundary for all later analysis.

## Core API

```python
from yeast_xrf.io.xrf_scan import XRFScan

with XRFScan.open("img.dat/bnp_fly0006.mda.h5") as scan:
    fe = scan.map("Fe", method="Fitted")
    zn = scan.map("Zn", method="NNLS")
    spectrum = scan.spectrum(y=40, x=70)
    us_ic = scan.scaler("US_IC")
```

## Lazy-read rules

Construction does not load the large image or spectral cubes.

- `scan.map("Fe")` reads one `(Y, X)` analyzed channel.
- `scan.spectrum(y=..., x=...)` reads one 2048-value energy vector.
- `scan.scaler("US_IC")` reads one `(Y, X)` scaler channel.
- `scan.map_stack([...])` loads only explicitly requested channels.
- full-file SHA-256 is only calculated by explicit `scan.strong_identity()`.

Whole-cube PCA/NMF and other expensive analyses will use explicit chunked workflows later.

## Spatial model

Current scans expose `("y", "x")`. The exact stored X/Y coordinate arrays are retained rather
than replaced with a synthetic grid. Coordinate units remain unknown until established from
the acquisition metadata.

## Analysis products

Fitted, NNLS, and ROI remain independently accessible. Change 03 does not declare a preferred
method.

## Channel roles

Channels are annotated descriptively as elemental/line-like, fluorescence, scatter,
fit-diagnostic, or auxiliary. These roles do not alter the data or imply biology.

## Spectral model

`/MAPS/Spectra/mca_arr` is treated as `(energy, y, x)`, not a spatial 3D volume.

## 3D boundary

All 12 current theta values are 0.0, so the current set is not an angular tomography series.
The higher-level interface remains dimension-aware so future volumetric data can add a
`("z", "y", "x")` adapter without redefining downstream scientific code.

## Validation

```bash
make validate-scans
```

This checks map/channel counts, X/Y geometry, spectral dimensions, and scaler dimensions for
every current HDF5 file.
