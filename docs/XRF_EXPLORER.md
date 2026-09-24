# XRF Explorer

Change 04 is the first interactive scientific interface built on the canonical `XRFScan`
loader.

## Architecture boundary

The Streamlit application does not know MAPS/HDF5 storage paths.

It uses only canonical concepts:

```python
scan.map(...)
scan.spectrum(...)
scan.scaler(...)
scan.geometry
scan.channels(...)
scan.extra_pvs()
```

This is enforced by a test that rejects direct `/MAPS/` references or `h5py` imports in the UI.

## Tabs

### Map Explorer

- scan selection
- Fitted / NNLS / ROI selection
- channel selection with role and unit
- exact stored X/Y coordinates
- raw map statistics
- raw-value histogram
- display-only transforms:
  - Raw
  - Percentile stretch
  - Signed log1p
  - Gamma
- interactive 2.5D surface preview

The 2.5D preview is explicitly a display surface. Its height is not physical Z depth.

### Spectrum Inspector

- choose a context channel
- choose X/Y pixel indices
- see exact stored X/Y coordinate values
- load one pixel spectrum from the 2048-channel MCA cube
- display that spectrum without loading the entire cube

### Scaler & QC

- primary 46-channel scaler family
- legacy 17-channel scaler family
- XRF diagnostic/scatter channels
- no hidden normalization

### Compare

- up to three channels within one analysis method
- or one channel across Fitted / NNLS / ROI
- optional shared raw color scale

### Acquisition

- raster dimensions
- geometry
- coordinate-step diagnostics
- energy axis
- channel catalog
- Extra PV table
- canonical scan summary

## Display transform rule

Display transforms operate on copies of the selected array and never overwrite or cache a
modified scientific source array as though it were raw data.

`Signed log1p` is used instead of blindly clipping negative fitted values, because negative
fit outputs can carry information about the fitting process.

## What Change 04 intentionally does not do

- no automatic scaler normalization
- no dead-time correction
- no background correction
- no element-ratio maps
- no gradients/Hessians/textures
- no cell segmentation
- no inferential statistics
- no claim of true 3D reconstruction

Those capabilities are added later as independently tested scientific modules.

## Original/raw reference view

Change 04.1 adds a persistent reference view so display transforms can always be judged
against the unmodified selected channel.

When the current display mode is Percentile stretch, Signed log1p, or Gamma, the Map Explorer
can show:

```text
Original / raw     |     Current display
```

side by side.

An expandable **Scan overview / original appearance** section also uses
`Total_Fluorescence_Yield` as a broad XRF scan-intensity reference when that channel is
available. This is an XRF-derived map and must not be interpreted as an optical microscopy
photograph of the yeast.
