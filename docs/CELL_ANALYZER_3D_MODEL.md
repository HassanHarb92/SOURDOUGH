# TFY Cell Analyzer and Inferred 3D Cell Model

## Cell detection

`Total_Fluorescence_Yield` is used as the morphology/segmentation source.

The pipeline supports:

- Otsu, Yen, Li, or percentile thresholding
- Gaussian pre-smoothing
- minimum-area filtering
- morphological closing
- hole filling
- optional watershed splitting of touching cell candidates

## Cropped cells

Any component touching the top, bottom, left, or right image boundary is flagged as cropped.

The application reports:

- detected cells
- complete cells
- cropped cells

Cropped cells remain visible/reportable but are excluded from complete-cell quantitative
analysis by default.

## Morphometry

Per-cell measurements include:

- pixel area
- approximate area in stored-coordinate units squared
- centroid X/Y
- equivalent diameter
- major/minor axes
- orientation
- eccentricity
- pixel circularity
- solidity
- TFY intensity statistics

The coordinate unit is not yet physically verified, so values are not labeled as micrometers.

## Inferred 3D envelope

The 2D TFY segmentation is a measured footprint.

The cell depth is inferred using a mask-conforming dome generated from distance to the measured
cell boundary. The maximum half-depth is tied to the measured minor-axis radius through an
explicit user-controlled ratio.

Therefore:

```text
X/Y footprint = measured
Z/depth        = inferred
```

This is not an experimental 3D reconstruction.

## Going inside the cell

The user can:

- make the cell envelope translucent
- show the lower envelope
- cut away the X or Y side of the cell
- move an internal Z slice through the inferred volume

## Projection-conserving interior XRF model

A measured XRF map is 2D.

For an optional internal visualization, Change 08.6 uses:

```text
modeled_density(x,y) = measured_2D_XRF(x,y) / inferred_thickness(x,y)
```

and assumes that density is uniform along inferred Z at each X/Y position.

This construction is projection-conserving: integrating the modeled density through the
inferred thickness recovers the original 2D XRF value.

It does **not** prove that the real cell has uniform concentration along Z. It is an explicit
model used to inspect one possible interior consistent with the measured projection.

True internal 3D XRF localization requires experimental Z information such as serial sections,
depth-resolved acquisition, or tomography.
