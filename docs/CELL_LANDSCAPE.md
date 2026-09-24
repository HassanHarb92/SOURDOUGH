# Interactive 2.5D Yeast Cell Chemical Landscape

Change 08.5 turns the 2D XRF raster into an interactive scientific visualization object.

## What it is

A 2.5D landscape:

```text
z_display(x,y) = f(XRF intensity at x,y)
```

The user may choose one XRF channel for display height and another chemical or differential
geometry quantity for surface color.

The 3D view supports rotation, zoom, perspective, lighting, contours, and structural overlays.

## What it is not

It is **not** a reconstructed physical 3D yeast cell.

The current 12 scans have no validated physical Z stack or angular tomography series. All
current theta values are identical.

The Z axis therefore says:

```text
Derived/display height
```

## Height geometry

Base height is generated from a robust percentile scaling of the chosen XRF channel.

Vertical exaggeration changes display geometry only.

Optional curvature emphasis uses Hessian curvedness:

```text
z_display <- z_display * (1 + alpha * normalized_curvedness)
```

This is explicitly a visualization effect.

## Surface color

The surface may be colored by:

- another chemical/XRF channel
- gradient magnitude
- gradient orientation
- Hessian curvedness
- shape index
- bright-ridge response
- dark-valley response
- bright-blob response
- dark-blob response

## Structural overlays

The 3D landscape can overlay high-response points from:

- bright ridges
- bright blobs

Thresholds are explicit percentiles.

## Background visibility mask

A low-signal mask can hide background from the rendered surface.

This is a **visualization mask, not cell segmentation**.

Cell segmentation remains a later scientific task.

## Point inspector

At a selected pixel the Explorer reports:

- raw XRF height-channel value
- derived display height
- gradient magnitude and direction
- Ixx / Ixy / Iyy
- Hessian eigenvalues
- curvedness
- shape index
- blob response
- deterministic Hessian-sign classification
- display-surface normal

The displayed normal is a property of the derived 2.5D geometry, not a measured physical cell
surface normal.

## Future evolution

Once individual yeast-cell segmentation is available, this view can evolve from one raster
landscape into isolated per-cell 2.5D objects with chemical overlays and per-cell statistics.
