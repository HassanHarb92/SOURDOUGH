# Hessian and Curvature

Change 08 adds coordinate-aware second derivatives.

For an XRF map I(x,y):

```text
H = [[Ixx, Ixy],
     [Ixy, Iyy]]
```

## Coordinate treatment

Ixx and Iyy are estimated using local quadratic least-squares fits against the actual stored
coordinate values.

This supports:

- nonuniform coordinate spacing
- repeated coordinate positions
- local window expansion when duplicate positions reduce polynomial rank

No artificial coordinate jitter and no invented uniform grid are used.

## Mixed derivative

Two independent paths are calculated:

```text
Ixy_path = d(Iy)/dx
Iyx_path = d(Ix)/dy
```

The Hessian uses:

```text
Ixy = 0.5 * (Ixy_path + Iyx_path)
```

and retains:

```text
abs(Ixy_path - Iyx_path)
```

as a numerical QC map.

## Core maps

- Ixx
- Iyy
- symmetric Ixy
- mixed-derivative disagreement
- trace
- determinant
- Laplacian
- lambda_min
- lambda_max
- principal eigenvector direction
- curvedness
- shape index

## Transparent ridge / valley / blob responses

Change 08 uses simple single-scale formulas:

```text
bright ridge = max(-lambda_min, 0)
dark valley  = max(lambda_max, 0)

blob strength = sqrt(max(det(H), 0))
bright blob uses trace(H) < 0
dark blob uses trace(H) > 0
```

These are intentionally not described as Frangi vesselness or multiscale blob detection.

## Units

Second derivatives are labeled:

```text
source-unit per stored-coordinate-unit^2
```

because the numeric coordinate values are known but their physical unit has not been verified.

## Next step

Change 09 will build the structure tensor, which complements the Hessian by describing
first-derivative orientation coherence and anisotropy rather than second-order curvature.
