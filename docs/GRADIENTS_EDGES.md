# Gradients and Edges

Change 07 adds first derivatives and edge structure.

Coordinate-aware derivatives use the stored X/Y coordinate values. Strictly monotonic axes use coordinate finite differences; repeated or locally nonmonotonic positions use local coordinate least-squares slopes without inventing a uniform grid:

- Ix = dI/dx
- Iy = dI/dy
- gradient magnitude
- gradient orientation
- directional derivative

Because coordinate units remain unverified, units are labeled `source-unit per stored-coordinate-unit`.

Classic Sobel, Scharr, and Prewitt remain pixel-grid operators and are labeled `source-unit per pixel`.

Optional Gaussian pre-smoothing uses sigma in pixels.

Binary edge masks use an explicit percentile threshold.

Second derivatives are intentionally deferred to Change 08.
