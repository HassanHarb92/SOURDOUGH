# Intensity and Contrast Analysis

Change 06 adds the first reusable derived-feature layer.

## Three separate representations

1. source scientific map
2. display-only transform
3. derived scientific feature

CLAHE, percentile stretch, gamma, and similar rendering changes stay display-only.

Derived maps include:

- asinh intensity
- robust z-score
- local mean
- local standard deviation
- local coefficient of variation
- local median
- local MAD
- local IQR
- local standardized contrast
- Gaussian background
- background-subtracted intensity

Unit-preserving features keep the source unit. CV, local standardized contrast, asinh, and
robust z-score are dimensionless.

Background subtraction may legitimately produce negative values.

Neighborhood windows and Gaussian sigma are expressed in pixels only. The software does not
invent micrometers or any other physical coordinate unit before the axis unit is verified.
