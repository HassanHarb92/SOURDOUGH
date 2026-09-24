# Architecture

## Principle 1: observe the HDF5 layout before interpreting it

The beamline/source schema is not yet assumed. `io/h5_inventory.py` is intentionally generic.

## Principle 2: preserve raw information

Raw HDF5 files under `img.dat/` are read-only inputs. Derived data goes under `analysis/`,
`cache/`, or `reports/`.

## Principle 3: separate transform classes

Future transforms should explicitly declare whether they are:

- visualization transforms
- feature transforms
- representation transforms

## Principle 4: every derived artifact has provenance

At minimum:

- source file identity
- source HDF5 dataset path
- feature/transform key
- parameters
- dimensionality
- pixel/voxel spacing if available
- code version later
- creation time later

## Principle 5: XRF is multichannel chemistry, not RGB

Each elemental signal can be analyzed spatially on its own and chemically in relation to other
elements. Cross-element maps and multivariate models should be first-class analyses.
