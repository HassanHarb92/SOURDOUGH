# Change 13A — Multichannel Cell Consensus

SOURDOUGH now builds an automated cell-consensus layer from multiple
cell-support channels instead of using TFY alone.

Default support ensemble:

- Total_Fluorescence_Yield
- P
- S
- K

Localized channels such as Zn, Fe, Ca, Cu and Mn are not required to outline a
whole cell.

Outputs:

- `cell_consensus.csv`
- `cell_channel_support.csv`
- `cell_consensus_scans.csv`
- per scan `cell_masks_multichannel_consensus.npz`
- per scan `cell_consensus.json`

Classes:

- `high_confidence_cell`
- `accepted_cell`
- `ambiguous_cell_candidate`
- `rejected_channel_specific_object`

Change 12 high-priority artifact masks are used as context/penalty only. They do
not automatically delete candidates.

Change 13A is automated consensus, not final human-reviewed canonical truth:

- `review_status = pending`
- `canonical_cell = False`

Change 14 will provide Cell Review. Change 15 will finalize the canonical cell
dataset.

## Change 13B — consensus visualization in RESULTS

The RESULTS workspace now includes a **Cell Consensus** tab.

It provides:

- study/run consensus counts;
- scan and consensus-class filtering;
- candidate table;
- candidate-level consensus score and morphology;
- artifact-overlap context;
- TFY/P/S/K support fractions;
- support-fraction bar chart;
- saved support-count map;
- all-candidate labels;
- automated-accepted labels;
- selected-candidate mask;
- per-channel support masks.

The visualization uses saved Change-13A artifacts and does not read MAPS HDF5
directly.

Automated acceptance remains explicitly review-pending. Change 13B is an
inspection interface only; it does not alter segmentation or consensus scores.
