# Change 15 — Canonical Cell Dataset

Change 15 converts resolved human review into the authoritative cell population
used by downstream SOURDOUGH biology.

## Finalization gate

- `accept` -> canonical cell
- `reject` -> excluded but preserved in lineage
- `pending` -> blocks finalization
- `ambiguous` -> blocks finalization

## Outputs

```text
canonical_cells.csv
canonical_cell_lineage.csv
canonical_cell_masks.npz
canonicalization_summary.json
scans/<SCAN>/canonical_cell_masks.npz
```

## IDs and lineage

Accepted cells are sorted deterministically by scan and source candidate ID and
assigned `CELL_000001`, `CELL_000002`, and so on.

Every canonical cell traces back to:

```text
scan
-> consensus_candidate_id
-> Change-13A candidate mask
-> human review decision
-> canonical mask label
-> CELL_###### ID
```

Change 15 does not edit masks.

## Border cells

An accepted border-touching object may be a real canonical cell, but it gets:

```text
default_complete_cell_population = False
```

so later whole-cell statistics can exclude incomplete cells by default.

## Staleness protection

The canonicalization summary hashes the consensus/support/review tables.
Changing review or consensus after finalization makes the canonical snapshot
stale until it is re-finalized.

## Report contract

Cell Identification & Review becomes `READY` only after fresh canonicalization.

Changes 16+ must use canonical cells for final cell-level biology.
