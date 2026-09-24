# Change 14A — Persistent Cell Review

Change 14A adds the persistent human-review boundary between automated
multichannel consensus and the future canonical cell dataset.

## Persistent artifact

Each study run gains:

```text
cell_review.csv
```

One row exists for every Change-13A consensus candidate.

Stored fields include:

- scan;
- consensus candidate ID;
- source automated class and score;
- review decision;
- reason;
- reviewer notes;
- optional reviewer name;
- UTC review timestamp;
- mask policy;
- explicit `canonical_cell = False`.

## Review decisions

- `pending`
- `accept`
- `reject`
- `ambiguous`

Review is non-destructive. Change 14A does not modify the saved Change-13A
consensus masks or scores.

An accepted review means only:

> this consensus candidate is approved to proceed toward Change 15.

It does not yet mean the cell is canonical.

## UI

`RESULTS -> Cell Consensus` now shows:

- review progress;
- accepted/rejected/ambiguous/pending counts;
- persistent review status in the candidate table;
- a candidate-specific review form;
- reason codes;
- free-text notes;
- optional reviewer name.

Saving a review atomically updates `cell_review.csv`.

## Next boundary

Change 15 will consume reviewed candidates and create the canonical cell dataset
and masks.

## Change 15.1 — Accept all pending

The Cell Consensus review workspace includes a guarded bulk action:

**Accept all pending**

Behavior:

- changes only rows currently marked `pending`;
- preserves existing `accept`, `reject`, and `ambiguous` decisions;
- requires an explicit confirmation checkbox;
- optionally records reviewer name and a bulk note;
- records `review_reason = bulk_accept_pending`;
- records `review_source = manual_ui_bulk_accept_pending`;
- writes one atomic update to `cell_review.csv`;
- invalidates any previously finalized canonical snapshot.

This is a review convenience action only. It does not itself canonicalize cells.
Run Change-15 canonicalization afterward.
