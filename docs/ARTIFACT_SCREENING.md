# Change 12 — Artifact / Contamination Candidate Screening

SOURDOUGH now runs an explicit review layer between quantitative XRF products
and the future multichannel-consensus cell detector.

## Scientific policy

**Retain + flag. Never silently erase.**

A suspicious feature is not automatically an artifact. Strongly localized Zn,
Fe, Ca, or another elemental feature may be real cellular biology.

Change 12 reports a heuristic review priority and its evidence. The suspicion
score is **not a calibrated probability**. No concentration map is modified.

## Evidence recorded

- source elemental channel;
- object area and pixel-space centroid/bounding box;
- robust within-channel signal extremeness;
- local median-residual spike extremeness;
- support in other elemental channels;
- TFY support fraction;
- overlap with provisional TFY cell masks;
- scan-edge contact;
- evidence/reason tags;
- review priority;
- explicit `retain_and_flag` action.

Candidate classes include:

- `isolated_channel_spike_candidate`
- `channel_specific_external_object_candidate`
- `cell_associated_channel_specific_feature`
- `cross_channel_supported_feature`
- `review_candidate`

They are screening descriptions, not definitive biological or detector labels.

## Screening product

The current automated screening product is:

```text
Fitted / US_IC concentration maps
```

This is a screening choice only. It does not declare Fitted or US_IC
scientifically preferred for biological conclusions. The complete Fitted,
NNLS, ROI, US_IC, and DS_IC quantitative products remain available.

## Outputs

Study level:

- `artifact_candidates.csv`
- `artifact_candidates.parquet` when available

Per scan:

- `artifact_candidate_masks.npz`
- `artifact_screening.json`

## Relationship to Change 13

Change 13 will use this review layer as context while building multichannel
consensus cells. Candidates are not automatically removed from raw or
quantitative images.
