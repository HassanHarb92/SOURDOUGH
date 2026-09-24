#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/hharb/Desktop/Projects/Yeast"
PKG="$ROOT/src/yeast_xrf"
ART="$PKG/analysis/artifacts.py"
ENGINE="$PKG/analysis/study_analysis.py"
CLI="$ROOT/scripts/analyze_study.py"
APP="$ROOT/app/streamlit_app.py"
VALIDATOR="$ROOT/scripts/validate_artifact_screening.py"
TEST="$ROOT/tests/test_artifact_screening.py"
ARCH_TEST="$ROOT/tests/test_artifact_results_ui.py"
DOC="$ROOT/docs/ARTIFACT_SCREENING.md"
STUDY_DOC="$ROOT/docs/STUDY_ANALYSIS.md"
MAKEFILE="$ROOT/Makefile"
ID="change_12_artifact_contamination_layer"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/.yeast-xrf-backups/${ID}_${STAMP}"

say(){ printf '\n[%s] %s\n' "$ID" "$*"; }
die(){ printf '\n[%s] ERROR: %s\n' "$ID" "$*" >&2; exit 1; }

for f in "$ENGINE" "$CLI" "$APP"; do
  [[ -f "$f" ]] || die "Required file missing: $f"
done

mkdir -p \
  "$BACKUP/src/yeast_xrf/analysis" \
  "$BACKUP/scripts" \
  "$BACKUP/app" \
  "$BACKUP/tests" \
  "$BACKUP/docs"

backup_or_note(){
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -p "$src" "$dst"
  else
    : > "${dst}.ABSENT"
  fi
}

backup_or_note "$ART" "$BACKUP/src/yeast_xrf/analysis/artifacts.py"
backup_or_note "$ENGINE" "$BACKUP/src/yeast_xrf/analysis/study_analysis.py"
backup_or_note "$CLI" "$BACKUP/scripts/analyze_study.py"
backup_or_note "$APP" "$BACKUP/app/streamlit_app.py"
backup_or_note "$VALIDATOR" "$BACKUP/scripts/validate_artifact_screening.py"
backup_or_note "$TEST" "$BACKUP/tests/test_artifact_screening.py"
backup_or_note "$ARCH_TEST" "$BACKUP/tests/test_artifact_results_ui.py"
backup_or_note "$DOC" "$BACKUP/docs/ARTIFACT_SCREENING.md"
backup_or_note "$STUDY_DOC" "$BACKUP/docs/STUDY_ANALYSIS.md"
cp -p "$MAKEFILE" "$BACKUP/Makefile"

APPLIED=0
restore_one(){
  local current="$1"
  local saved="$2"
  if [[ -f "$saved" ]]; then
    cp -p "$saved" "$current" || true
  elif [[ -f "${saved}.ABSENT" ]]; then
    rm -f "$current"
  fi
}
rollback(){
  rc=$?
  if [[ $rc -ne 0 && $APPLIED -eq 1 ]]; then
    say "Validation failed. Rolling back Change 12..."
    restore_one "$ART" "$BACKUP/src/yeast_xrf/analysis/artifacts.py"
    restore_one "$ENGINE" "$BACKUP/src/yeast_xrf/analysis/study_analysis.py"
    restore_one "$CLI" "$BACKUP/scripts/analyze_study.py"
    restore_one "$APP" "$BACKUP/app/streamlit_app.py"
    restore_one "$VALIDATOR" "$BACKUP/scripts/validate_artifact_screening.py"
    restore_one "$TEST" "$BACKUP/tests/test_artifact_screening.py"
    restore_one "$ARCH_TEST" "$BACKUP/tests/test_artifact_results_ui.py"
    restore_one "$DOC" "$BACKUP/docs/ARTIFACT_SCREENING.md"
    restore_one "$STUDY_DOC" "$BACKUP/docs/STUDY_ANALYSIS.md"
    cp -p "$BACKUP/Makefile" "$MAKEFILE" || true
    say "Rollback complete. Backup retained at: $BACKUP"
  fi
  exit "$rc"
}
trap rollback EXIT

say "Writing non-destructive artifact/contamination screening module..."
cat > "$ART" <<'PY'
'''Artifact / contamination candidate screening for SOURDOUGH.

This module is intentionally flag-first and non-destructive.

It identifies high-signal spatial objects that deserve review using robust
within-channel extremeness, local spike behavior, cross-channel support, TFY
support, provisional-cell overlap, size, and scan-edge context.

A candidate is NOT automatically an artifact or contaminant. Real cellular
chemistry may be strongly localized to one elemental channel. The module emits
review evidence and a heuristic suspicion score while leaving all source and
concentration arrays unchanged.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ArtifactScreenConfig:
    candidate_z: float = 6.0
    cross_channel_support_z: float = 3.0
    tfy_support_z: float = 2.0
    extreme_z: float = 10.0
    local_window_pixels: int = 3
    local_spike_z: float = 8.0
    support_dilation_pixels: int = 1
    support_fraction_threshold: float = 0.10
    tiny_object_pixels: int = 2
    high_priority_score: float = 0.70


@dataclass
class ArtifactScreenResult:
    candidates: list[dict]
    candidate_masks: dict[str, np.ndarray]
    high_priority_masks: dict[str, np.ndarray]
    union_candidate_mask: np.ndarray
    union_high_priority_mask: np.ndarray
    config: dict
    summary: dict


def _robust_z(data) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        return out

    center = float(np.median(vals))
    mad = float(np.median(np.abs(vals - center)))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= 0:
        q25, q75 = np.percentile(vals, [25, 75])
        scale = float((q75 - q25) / 1.349)

    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(vals))

    if not np.isfinite(scale) or scale <= 0:
        out[finite] = 0.0
        return out

    out[finite] = (vals - center) / scale
    return out


def _touches_edge(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _component_bbox(mask: np.ndarray):
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    return (
        int(yy.min()),
        int(yy.max()) + 1,
        int(xx.min()),
        int(xx.max()) + 1,
    )


def _support_fraction(support_mask, region):
    denom = int(np.count_nonzero(region))
    if denom == 0:
        return 0.0
    return float(
        np.count_nonzero(np.asarray(support_mask, dtype=bool) & region)
        / denom
    )


def _safe_peak(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else float("nan")


def screen_artifact_candidates(
    elemental_maps: Mapping[str, np.ndarray],
    *,
    tfy: np.ndarray | None = None,
    cell_labels: np.ndarray | None = None,
    config: ArtifactScreenConfig | None = None,
) -> ArtifactScreenResult:
    cfg = config or ArtifactScreenConfig()

    if not elemental_maps:
        raise ValueError("At least one elemental map is required.")

    arrays = {
        str(name): np.asarray(data, dtype=float)
        for name, data in elemental_maps.items()
    }
    shapes = {arr.shape for arr in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("All elemental maps must have the same shape.")

    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("Artifact screening expects 2D elemental maps.")

    if tfy is not None and np.asarray(tfy).shape != shape:
        raise ValueError("TFY shape must match elemental maps.")
    if cell_labels is not None and np.asarray(cell_labels).shape != shape:
        raise ValueError("cell_labels shape must match elemental maps.")

    z_maps = {name: _robust_z(arr) for name, arr in arrays.items()}

    local_z_maps = {}
    for name, arr in arrays.items():
        finite = np.isfinite(arr)
        fill = float(np.median(arr[finite])) if np.any(finite) else 0.0
        work = np.where(finite, arr, fill)
        local_med = ndimage.median_filter(
            work,
            size=max(3, int(cfg.local_window_pixels) | 1),
            mode="nearest",
        )
        residual = np.where(finite, work - local_med, np.nan)
        local_z_maps[name] = _robust_z(residual)

    tfy_support = (
        _robust_z(tfy) >= float(cfg.tfy_support_z)
        if tfy is not None
        else None
    )
    cell_support = (
        np.asarray(cell_labels) > 0
        if cell_labels is not None
        else None
    )

    candidate_masks = {}
    high_priority_masks = {
        name: np.zeros(shape, dtype=bool)
        for name in arrays
    }
    candidates = []

    structure = ndimage.generate_binary_structure(2, 2)
    object_counter = 0

    for channel, arr in arrays.items():
        z = z_maps[channel]
        local_z = local_z_maps[channel]
        finite = np.isfinite(arr)
        candidate_mask = finite & (
            (z >= float(cfg.candidate_z))
            | (local_z >= float(cfg.local_spike_z))
        )
        candidate_masks[channel] = candidate_mask.copy()

        labels, count = ndimage.label(candidate_mask, structure=structure)

        for label_id in range(1, count + 1):
            component = labels == label_id
            area = int(np.count_nonzero(component))
            if area == 0:
                continue

            object_counter += 1
            dilated = component.copy()
            for _ in range(max(0, int(cfg.support_dilation_pixels))):
                dilated = ndimage.binary_dilation(
                    dilated,
                    structure=structure,
                )

            support_channels = []
            for other, other_z in z_maps.items():
                if other == channel:
                    continue
                frac = _support_fraction(
                    other_z >= float(cfg.cross_channel_support_z),
                    dilated,
                )
                if frac >= float(cfg.support_fraction_threshold):
                    support_channels.append(other)

            tfy_fraction = (
                _support_fraction(tfy_support, dilated)
                if tfy_support is not None
                else float("nan")
            )
            cell_fraction = (
                _support_fraction(cell_support, component)
                if cell_support is not None
                else float("nan")
            )
            peak_z = _safe_peak(z[component])
            peak_local_z = _safe_peak(local_z[component])
            tiny = area <= int(cfg.tiny_object_pixels)
            touches_edge = _touches_edge(component)
            cross_count = len(support_channels)

            score = 0.0
            reasons = []

            if tiny and np.isfinite(peak_local_z) and (
                peak_local_z >= float(cfg.local_spike_z)
            ):
                score += 0.35
                reasons.append("tiny_extreme_local_spike")
            elif area <= 4:
                score += 0.15
                reasons.append("very_small_high_signal_object")

            if cross_count == 0:
                score += 0.22
                reasons.append("no_cross_channel_support")
            elif cross_count == 1:
                score += 0.08
                reasons.append("limited_cross_channel_support")
            elif cross_count >= 2:
                score -= 0.18
                reasons.append("supported_by_multiple_channels")

            if np.isfinite(tfy_fraction):
                if tfy_fraction < 0.10:
                    score += 0.15
                    reasons.append("weak_tfy_support")
                elif tfy_fraction >= 0.50:
                    score -= 0.10
                    reasons.append("strong_tfy_support")

            if np.isfinite(cell_fraction):
                if cell_fraction < 0.10:
                    score += 0.15
                    reasons.append("outside_provisional_cells")
                elif cell_fraction >= 0.50:
                    score -= 0.10
                    reasons.append("inside_provisional_cell")

            if np.isfinite(peak_z) and peak_z >= float(cfg.extreme_z):
                score += 0.08
                reasons.append("extreme_within_channel_signal")

            if touches_edge:
                score += 0.04
                reasons.append("touches_scan_edge")

            if area >= 25:
                score -= 0.08
                reasons.append("extended_spatial_object")

            score = float(np.clip(score, 0.0, 1.0))

            if (
                tiny
                and cross_count == 0
                and np.isfinite(peak_local_z)
                and peak_local_z >= float(cfg.local_spike_z)
            ):
                classification = "isolated_channel_spike_candidate"
            elif (
                cross_count == 0
                and (not np.isfinite(cell_fraction) or cell_fraction < 0.20)
                and (not np.isfinite(tfy_fraction) or tfy_fraction < 0.20)
            ):
                classification = (
                    "channel_specific_external_object_candidate"
                )
            elif (
                cross_count == 0
                and np.isfinite(cell_fraction)
                and cell_fraction >= 0.20
            ):
                classification = (
                    "cell_associated_channel_specific_feature"
                )
            elif cross_count >= 2:
                classification = "cross_channel_supported_feature"
            else:
                classification = "review_candidate"

            priority = (
                "high"
                if score >= float(cfg.high_priority_score)
                else "review"
                if score >= 0.40
                else "context_supported"
            )

            if priority == "high":
                high_priority_masks[channel][component] = True

            yy, xx = np.nonzero(component)
            bbox = _component_bbox(component)

            candidates.append(
                {
                    "candidate_id": (
                        f"artifact_candidate_{object_counter:05d}"
                    ),
                    "channel": channel,
                    "classification": classification,
                    "review_priority": priority,
                    "suspicion_score": score,
                    "score_is_probability": False,
                    "action": "retain_and_flag",
                    "area_pixels": area,
                    "centroid_y_pixel": float(np.mean(yy)),
                    "centroid_x_pixel": float(np.mean(xx)),
                    "bbox_y0": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x0": bbox[2],
                    "bbox_x1": bbox[3],
                    "touches_scan_edge": touches_edge,
                    "peak_robust_z": peak_z,
                    "peak_local_spike_z": peak_local_z,
                    "cross_channel_support_count": cross_count,
                    "supporting_channels": ";".join(
                        sorted(support_channels)
                    ),
                    "tfy_support_fraction": tfy_fraction,
                    "provisional_cell_overlap_fraction": cell_fraction,
                    "reasons": ";".join(reasons),
                }
            )

    union_candidate = np.zeros(shape, dtype=bool)
    union_high = np.zeros(shape, dtype=bool)
    for mask in candidate_masks.values():
        union_candidate |= mask
    for mask in high_priority_masks.values():
        union_high |= mask

    priorities = [row["review_priority"] for row in candidates]
    summary = {
        "candidate_count": int(len(candidates)),
        "high_priority_count": int(
            sum(value == "high" for value in priorities)
        ),
        "review_count": int(
            sum(value == "review" for value in priorities)
        ),
        "context_supported_count": int(
            sum(value == "context_supported" for value in priorities)
        ),
        "channels_screened": sorted(arrays),
        "raw_or_quantitative_maps_modified": False,
        "screening_action": "retain_and_flag",
        "scores_are_probabilities": False,
    }

    return ArtifactScreenResult(
        candidates=candidates,
        candidate_masks=candidate_masks,
        high_priority_masks=high_priority_masks,
        union_candidate_mask=union_candidate,
        union_high_priority_mask=union_high,
        config=asdict(cfg),
        summary=summary,
    )
PY

say "Patching automated Study Analysis to run artifact screening..."
python - "$ENGINE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()


def replace_once(old, new, label):
    global text
    if old not in text:
        raise SystemExit(f"Change-12 patch anchor not found: {label}")
    if text.count(old) != 1:
        raise SystemExit(
            f"Change-12 patch anchor not unique "
            f"({text.count(old)}): {label}"
        )
    text = text.replace(old, new, 1)


replace_once(
    "from yeast_xrf.analysis.maps_concentration import maps_concentration\n",
    "from yeast_xrf.analysis.artifacts import screen_artifact_candidates\n"
    "from yeast_xrf.analysis.maps_concentration import maps_concentration\n",
    "artifact import",
)

replace_once(
    '''    compute_element_pairs: bool = True
    max_scans: int | None = None
''',
    '''    compute_element_pairs: bool = True
    artifact_screening: bool = True
    artifact_method: str = "Fitted"
    artifact_reference: str = "US_IC"
    max_scans: int | None = None
''',
    "StudyConfig fields",
)

replace_once(
    '''    compute_element_pairs=True,
    max_scans=None,
):
''',
    '''    compute_element_pairs=True,
    artifact_screening=True,
    artifact_method="Fitted",
    artifact_reference="US_IC",
    max_scans=None,
):
''',
    "analyze_study signature",
)

replace_once(
    '''        compute_element_pairs=bool(compute_element_pairs),
        max_scans=(
''',
    '''        compute_element_pairs=bool(compute_element_pairs),
        artifact_screening=bool(artifact_screening),
        artifact_method=str(artifact_method),
        artifact_reference=str(artifact_reference),
        max_scans=(
''',
    "StudyConfig construction",
)

replace_once(
    '''    pair_rows = []
    qc_rows = []
    failures = []
''',
    '''    pair_rows = []
    artifact_rows = []
    qc_rows = []
    failures = []
''',
    "study row containers",
)

replace_once(
    '''                product_count = 0
                method_reference_channel_counts = {}

                for method in methods:
''',
    '''                product_count = 0
                method_reference_channel_counts = {}
                artifact_screen_maps = {}

                for method in methods:
''',
    "per-scan artifact map container",
)

replace_once(
    '''                        if (
                            save_concentration_arrays
                            and concentration_maps
                        ):
''',
    '''                        if (
                            artifact_screening
                            and method == artifact_method
                            and reference == artifact_reference
                        ):
                            artifact_screen_maps = {
                                name: np.asarray(data, dtype=float).copy()
                                for name, data in concentration_maps.items()
                            }

                        if (
                            save_concentration_arrays
                            and concentration_maps
                        ):
''',
    "capture artifact screening maps",
)

artifact_block = '''                artifact_candidate_count = 0
                artifact_high_priority_count = 0
                artifact_screening_status = "disabled"

                if artifact_screening:
                    if artifact_screen_maps:
                        artifact_result = screen_artifact_candidates(
                            artifact_screen_maps,
                            tfy=tfy,
                            cell_labels=cells.labels,
                        )
                        artifact_screening_status = "completed"
                        artifact_candidate_count = int(
                            artifact_result.summary["candidate_count"]
                        )
                        artifact_high_priority_count = int(
                            artifact_result.summary[
                                "high_priority_count"
                            ]
                        )

                        for candidate in artifact_result.candidates:
                            artifact_rows.append(
                                {
                                    "study_run_id": run_id,
                                    "scan": scan_name,
                                    "sample_id": metadata.get(
                                        "sample_id",
                                        scan_stem,
                                    ),
                                    "condition": metadata.get(
                                        "condition",
                                        "",
                                    ),
                                    "strain_or_mutant": metadata.get(
                                        "strain_or_mutant",
                                        "",
                                    ),
                                    "treatment": metadata.get(
                                        "treatment",
                                        "",
                                    ),
                                    "biological_replicate": metadata.get(
                                        "biological_replicate",
                                        "",
                                    ),
                                    "technical_replicate": metadata.get(
                                        "technical_replicate",
                                        "",
                                    ),
                                    "screening_method": artifact_method,
                                    "screening_reference": (
                                        artifact_reference
                                    ),
                                    **candidate,
                                }
                            )

                        mask_payload = {
                            "union_candidate_mask": np.asarray(
                                artifact_result.union_candidate_mask,
                                dtype=np.uint8,
                            ),
                            "union_high_priority_mask": np.asarray(
                                artifact_result.union_high_priority_mask,
                                dtype=np.uint8,
                            ),
                        }
                        for name, mask in (
                            artifact_result.candidate_masks.items()
                        ):
                            mask_payload[
                                f"{name}__candidate"
                            ] = np.asarray(mask, dtype=np.uint8)
                        for name, mask in (
                            artifact_result.high_priority_masks.items()
                        ):
                            mask_payload[
                                f"{name}__high_priority"
                            ] = np.asarray(mask, dtype=np.uint8)

                        np.savez_compressed(
                            scan_dir / "artifact_candidate_masks.npz",
                            **mask_payload,
                        )
                        (
                            scan_dir / "artifact_screening.json"
                        ).write_text(
                            json.dumps(
                                {
                                    "status": "completed",
                                    "method": artifact_method,
                                    "reference": artifact_reference,
                                    "summary": artifact_result.summary,
                                    "config": artifact_result.config,
                                    "policy": (
                                        "retain_and_flag; no automatic "
                                        "pixel deletion or correction"
                                    ),
                                },
                                indent=2,
                                default=str,
                            )
                        )
                    else:
                        artifact_screening_status = (
                            "not_run_screening_product_unavailable"
                        )
                        qc_rows.append(
                            {
                                "level": "scan",
                                "scan": scan_name,
                                "cell_uid": "",
                                "flag": (
                                    "artifact_screening_product_unavailable"
                                ),
                                "severity": "info",
                                "source": "artifact_screening",
                                "details": (
                                    f"Requested screening product "
                                    f"{artifact_method}/"
                                    f"{artifact_reference} was not "
                                    "generated in this study run."
                                ),
                            }
                        )

'''

replace_once(
    "                scan_rows.append(\n"
    "                    {\n"
    '                        "study_run_id": run_id,\n',
    artifact_block
    + "                scan_rows.append(\n"
    + "                    {\n"
    + '                        "study_run_id": run_id,\n',
    "artifact screening before scan row",
)

replace_once(
    '''                        "concentration_products": int(
                            product_count
                        ),
                        "method_reference_channel_counts": (
''',
    '''                        "concentration_products": int(
                            product_count
                        ),
                        "artifact_screening_status": (
                            artifact_screening_status
                        ),
                        "artifact_candidate_count": int(
                            artifact_candidate_count
                        ),
                        "artifact_high_priority_count": int(
                            artifact_high_priority_count
                        ),
                        "method_reference_channel_counts": (
''',
    "scan artifact counts",
)

replace_once(
    '''                            "artifact_correction": {
                                "status": "not_yet_applied",
                                "planned_change": 12,
                            },
''',
    '''                            "artifact_screening": {
                                "status": artifact_screening_status,
                                "method": artifact_method,
                                "reference": artifact_reference,
                                "candidate_count": int(
                                    artifact_candidate_count
                                ),
                                "high_priority_count": int(
                                    artifact_high_priority_count
                                ),
                                "action": "retain_and_flag",
                                "maps_modified": False,
                            },
''',
    "scan provenance artifact block",
)

replace_once(
    '''    element_pairs_df = pd.DataFrame(pair_rows)
    qc_df = pd.DataFrame(qc_rows)
''',
    '''    element_pairs_df = pd.DataFrame(pair_rows)
    artifact_candidates_df = pd.DataFrame(artifact_rows)
    qc_df = pd.DataFrame(qc_rows)
''',
    "artifact dataframe",
)

replace_once(
    '''        ("element_pairs", element_pairs_df),
        ("qc_flags", qc_df),
''',
    '''        ("element_pairs", element_pairs_df),
        ("artifact_candidates", artifact_candidates_df),
        ("qc_flags", qc_df),
''',
    "artifact output table",
)

replace_once(
    '''        "element_pair_records": int(
            len(element_pairs_df)
        ),
        "qc_flags": int(len(qc_df)),
        "segmentation_status": "provisional_tfy_v1",
        "canonical_cells_ready": False,
        "artifact_layer_ready": False,
''',
    '''        "element_pair_records": int(
            len(element_pairs_df)
        ),
        "artifact_candidate_records": int(
            len(artifact_candidates_df)
        ),
        "artifact_high_priority_records": int(
            np.count_nonzero(
                artifact_candidates_df.get(
                    "review_priority",
                    pd.Series(dtype=object),
                )
                == "high"
            )
        )
        if not artifact_candidates_df.empty
        else 0,
        "qc_flags": int(len(qc_df)),
        "segmentation_status": "provisional_tfy_v1",
        "canonical_cells_ready": False,
        "artifact_layer_ready": bool(artifact_screening),
        "artifact_policy": "retain_and_flag",
''',
    "study summary artifact status",
)

replace_once(
    '''                    (
                        "No channel-specific artifact correction is "
                        "applied yet."
                    ),
''',
    '''                    (
                        "Artifact screening is flag-first and "
                        "non-destructive. Candidates are retained; "
                        "no pixels are silently deleted or corrected."
                    ),
''',
    "study provenance boundary",
)

replace_once(
    '''        "element_pairs": element_pairs_df,
        "qc_flags": qc_df,
''',
    '''        "element_pairs": element_pairs_df,
        "artifact_candidates": artifact_candidates_df,
        "qc_flags": qc_df,
''',
    "returned artifact dataframe",
)

path.write_text(text)
print("Study engine patched for Change 12.")
PY

say "Patching Analyze Study CLI..."
python - "$CLI" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

old = '''    parser.add_argument(
        "--no-pairs",
        action="store_true",
    )
'''
new = '''    parser.add_argument(
        "--no-pairs",
        action="store_true",
    )
    parser.add_argument(
        "--no-artifact-screening",
        action="store_true",
        help=(
            "Disable flag-only artifact/contamination candidate screening."
        ),
    )
'''
if old not in text:
    raise SystemExit("Could not locate CLI --no-pairs anchor.")
text = text.replace(old, new, 1)

old = '''        compute_element_pairs=not args.no_pairs,
        max_scans=args.max_scans,
    )
'''
new = '''        compute_element_pairs=not args.no_pairs,
        artifact_screening=not args.no_artifact_screening,
        max_scans=args.max_scans,
    )
'''
if old not in text:
    raise SystemExit("Could not locate analyze_study CLI call anchor.")
text = text.replace(old, new, 1)

path.write_text(text)
print("Analyze Study CLI patched.")
PY

say "Patching ANALYZE / RESULTS UI..."
python - "$APP" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()


def replace_once(old, new, label):
    global text
    if old not in text:
        raise SystemExit(f"Change-12 UI anchor not found: {label}")
    if text.count(old) != 1:
        raise SystemExit(
            f"Change-12 UI anchor not unique "
            f"({text.count(old)}): {label}"
        )
    text = text.replace(old, new, 1)


replace_once(
    '''            ✓ concentration arrays + comparison tables  
            ✓ QC flags + full provenance
''',
    '''            ✓ concentration arrays + comparison tables  
            ✓ artifact / contamination candidate screening  
            ✓ QC flags + full provenance
''',
    "Analyze contract",
)

replace_once(
    '''              claim these as canonical multichannel-consensus cells. Artifact
              correction and multichannel cell consensus come next.
''',
    '''              claim these as canonical multichannel-consensus cells. Artifact
              screening is now flag-only and non-destructive; multichannel cell
              consensus comes next.
''',
    "Analyze warning",
)

replace_once(
    '''        _sd_arrays = st.checkbox(
                "Save concentration arrays",
                value=True,
                key="study_analysis_save_arrays",
            )
        with _sd_a3:
            _sd_pairs = st.checkbox(
                "Compute element pairs",
                value=True,
                key="study_analysis_pairs",
            )
''',
    '''        _sd_arrays = st.checkbox(
                "Save concentration arrays",
                value=True,
                key="study_analysis_save_arrays",
            )
        with _sd_a3:
            _sd_pairs = st.checkbox(
                "Compute element pairs",
                value=True,
                key="study_analysis_pairs",
            )

        _sd_artifact_screening = st.checkbox(
            "Screen artifact / contamination candidates (retain + flag)",
            value=True,
            key="study_analysis_artifact_screening",
            help=(
                "Flags suspicious channel-specific objects and local spikes. "
                "No pixels are deleted or corrected."
            ),
        )
''',
    "Analyze artifact toggle",
)

replace_once(
    '''                    compute_element_pairs=bool(_sd_pairs),
                )
''',
    '''                    compute_element_pairs=bool(_sd_pairs),
                    artifact_screening=bool(
                        _sd_artifact_screening
                    ),
                )
''',
    "Analyze artifact argument",
)

replace_once(
    '''          Results are quantitative, but cell delineation will be replaced by
          the upcoming multichannel-consensus engine before these become the
          canonical biological cell dataset.
''',
    '''          Results are quantitative, but cell delineation will be replaced by
          the upcoming multichannel-consensus engine before these become the
          canonical biological cell dataset. Artifact candidates are retained
          and flagged for review rather than automatically removed.
''',
    "Results warning",
)

replace_once(
    '''    _sd_pairs_df = _sd_read_table("element_pairs")
    _sd_qc = _sd_read_table("qc_flags")
''',
    '''    _sd_pairs_df = _sd_read_table("element_pairs")
    _sd_artifacts = _sd_read_table("artifact_candidates")
    _sd_qc = _sd_read_table("qc_flags")
''',
    "Results artifact table load",
)

replace_once(
    '''    with _sd_tabs[4]:
        if not _sd_qc.empty:
            st.markdown("#### QC flags")
''',
    '''    with _sd_tabs[4]:
        st.markdown("#### Artifact / contamination review")
        st.caption(
            "Flag-first screening only. A candidate is not automatically an "
            "artifact: real elemental localization can be channel-specific. "
            "Suspicion scores are heuristic review priorities, not probabilities."
        )
        if _sd_artifacts.empty:
            st.success(
                "No artifact candidates were recorded for this study run, "
                "or this run predates Change 12."
            )
        else:
            _sd_artifact_filter = st.columns(3)
            _sd_priority = _sd_artifact_filter[0].multiselect(
                "Review priority",
                sorted(
                    _sd_artifacts["review_priority"]
                    .dropna().astype(str).unique()
                )
                if "review_priority" in _sd_artifacts
                else [],
                default=(
                    ["high"]
                    if "review_priority" in _sd_artifacts
                    and "high"
                    in set(
                        _sd_artifacts["review_priority"]
                        .dropna().astype(str)
                    )
                    else []
                ),
                key="results_artifact_priority",
            )
            _sd_artifact_element = _sd_artifact_filter[1].multiselect(
                "Artifact-review element",
                sorted(
                    _sd_artifacts["channel"]
                    .dropna().astype(str).unique()
                )
                if "channel" in _sd_artifacts
                else [],
                key="results_artifact_element",
            )
            _sd_artifact_scan = _sd_artifact_filter[2].multiselect(
                "Artifact-review scan",
                sorted(
                    _sd_artifacts["scan"]
                    .dropna().astype(str).unique()
                )
                if "scan" in _sd_artifacts
                else [],
                key="results_artifact_scan",
            )

            _sd_artifact_view = _sd_artifacts.copy()
            if _sd_priority:
                _sd_artifact_view = _sd_artifact_view[
                    _sd_artifact_view["review_priority"]
                    .astype(str).isin(_sd_priority)
                ]
            if _sd_artifact_element:
                _sd_artifact_view = _sd_artifact_view[
                    _sd_artifact_view["channel"]
                    .astype(str).isin(_sd_artifact_element)
                ]
            if _sd_artifact_scan:
                _sd_artifact_view = _sd_artifact_view[
                    _sd_artifact_view["scan"]
                    .astype(str).isin(_sd_artifact_scan)
                ]

            st.dataframe(
                _sd_artifact_view,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        if not _sd_qc.empty:
            st.markdown("#### QC flags")
''',
    "QC artifact review section",
)

replace_once(
    '''            "element_pairs.csv",
            "qc_flags.csv",
''',
    '''            "element_pairs.csv",
            "artifact_candidates.csv",
            "qc_flags.csv",
''',
    "Files artifact table",
)

path.write_text(text)
print("ANALYZE / RESULTS UI patched for Change 12.")
PY

say "Writing synthetic artifact-screening tests..."
cat > "$TEST" <<'PY'
import numpy as np

from yeast_xrf.analysis.artifacts import (
    ArtifactScreenConfig,
    screen_artifact_candidates,
)


def _disk(shape, cy, cx, r):
    yy, xx = np.indices(shape)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def test_isolated_channel_spike_is_flagged_without_mutation():
    shape = (48, 48)
    cell = _disk(shape, 24, 24, 9)

    maps = {}
    for name, scale in (
        ("P", 5.0),
        ("S", 4.0),
        ("K", 6.0),
        ("Zn", 1.0),
    ):
        arr = np.zeros(shape, dtype=float)
        arr[cell] = scale
        maps[name] = arr

    maps["Zn"][5, 6] = 100.0

    tfy = np.zeros(shape, dtype=float)
    tfy[cell] = 10.0
    labels = np.zeros(shape, dtype=int)
    labels[cell] = 1

    originals = {k: v.copy() for k, v in maps.items()}

    result = screen_artifact_candidates(
        maps,
        tfy=tfy,
        cell_labels=labels,
        config=ArtifactScreenConfig(
            candidate_z=5.0,
            local_spike_z=6.0,
            high_priority_score=0.60,
        ),
    )

    zn_rows = [
        row
        for row in result.candidates
        if row["channel"] == "Zn"
        and row["centroid_y_pixel"] < 10
    ]
    assert zn_rows
    row = max(zn_rows, key=lambda x: x["suspicion_score"])
    assert row["action"] == "retain_and_flag"
    assert row["score_is_probability"] is False
    assert row["cross_channel_support_count"] == 0
    assert row["tfy_support_fraction"] < 0.2
    assert row["provisional_cell_overlap_fraction"] < 0.2
    assert row["review_priority"] == "high"

    for key in maps:
        assert np.array_equal(maps[key], originals[key])


def test_cross_channel_cell_feature_is_retained():
    shape = (50, 50)
    cell = _disk(shape, 25, 25, 8)
    maps = {}
    for name in ("P", "S", "K"):
        arr = np.zeros(shape, dtype=float)
        arr[cell] = 8.0
        maps[name] = arr

    tfy = np.zeros(shape, dtype=float)
    tfy[cell] = 12.0
    labels = np.zeros(shape, dtype=int)
    labels[cell] = 1

    result = screen_artifact_candidates(
        maps,
        tfy=tfy,
        cell_labels=labels,
        config=ArtifactScreenConfig(candidate_z=2.0),
    )

    assert result.summary["raw_or_quantitative_maps_modified"] is False
    assert all(
        row["action"] == "retain_and_flag"
        for row in result.candidates
    )


def test_shape_mismatch_is_rejected():
    maps = {
        "P": np.zeros((10, 10)),
        "Zn": np.zeros((11, 10)),
    }
    try:
        screen_artifact_candidates(maps)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch ValueError")
PY

say "Writing UI / study architecture tests..."
cat > "$ARCH_TEST" <<'PY'
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
ENGINE = (
    ROOT
    / "src"
    / "yeast_xrf"
    / "analysis"
    / "study_analysis.py"
)


def test_artifact_screening_is_default_and_non_destructive():
    text = APP.read_text()
    assert (
        "Screen artifact / contamination candidates (retain + flag)"
        in text
    )
    assert "study_analysis_artifact_screening" in text
    assert "artifact_screening=bool(" in text
    assert (
        "Suspicion scores are heuristic review priorities"
        in text
    )


def test_results_show_artifact_review_table():
    text = APP.read_text()
    for term in (
        "artifact_candidates",
        "Artifact / contamination review",
        "results_artifact_priority",
        "results_artifact_element",
        "retain + flag",
    ):
        assert term in text


def test_study_engine_writes_artifact_layer():
    text = ENGINE.read_text()
    for term in (
        "screen_artifact_candidates",
        "artifact_candidate_masks.npz",
        "artifact_screening.json",
        '"artifact_layer_ready": bool(artifact_screening)',
        '"artifact_policy": "retain_and_flag"',
    ):
        assert term in text


def test_ui_still_respects_hdf5_boundary():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
PY

say "Writing real-data Change-12 validator..."
cat > "$VALIDATOR" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

from yeast_xrf.analysis.study_analysis import analyze_study


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT = (
    ROOT
    / "analysis"
    / "validation"
    / "change12_artifact_smoke"
)
REPORT = (
    ROOT
    / "analysis"
    / "validation"
    / "change12_artifact_validation.json"
)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(
        "SOURDOUGH Change 12 · "
        "artifact / contamination screening"
    )
    print("=" * 88)

    result = analyze_study(
        DATA,
        output_root=OUT,
        methods=("Fitted", "NNLS", "ROI"),
        references=("US_IC", "DS_IC"),
        save_concentration_arrays=False,
        compute_element_pairs=False,
        artifact_screening=True,
        max_scans=1,
    )

    summary = result["summary"]
    scans = result["scans"]
    artifacts = result["artifact_candidates"]

    scan_status = (
        str(
            scans.iloc[0].get(
                "artifact_screening_status",
                "",
            )
        )
        if not scans.empty
        else ""
    )

    checks = {
        "scan_analyzed": summary["scans_analyzed"] == 1,
        "no_scan_failure": summary["scan_failures"] == 0,
        "artifact_layer_ready": (
            summary["artifact_layer_ready"] is True
        ),
        "flag_only_policy": (
            summary["artifact_policy"] == "retain_and_flag"
        ),
        "screening_executed": scan_status == "completed",
        "artifact_dataframe_returned": artifacts is not None,
    }

    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "artifact_candidate_count": int(len(artifacts)),
        "note": (
            "A real scan is not required to contain a high-priority "
            "artifact. Validation checks execution and the "
            "non-destructive study artifact layer."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, indent=2, default=str)
    )

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print("-" * 88)
    print(f"Artifact candidates recorded: {len(artifacts)}")
    print(f"Wrote: {REPORT.relative_to(ROOT)}")

    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
PY
chmod +x "$VALIDATOR"

say "Writing Change-12 documentation..."
cat > "$DOC" <<'EOF'
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
EOF

cat >> "$STUDY_DOC" <<'EOF'

## Change 12 — artifact / contamination screening

Automated study runs include a non-destructive artifact review layer.
Candidates are retained and flagged using within-channel extremeness, local
spike behavior, cross-channel support, TFY support, provisional-cell overlap,
size, and edge context.

The output is `artifact_candidates.csv` plus per-scan masks and provenance.
Suspicion scores are heuristic review priorities, not probabilities.

No concentration maps are modified. The current screening product is explicitly
Fitted / US_IC; this is a screening choice and does not make that product the
scientifically preferred quantitative result.
EOF

say "Adding permanent validator target..."
if ! grep -q '^validate-artifacts:' "$MAKEFILE"; then
cat >> "$MAKEFILE" <<'EOF'

validate-artifacts:
	PYTHONPATH=$$PWD/src python scripts/validate_artifact_screening.py
EOF
fi

say "Syntax-checking changed Python..."
python -m py_compile \
  "$ART" \
  "$ENGINE" \
  "$CLI" \
  "$APP" \
  "$VALIDATOR" \
  "$TEST" \
  "$ARCH_TEST"

APPLIED=1

say "Running synthetic + architecture + scientific regressions..."
cd "$ROOT"
TESTS=(
  tests/test_artifact_screening.py
  tests/test_artifact_results_ui.py
  tests/test_study_analysis.py
  tests/test_study_front_door.py
  tests/test_empty_study_tables.py
  tests/test_maps_concentration.py
  tests/test_maps_concentration_io.py
)
[[ -f tests/test_change10_quantification_ui.py ]] && \
  TESTS+=(tests/test_change10_quantification_ui.py)

PYTHONPATH="$ROOT/src" python -m pytest -q "${TESTS[@]}"

say "Running real-data artifact-screening smoke..."
PYTHONPATH="$ROOT/src" python scripts/validate_artifact_screening.py

say "Final UI architecture boundary checks..."
if grep -n '/MAPS/' "$APP"; then
  die "UI contains direct /MAPS/ access."
fi
if grep -n 'import h5py' "$APP"; then
  die "UI imports h5py directly."
fi

trap - EXIT
APPLIED=0

say "CHANGE 12 COMPLETE."
cat <<EOF

Artifact / contamination candidate layer installed.

Automatic ANALYZE runs now:
  ✓ screen quantitative elemental maps for suspicious spatial objects
  ✓ use local-spike + cross-channel + TFY + cell-context evidence
  ✓ write artifact_candidates.csv
  ✓ write per-scan artifact_candidate_masks.npz
  ✓ write per-scan artifact_screening.json
  ✓ expose artifact review under RESULTS -> QC & Failures
  ✓ retain every candidate by default
  ✓ never modify raw HDF5
  ✓ never silently alter concentration maps

Scientific policy:
  RETAIN + FLAG
  suspicion_score = heuristic review priority, NOT probability
  channel-specific cellular chemistry is not automatically called artifact

Current screening product:
  Fitted / US_IC concentration
  screening choice only; not a declaration of scientific preference

Permanent validation:
  make validate-artifacts

Restart:
  cd "$ROOT"
  make app

For a NEW full automated run with Change 12:
  choose ANALYZE
  keep "Screen artifact / contamination candidates" checked
  click "Analyze entire study"

Next:
  Change 13 — Multichannel Cell Consensus

Backup:
  $BACKUP
EOF
