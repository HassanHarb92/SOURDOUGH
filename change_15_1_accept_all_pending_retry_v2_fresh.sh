#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/hharb/Desktop/Projects/Yeast"
REVIEW="$ROOT/src/yeast_xrf/analysis/cell_review.py"
APP="$ROOT/app/streamlit_app.py"
TEST="$ROOT/tests/test_bulk_accept_cell_review.py"
UI_TEST="$ROOT/tests/test_bulk_accept_cell_review_ui.py"
DOC="$ROOT/docs/CELL_REVIEW.md"

ID="change_15_1_accept_all_pending_retry_v2"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/.yeast-xrf-backups/${ID}_${STAMP}"

say(){ printf '\n[%s] %s\n' "$ID" "$*"; }
die(){ printf '\n[%s] ERROR: %s\n' "$ID" "$*" >&2; exit 1; }

[[ -f "$REVIEW" ]] || die "Persistent Cell Review module not found."
[[ -f "$APP" ]] || die "SOURDOUGH app not found."
grep -q 'Canonical Cell Dataset' "$APP" || \
  die "Change 15 Canonical Cell Dataset is required."

mkdir -p "$BACKUP/src/yeast_xrf/analysis" "$BACKUP/app" "$BACKUP/tests" "$BACKUP/docs"

backup_or_absent(){
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    cp -p "$src" "$dst"
  else
    : > "${dst}.ABSENT"
  fi
}

backup_or_absent "$REVIEW" "$BACKUP/src/yeast_xrf/analysis/cell_review.py"
backup_or_absent "$APP" "$BACKUP/app/streamlit_app.py"
backup_or_absent "$TEST" "$BACKUP/tests/test_bulk_accept_cell_review.py"
backup_or_absent "$UI_TEST" "$BACKUP/tests/test_bulk_accept_cell_review_ui.py"
backup_or_absent "$DOC" "$BACKUP/docs/CELL_REVIEW.md"

APPLIED=0

restore(){
  local current="$1" saved="$2"
  if [[ -f "$saved" ]]; then
    cp -p "$saved" "$current" || true
  elif [[ -f "${saved}.ABSENT" ]]; then
    rm -f "$current"
  fi
}

rollback(){
  rc=$?
  if [[ $rc -ne 0 && $APPLIED -eq 1 ]]; then
    say "Validation failed. Rolling back Change 15.1..."
    restore "$REVIEW" "$BACKUP/src/yeast_xrf/analysis/cell_review.py"
    restore "$APP" "$BACKUP/app/streamlit_app.py"
    restore "$TEST" "$BACKUP/tests/test_bulk_accept_cell_review.py"
    restore "$UI_TEST" "$BACKUP/tests/test_bulk_accept_cell_review_ui.py"
    restore "$DOC" "$BACKUP/docs/CELL_REVIEW.md"
    say "Rollback complete. Backup retained at: $BACKUP"
  fi
  exit "$rc"
}
trap rollback EXIT

say "Adding bulk accept-pending review action..."
python - "$REVIEW" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()

if '"bulk_accept_pending",' not in text:
    anchor = '    "automated_call_confirmed",\n'
    if anchor not in text:
        raise SystemExit(
            "Change-15.1 could not find REVIEW_REASONS anchor."
        )
    text = text.replace(
        anchor,
        anchor + '    "bulk_accept_pending",\n',
        1,
    )

if "def bulk_accept_pending_reviews(" not in text:
    anchor = '''def _mark_canonicalization_stale(run_dir: Path) -> None:
'''
    if anchor not in text:
        raise SystemExit(
            "Change-15.1 requires the Change-15 staleness hook."
        )

    function = '''def bulk_accept_pending_reviews(
    run_dir,
    *,
    reviewer: str = "",
    notes: str = "",
) -> dict:
    # Accept every currently pending review row in one atomic update.
    # Existing accept/reject/ambiguous decisions are preserved.
    run_dir = Path(run_dir)
    review_path = run_dir / "cell_review.csv"

    review = _read_csv_safe(review_path)
    if review.empty:
        review = ensure_cell_review(run_dir)

    required = {
        "scan",
        "consensus_candidate_id",
        "review_decision",
    }
    missing = required - set(review.columns)
    if missing:
        raise ValueError(
            "cell_review.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    for column, default in (
        ("review_reason", ""),
        ("review_notes", ""),
        ("reviewer", ""),
        ("reviewed_at_utc", ""),
        ("review_source", ""),
        ("review_mask_policy", "consensus_mask_unmodified"),
        ("canonical_cell", False),
    ):
        if column not in review:
            review[column] = default

    decisions = (
        review["review_decision"]
        .fillna("pending")
        .astype(str)
        .str.lower()
    )
    pending = decisions.eq("pending")
    updated_count = int(pending.sum())

    if updated_count == 0:
        return {
            "review": review,
            "updated_count": 0,
        }

    timestamp = datetime.now(timezone.utc).isoformat()

    review.loc[pending, "review_decision"] = "accept"
    review.loc[pending, "review_reason"] = "bulk_accept_pending"
    review.loc[pending, "review_notes"] = str(notes)
    review.loc[pending, "reviewer"] = str(reviewer)
    review.loc[pending, "reviewed_at_utc"] = timestamp
    review.loc[pending, "review_source"] = (
        "manual_ui_bulk_accept_pending"
    )
    review.loc[pending, "review_mask_policy"] = (
        "consensus_mask_unmodified_accept_for_change15"
    )
    review.loc[pending, "canonical_cell"] = False

    _atomic_csv(review, review_path)
    _mark_canonicalization_stale(run_dir)

    return {
        "review": review,
        "updated_count": updated_count,
    }


'''
    text = text.replace(
        anchor,
        function + anchor,
        1,
    )

p.write_text(text)
print("Bulk accept-pending review action installed.")
PY

say "Adding Accept all pending button to RESULTS -> Cell Consensus..."
python - "$APP" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()


def one(old, new, label):
    global text
    if old not in text:
        raise SystemExit(
            f"Change-15.1 UI anchor not found: {label}"
        )
    if text.count(old) != 1:
        raise SystemExit(
            f"Change-15.1 UI anchor not unique "
            f"({text.count(old)}): {label}"
        )
    text = text.replace(old, new, 1)


one(
    '''    ensure_cell_review as _sd_ensure_cell_review,
    review_summary as _sd_review_summary_fn,
    save_cell_review as _sd_save_cell_review,
)
''',
    '''    ensure_cell_review as _sd_ensure_cell_review,
    review_summary as _sd_review_summary_fn,
    save_cell_review as _sd_save_cell_review,
    bulk_accept_pending_reviews as _sd_bulk_accept_pending_reviews,
)
''',
    "cell-review import",
)

bulk_ui = r'''
            with st.expander(
                "Bulk review actions",
                expanded=False,
            ):
                st.warning(
                    "Accept all pending marks every candidate that is "
                    "currently `pending` as `accept` across this study. "
                    "Existing accepted, rejected, and ambiguous decisions "
                    "are not changed."
                )

                _sd_bulk_cols = st.columns([1.2, 1.8])
                _sd_bulk_reviewer = _sd_bulk_cols[0].text_input(
                    "Bulk reviewer",
                    key="results_bulk_accept_reviewer",
                    placeholder="optional",
                )
                _sd_bulk_note = _sd_bulk_cols[1].text_input(
                    "Bulk review note",
                    key="results_bulk_accept_note",
                    placeholder="optional",
                )

                _sd_bulk_confirm = st.checkbox(
                    (
                        "I understand this will accept all "
                        f"{_sd_review_summary['pending']} currently "
                        "pending candidates without individual decisions."
                    ),
                    key="results_bulk_accept_confirm",
                    disabled=(
                        _sd_review_summary["pending"] == 0
                    ),
                )

                _sd_bulk_accept_clicked = st.button(
                    (
                        "Accept all pending "
                        f"({_sd_review_summary['pending']})"
                    ),
                    type="primary",
                    key="results_bulk_accept_pending",
                    disabled=(
                        _sd_review_summary["pending"] == 0
                        or not _sd_bulk_confirm
                    ),
                )

                if _sd_bulk_accept_clicked:
                    _sd_bulk_result = (
                        _sd_bulk_accept_pending_reviews(
                            _sd_run,
                            reviewer=_sd_bulk_reviewer,
                            notes=_sd_bulk_note,
                        )
                    )
                    _sd_refresh_report_contract(_sd_run)
                    st.success(
                        "Accepted "
                        f"{_sd_bulk_result['updated_count']} "
                        "pending candidate(s). Existing non-pending "
                        "decisions were preserved."
                    )
                    st.rerun()

'''

one(
    '''            _sd_class_counts = (
''',
    bulk_ui + '''            _sd_class_counts = (
''',
    "bulk review UI insertion",
)

p.write_text(text)
print("Accept all pending button installed.")
PY

say "Writing Change-15.1 tests..."
cat > "$TEST" <<'PY'
import pandas as pd

from yeast_xrf.analysis.cell_review import (
    bulk_accept_pending_reviews,
    ensure_cell_review,
)


def consensus():
    return pd.DataFrame(
        [
            {
                "scan": "a.h5",
                "consensus_candidate_id": 1,
                "consensus_class": "high_confidence_cell",
                "consensus_score": 0.9,
                "automated_consensus_accept": True,
            },
            {
                "scan": "a.h5",
                "consensus_candidate_id": 2,
                "consensus_class": "accepted_cell",
                "consensus_score": 0.7,
                "automated_consensus_accept": True,
            },
            {
                "scan": "a.h5",
                "consensus_candidate_id": 3,
                "consensus_class": "ambiguous_cell_candidate",
                "consensus_score": 0.5,
                "automated_consensus_accept": False,
            },
            {
                "scan": "a.h5",
                "consensus_candidate_id": 4,
                "consensus_class": "rejected_channel_specific_object",
                "consensus_score": 0.2,
                "automated_consensus_accept": False,
            },
        ]
    )


def seed(tmp_path):
    pd.DataFrame(consensus()).to_csv(
        tmp_path / "cell_consensus.csv",
        index=False,
    )
    review = ensure_cell_review(
        tmp_path,
        consensus(),
    )
    review.loc[
        review["consensus_candidate_id"].astype(int) == 2,
        "review_decision",
    ] = "accept"
    review.loc[
        review["consensus_candidate_id"].astype(int) == 3,
        "review_decision",
    ] = "ambiguous"
    review.loc[
        review["consensus_candidate_id"].astype(int) == 4,
        "review_decision",
    ] = "reject"
    review.to_csv(
        tmp_path / "cell_review.csv",
        index=False,
    )


def test_bulk_accept_changes_only_pending(tmp_path):
    seed(tmp_path)

    result = bulk_accept_pending_reviews(
        tmp_path,
        reviewer="HH",
        notes="bulk review",
    )
    review = result["review"].copy()

    assert result["updated_count"] == 1

    decisions = {
        int(row.consensus_candidate_id): row.review_decision
        for row in review.itertuples()
    }
    assert decisions == {
        1: "accept",
        2: "accept",
        3: "ambiguous",
        4: "reject",
    }


def test_bulk_accept_records_provenance(tmp_path):
    seed(tmp_path)
    result = bulk_accept_pending_reviews(
        tmp_path,
        reviewer="HH",
        notes="accept remaining pending",
    )
    row = result["review"][
        result["review"][
            "consensus_candidate_id"
        ].astype(int)
        == 1
    ].iloc[0]

    assert row["review_reason"] == "bulk_accept_pending"
    assert row["review_source"] == (
        "manual_ui_bulk_accept_pending"
    )
    assert row["reviewer"] == "HH"
    assert row["review_notes"] == (
        "accept remaining pending"
    )
    assert row["reviewed_at_utc"]
    assert not bool(row["canonical_cell"])


def test_bulk_accept_noop_when_no_pending(tmp_path):
    seed(tmp_path)
    bulk_accept_pending_reviews(tmp_path)
    second = bulk_accept_pending_reviews(tmp_path)
    assert second["updated_count"] == 0


def test_bulk_accept_is_persistent(tmp_path):
    seed(tmp_path)
    bulk_accept_pending_reviews(tmp_path)
    persisted = pd.read_csv(
        tmp_path / "cell_review.csv",
        keep_default_na=False,
    )
    row = persisted[
        persisted["consensus_candidate_id"].astype(int)
        == 1
    ].iloc[0]
    assert row["review_decision"] == "accept"
PY

cat > "$UI_TEST" <<'PY'
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_accept_all_pending_button_exists():
    text = APP.read_text()
    for term in (
        "Bulk review actions",
        "Accept all pending",
        "_sd_bulk_accept_pending_reviews",
        "results_bulk_accept_confirm",
        "currently `pending` as `accept`",
    ):
        assert term in text


def test_bulk_accept_ui_preserves_non_pending_decisions_message():
    text = APP.read_text()
    assert (
        "Existing accepted, rejected, and ambiguous decisions"
        in text
    )
    assert "are not changed" in text


def test_bulk_accept_ui_requires_confirmation():
    text = APP.read_text()
    assert "or not _sd_bulk_confirm" in text


def test_ui_hdf5_boundary_preserved():
    text = APP.read_text()
    assert "import h5py" not in text
    assert "/MAPS/" not in text
PY

cat >> "$DOC" <<'EOF'

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
EOF

say "Syntax checks..."
python -m py_compile \
  "$REVIEW" \
  "$APP" \
  "$TEST" \
  "$UI_TEST"

APPLIED=1

say "Running Change-15.1 + review/canonical regressions..."
cd "$ROOT"
TESTS=(
  tests/test_bulk_accept_cell_review.py
  tests/test_bulk_accept_cell_review_ui.py
  tests/test_cell_review.py
  tests/test_cell_review_ui.py
  tests/test_canonical_cells.py
  tests/test_canonical_cells_ui.py
  tests/test_canonical_report_contract.py
  tests/test_report_contract.py
  tests/test_report_contract_ui.py
  tests/test_cell_consensus.py
  tests/test_cell_consensus_results_ui.py
)
PYTHONPATH="$ROOT/src" python -m pytest -q "${TESTS[@]}"

say "Final UI architecture boundary checks..."
if grep -n '/MAPS/' "$APP"; then
  die "UI contains forbidden direct HDF5-internal path text/access."
fi
if grep -n 'import h5py' "$APP"; then
  die "UI imports h5py directly."
fi

trap - EXIT
APPLIED=0

say "CHANGE 15.1 RETRY V2 COMPLETE."
cat <<EOF

Accept-all review action installed.

Open:
  RESULTS -> Cell Consensus -> Bulk review actions

Button:
  Accept all pending

Behavior:
  pending   -> accept
  accept    -> unchanged
  reject    -> unchanged
  ambiguous -> unchanged

The button requires explicit confirmation and records bulk-review provenance.

After using it:
  make canonical-cells-status

If Pending = 0 and Ambiguous = 0:
  make canonical-cells

Backup:
  $BACKUP
EOF
