'''SOURDOUGH collaborator-facing report contract.

This module does not generate the final narrative report yet. It defines the
stable report structure, evidence requirements, provenance rules, and output
folders that every downstream analysis change must populate.

A report section may be:
- ready: current artifacts are sufficient for that section at its present stage;
- partial: useful evidence exists but a required scientific boundary is open;
- planned: the required downstream analysis has not been implemented yet.

The final report is never declared ready merely because a provisional table
exists.
'''

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import pandas as pd


CONTRACT_VERSION = 1

REPORT_SECTIONS = (
    {
        "id": "study_overview",
        "title": "1. Study Overview",
        "description": (
            "Conditions, mutants/treatments, replicates, scans, "
            "inclusion/exclusion, and acquisition summary."
        ),
        "required_final": (
            "study_manifest.csv",
            "scans.csv",
            "study_summary.json",
        ),
        "planned_change": "available_now",
    },
    {
        "id": "quality_quantification",
        "title": "2. Data Quality & Quantification",
        "description": (
            "Scaler QC, calibration provenance, valid-pixel fractions, "
            "artifact screening, and method/reference transparency."
        ),
        "required_final": (
            "scan_elements.csv",
            "study_provenance.json",
            "qc_flags.csv",
        ),
        "planned_change": "available_now_scan_level",
    },
    {
        "id": "cell_identification",
        "title": "3. Cell Identification & Review",
        "description": (
            "TFY/P/S/K consensus, artifact context, accepted/rejected/"
            "ambiguous candidates, cropped objects, and review decisions."
        ),
        "required_final": (
            "cell_consensus.csv",
            "cell_channel_support.csv",
            "cell_review.csv",
            "canonical_cells.csv",
            "canonical_cell_lineage.csv",
            "canonicalization_summary.json",
        ),
        "planned_change": "15",
    },
    {
        "id": "cellular_composition",
        "title": "4. Cellular Elemental Composition",
        "description": (
            "Canonical cell × element concentrations, summary statistics, "
            "heterogeneity, and cell-to-cell variability."
        ),
        "required_final": (
            "canonical_cell_elements.csv",
        ),
        "planned_change": "16",
    },
    {
        "id": "element_localization",
        "title": "5. Element Localization",
        "description": (
            "Centroids, radial profiles, core-edge enrichment, spatial "
            "coverage, polarization, and localization classes."
        ),
        "required_final": (
            "cell_element_localization.csv",
        ),
        "planned_change": "17",
    },
    {
        "id": "chemical_domains",
        "title": "6. Chemical Domains / Granules / Blobs",
        "description": (
            "Domain counts, sizes, largest-domain area fraction, signal "
            "fraction, granularity, morphology, and spatial position."
        ),
        "required_final": (
            "chemical_domains.csv",
            "cell_domain_summary.csv",
        ),
        "planned_change": "18",
    },
    {
        "id": "element_relationships",
        "title": "7. Element ↔ Element Relationships",
        "description": (
            "Canonical-cell pixel correlation, hotspot overlap, centroid "
            "separation, radial similarity, and cross-enrichment."
        ),
        "required_final": (
            "canonical_element_pairs.csv",
        ),
        "planned_change": "20",
    },
    {
        "id": "cell_phenotypes",
        "title": "8. Cell Phenotypes",
        "description": (
            "Composition + localization + morphology + domain structure "
            "assembled into comparison-ready per-cell phenotype vectors."
        ),
        "required_final": (
            "cell_phenotypes.csv",
        ),
        "planned_change": "20",
    },
    {
        "id": "condition_comparison",
        "title": "9. Condition / Mutant / Treatment Comparison",
        "description": (
            "Replicate-aware distributions, effect sizes, heterogeneity, "
            "composition shifts, localization shifts, and phenotype shifts."
        ),
        "required_final": (
            "condition_comparisons.csv",
            "study_statistics.csv",
        ),
        "planned_change": "20_to_21",
    },
    {
        "id": "organelle_likeness",
        "title": "10. Organelle-Likeness",
        "description": (
            "Evidence-based organelle-likeness only; never definitive "
            "organelle identity from XRF chemistry alone."
        ),
        "required_final": (
            "organelle_likeness.csv",
        ),
        "planned_change": "19",
    },
    {
        "id": "biological_findings",
        "title": "11. Biological Findings",
        "description": (
            "Artifact-grounded factual observations generated only from "
            "validated quantitative outputs."
        ),
        "required_final": (
            "report/biological_findings.json",
        ),
        "planned_change": "22",
    },
    {
        "id": "methods_provenance",
        "title": "12. Methods / Provenance / QC Appendix",
        "description": (
            "Settings, calibration, exclusions, review decisions, software "
            "versioning, input lineage, and scientific boundaries."
        ),
        "required_final": (
            "study_config.json",
            "study_provenance.json",
            "scans.csv",
        ),
        "planned_change": "available_now",
    },
)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def _exists(run_dir: Path, relative: str) -> bool:
    path = run_dir / relative
    return path.is_file() and path.stat().st_size > 0


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _review_state(run_dir: Path) -> dict:
    path = run_dir / "cell_review.csv"
    review = _read_csv_safe(path)
    if review.empty or "review_decision" not in review:
        return {
            "exists": False,
            "candidate_count": 0,
            "pending": 0,
            "accepted": 0,
            "review_complete": False,
        }

    decisions = (
        review["review_decision"]
        .astype(str)
        .str.lower()
    )
    total = int(len(review))
    pending = int((decisions == "pending").sum())
    accepted = int((decisions == "accept").sum())
    return {
        "exists": True,
        "candidate_count": total,
        "pending": pending,
        "accepted": accepted,
        "review_complete": (
            total > 0 and pending == 0
        ),
    }


def _section_status(run_dir: Path, section: dict) -> tuple[str, str]:
    sid = section["id"]
    required = tuple(section["required_final"])
    available = [r for r in required if _exists(run_dir, r)]
    all_present = len(available) == len(required)

    if sid == "study_overview":
        return (
            ("ready" if all_present else "partial"),
            (
                "Study manifest, scan inventory, and summary are available."
                if all_present
                else "Study-level inventory is incomplete."
            ),
        )

    if sid == "quality_quantification":
        essentials = (
            _exists(run_dir, "scan_elements.csv")
            and _exists(run_dir, "study_provenance.json")
        )
        return (
            ("partial" if essentials else "planned"),
            (
                "Scan-level quantification/QC exists. Final cellular "
                "quantification waits for reviewed canonical cells."
                if essentials
                else "Quantification/QC evidence is not yet available."
            ),
        )

    if sid == "cell_identification":
        consensus = _exists(run_dir, "cell_consensus.csv")
        review = _review_state(run_dir)

        canonical_summary_path = (
            run_dir / "canonicalization_summary.json"
        )
        canonical_fresh = False
        canonical_count = 0
        canonical_stale = False

        if canonical_summary_path.is_file():
            try:
                canonical_payload = json.loads(
                    canonical_summary_path.read_text()
                )
            except Exception:
                canonical_payload = {}

            expected_review_hash = (
                canonical_payload
                .get("source_hashes", {})
                .get("cell_review.csv")
            )
            current_review_hash = _sha256_file(
                run_dir / "cell_review.csv"
            )
            canonical_fresh = bool(
                canonical_payload.get("finalized", False)
                and expected_review_hash
                and expected_review_hash == current_review_hash
                and _exists(run_dir, "canonical_cells.csv")
                and _exists(
                    run_dir,
                    "canonical_cell_lineage.csv",
                )
            )
            canonical_stale = bool(
                canonical_payload.get("finalized", False)
                and not canonical_fresh
            )
            canonical_count = int(
                canonical_payload.get(
                    "canonical_cell_count",
                    0,
                )
            )

        if canonical_fresh:
            return (
                "ready",
                (
                    f"Human review is resolved and {canonical_count} "
                    "canonical cell(s) are finalized with current lineage."
                ),
            )

        if consensus and review["review_complete"]:
            note = (
                "Human review is complete, but canonical cells are "
                "stale and must be re-finalized."
                if canonical_stale
                else (
                    "Human review is complete, but canonical cells are "
                    "not finalized. Run Change 15 canonicalization."
                )
            )
            return ("partial", note)

        if consensus:
            return (
                "partial",
                (
                    "Automated consensus exists; human review is still "
                    "incomplete."
                ),
            )
        return (
            "planned",
            "Multichannel cell consensus is not available.",
        )

    if sid == "element_relationships":
        if _exists(run_dir, "element_pairs.csv"):
            return (
                "partial",
                (
                    "Pre-canonical element-pair evidence exists, but final "
                    "relationships must be recomputed on canonical cells."
                ),
            )
        return ("planned", "Canonical element relationships are pending.")

    if sid == "methods_provenance":
        return (
            ("ready" if all_present else "partial"),
            (
                "Core study configuration and provenance are available."
                if all_present
                else "Core provenance inputs are incomplete."
            ),
        )

    return (
        ("ready" if all_present else "planned"),
        (
            "Required final artifacts are present."
            if all_present
            else (
                f"Planned downstream analysis: "
                f"Change {section['planned_change']}."
            )
        ),
    )


def _skeleton_markdown(manifest: dict) -> str:
    lines = [
        "# SOURDOUGH Study Report",
        "",
        "> Report contract / skeleton only. Final scientific report "
        "generation comes later.",
        "",
        "## Scientific reporting rules",
        "",
        "- Every reported claim must trace to a saved table, mask, map, "
        "figure, or provenance artifact.",
        "- Automated consensus is not canonical truth until the review/"
        "canonicalization boundary is complete.",
        "- Current XRF measurements are 2D projections; volume claims must "
        "be explicitly model-based.",
        "- Organelle labels are reported as evidence-based likeness, not "
        "definitive identity.",
        "- Physical length/area units are not invented when the coordinate "
        "unit has not been verified.",
        "",
    ]

    for section in manifest["sections"]:
        lines.extend(
            [
                f"## {section['title']}",
                "",
                f"**Status:** {section['status']}",
                "",
                section["description"],
                "",
                f"**Current note:** {section['status_note']}",
                "",
                "**Required final artifacts:**",
                "",
            ]
        )
        for item in section["required_final"]:
            marker = "x" if item in section["available_inputs"] else " "
            lines.append(f"- [{marker}] `{item}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def refresh_report_contract(run_dir) -> dict:
    '''Create/update the report folder, manifest, status table and skeleton.'''
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Study run does not exist: {run_dir}")

    report_dir = run_dir / "report"
    for subdir in (
        report_dir,
        report_dir / "figures",
        report_dir / "tables",
        report_dir / "cells",
        report_dir / "samples",
        report_dir / "conditions",
        report_dir / "appendix",
    ):
        subdir.mkdir(parents=True, exist_ok=True)

    sections = []
    for definition in REPORT_SECTIONS:
        status, note = _section_status(
            run_dir,
            definition,
        )
        required = list(definition["required_final"])
        available = [
            path
            for path in required
            if _exists(run_dir, path)
        ]
        missing = [
            path
            for path in required
            if path not in available
        ]
        sections.append(
            {
                **definition,
                "required_final": required,
                "status": status,
                "status_note": note,
                "available_inputs": available,
                "missing_final_inputs": missing,
            }
        )

    review = _review_state(run_dir)

    final_ready = all(
        section["status"] == "ready"
        for section in sections
    )

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "study_run": run_dir.name,
        "study_run_path": str(run_dir),
        "report_stage": "contract_and_skeleton",
        "final_report_ready": final_ready,
        "review_state": review,
        "report_policy": {
            "artifact_grounded_claims_only": True,
            "canonical_cells_required_for_final_cell_analysis": True,
            "raw_hdf5_modified": False,
            "unverified_physical_units_forbidden": True,
            "2d_xrf_volume_claims_require_model_label": True,
            "organelle_output_is_likeness_not_identity": True,
            "automated_scores_are_not_probabilities_unless_calibrated": True,
        },
        "planned_final_deliverables": [
            "SOURDOUGH_Report.html",
            "SOURDOUGH_Report.pdf",
            "SOURDOUGH_Data.xlsx",
            "SOURDOUGH_Data/",
            "publication_ready_figures/",
        ],
        "sections": sections,
    }

    manifest_path = report_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        )
    )

    status_rows = []
    for section in sections:
        status_rows.append(
            {
                "section_id": section["id"],
                "title": section["title"],
                "status": section["status"],
                "status_note": section["status_note"],
                "planned_change": section["planned_change"],
                "available_inputs": ";".join(
                    section["available_inputs"]
                ),
                "missing_final_inputs": ";".join(
                    section["missing_final_inputs"]
                ),
            }
        )
    pd.DataFrame(status_rows).to_csv(
        report_dir / "report_status.csv",
        index=False,
    )

    (report_dir / "REPORT_SKELETON.md").write_text(
        _skeleton_markdown(manifest)
    )

    (report_dir / "README.md").write_text(
        "# SOURDOUGH report workspace\n\n"
        "This folder is the stable destination for collaborator-facing "
        "report artifacts.\n\n"
        "Subfolders:\n\n"
        "- `figures/` publication/report figures\n"
        "- `tables/` report-specific tables\n"
        "- `cells/` per-cell report panels/assets\n"
        "- `samples/` sample/replicate summaries\n"
        "- `conditions/` condition/mutant/treatment summaries\n"
        "- `appendix/` QC, methods, and provenance assets\n\n"
        "The report contract is intentionally created before the final "
        "report so later scientific changes populate one stable schema.\n"
    )

    return manifest
