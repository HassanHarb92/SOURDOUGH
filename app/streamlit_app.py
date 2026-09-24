"""Interactive Yeast XRF Explorer.

Important architecture rule:
    This UI uses only the canonical XRFScan API. HDF5 storage paths belong in the IO layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from yeast_xrf.analysis.beam_quantification import (
    apply_explicit_areal_density_conversion,
    beam_normalize,
    normalization_comparison,
    summarize_full_cells,
    transmission_diagnostic,
)
from yeast_xrf.io.maps_quantification import inspect_maps_quantification
from yeast_xrf.analysis.batch import analyze_directory
from yeast_xrf.ui.olive_shell import SHELL_HTML

from yeast_xrf.io.channel_roles import ChannelRole
from yeast_xrf.io.xrf_scan import XRFScan
from yeast_xrf.analysis.normalization import (
    invalid_reason_counts,
    normalize_by_reference,
    positive_reference_percentile_floor,
)
from yeast_xrf.analysis.qc import (
    build_qc_flags,
    count_rate_diagnostics,
    live_time_diagnostics,
)
from yeast_xrf.features.background import background_subtracted, gaussian_background
from yeast_xrf.features.intensity import asinh_feature, robust_zscore_feature
from yeast_xrf.features.local_stats import (
    local_contrast_z,
    local_cv,
    local_iqr,
    local_mad,
    local_mean,
    local_median,
    local_std,
)
from yeast_xrf.features.gradients import (
    coordinate_gradient,
    directional_derivative,
    edge_mask,
    pixel_operator_gradient,
)
from yeast_xrf.features.hessian import coordinate_hessian
from yeast_xrf.visualization.landscape import (
    build_landscape_height,
    display_surface_normals,
    hessian_point_class,
    landscape_figure,
)
from yeast_xrf.features.cells import segment_cells_tfy
from yeast_xrf.visualization.cell_model import (
    build_mask_conforming_envelope,
    cell_model_figure,
    internal_slice,
    projection_conserving_density,
)
from yeast_xrf.visualization.plotting import (
    heatmap_figure,
    histogram_figure,
    spectrum_figure,
    surface_figure,
)
from yeast_xrf.visualization.statistics import finite_shared_range, map_statistics
from yeast_xrf.visualization.transforms import display_transform


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "img.dat"

st.set_page_config(
    page_title="SOURDOUGH",
    page_icon="🔬",
    layout="wide",
)

st.markdown(SHELL_HTML, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_scan_metadata(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    with XRFScan.open(path) as scan:
        return {
            "summary": scan.summary(),
            "channels": {
                method: [
                    {
                        "index": info.index,
                        "name": info.name,
                        "unit": info.unit,
                        "role": info.role.value,
                    }
                    for info in scan.channels(method)
                ]
                for method in ("Fitted", "NNLS", "ROI")
            },
            "primary_scalers": list(scan.scaler_names("primary")),
            "legacy_scalers": list(scan.scaler_names("legacy")),
            "extra_pvs": list(scan.extra_pvs()),
            "x": scan.x,
            "y": scan.y,
        }


@st.cache_data(show_spinner=False)
def load_map(path_text: str, method: str, channel: str) -> np.ndarray:
    with XRFScan.open(Path(path_text)) as scan:
        return scan.map(channel, method)


@st.cache_data(show_spinner=False)
def load_spectrum(
    path_text: str,
    y_index: int,
    x_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    with XRFScan.open(Path(path_text)) as scan:
        return scan.energy, scan.spectrum(y=y_index, x=x_index)


@st.cache_data(show_spinner=False)
def load_scaler(
    path_text: str,
    family: str,
    name: str,
) -> np.ndarray:
    with XRFScan.open(Path(path_text)) as scan:
        return scan.scaler(name, family)  # type: ignore[arg-type]


files = sorted(DATA_DIR.glob("*.h5"))
if not files:
    st.error(f"No HDF5 files found under {DATA_DIR}")
    st.stop()

# ---------------------------------------------------------------------------
# SOURDOUGH study-first front door
# Change 11B: ANALYZE / RESULTS / EXPLORE
# ---------------------------------------------------------------------------
from pathlib import Path as _SDPath
import json as _sd_json
import pandas as _sd_pd

from yeast_xrf.analysis.study_analysis import (
    analyze_study as _sd_analyze_study,
    discover_scans as _sd_discover_scans,
)


_SD_ROOT = _SDPath(__file__).resolve().parents[1]
_SD_STUDIES = _SD_ROOT / "analysis" / "studies"

st.markdown(
    """
    <style>
    .sd-front-hero{
        border:1px solid rgba(49,51,63,.14);
        border-radius:20px;
        padding:1.25rem 1.35rem 1.15rem 1.35rem;
        background:linear-gradient(180deg,#fbfaf7 0%,#f5f3ed 100%);
        margin:0.55rem 0 1rem 0;
    }
    .sd-front-kicker{
        font-size:.72rem;
        text-transform:uppercase;
        letter-spacing:.09em;
        font-weight:700;
        color:#806d58;
        margin-bottom:.25rem;
    }
    .sd-front-title{
        font-size:1.65rem;
        font-weight:760;
        color:#202020;
        line-height:1.15;
        margin-bottom:.3rem;
    }
    .sd-front-copy{
        font-size:.96rem;
        color:#5c554c;
        max-width:850px;
    }
    .sd-front-card{
        border:1px solid rgba(49,51,63,.12);
        border-radius:16px;
        padding:1rem 1.05rem;
        background:#faf9f6;
        margin-bottom:.8rem;
    }
    .sd-front-label{
        font-size:.72rem;
        text-transform:uppercase;
        letter-spacing:.08em;
        font-weight:700;
        color:#88745e;
        margin-bottom:.18rem;
    }
    .sd-front-card-title{
        font-size:1.08rem;
        font-weight:720;
        color:#242424;
        margin-bottom:.15rem;
    }
    .sd-front-muted{
        color:#70685e;
        font-size:.88rem;
    }
    .sd-front-good{
        border:1px solid rgba(50,105,70,.20);
        border-radius:16px;
        padding:.9rem 1rem;
        background:rgba(238,247,240,.78);
        margin:.65rem 0;
    }
    .sd-front-warning{
        border:1px solid rgba(150,110,35,.22);
        border-radius:16px;
        padding:.9rem 1rem;
        background:rgba(250,246,232,.80);
        margin:.65rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_sd_mode = st.radio(
    "SOURDOUGH workspace",
    ["ANALYZE", "RESULTS", "EXPLORE"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="sourdough_primary_workspace",
)

if _sd_mode == "ANALYZE":
    st.markdown(
        """
        <div class="sd-front-hero">
          <div class="sd-front-kicker">Automated study analysis</div>
          <div class="sd-front-title">Directory in. Complete cellular XRF study out.</div>
          <div class="sd-front-copy">
            Point SOURDOUGH at a study folder. The automated pipeline discovers
            scans, performs validated MAPS quantification, detects cells, builds
            cell × element datasets, computes element-pair relationships, records
            QC and provenance, and writes comparison-ready study tables.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _sd_left, _sd_right = st.columns([1.2, 1.0], gap="large")

    with _sd_left:
        st.markdown(
            """
            <div class="sd-front-card">
              <div class="sd-front-label">1 · Study source</div>
              <div class="sd-front-card-title">Choose the study directory</div>
              <div class="sd-front-muted">
                HDF5 files are discovered recursively. Raw files remain read-only.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _sd_input = st.text_input(
            "Study directory",
            value=str(_SD_ROOT / "img.dat"),
            key="study_analysis_input_dir",
        )
        _sd_metadata = st.text_input(
            "Optional metadata CSV",
            value="",
            placeholder=(
                "scan, sample_id, condition, strain_or_mutant, "
                "treatment, biological_replicate, ..."
            ),
            key="study_analysis_metadata_csv",
        )

        _sd_output_root = st.text_input(
            "Study results directory",
            value=str(_SD_STUDIES),
            key="study_analysis_output_root",
        )

        try:
            _sd_discovered = _sd_discover_scans(_sd_input)
            st.success(
                f"Found {len(_sd_discovered)} HDF5 scan(s) ready for study analysis."
            )
            with st.expander("Discovered scans", expanded=False):
                st.dataframe(
                    _sd_pd.DataFrame(
                        {
                            "scan": [p.name for p in _sd_discovered],
                            "path": [str(p) for p in _sd_discovered],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as _sd_exc:
            _sd_discovered = []
            st.warning(str(_sd_exc))

    with _sd_right:
        st.markdown(
            """
            <div class="sd-front-card">
              <div class="sd-front-label">2 · Automatic analysis contract</div>
              <div class="sd-front-card-title">Full analysis is the default</div>
              <div class="sd-front-muted">
                No manual tab-by-tab workflow is required. Expert tools remain
                available under EXPLORE.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **Automatic pipeline**

            ✓ recursive scan discovery  
            ✓ Fitted + NNLS + ROI kept separate  
            ✓ US_IC + DS_IC MAPS concentration  
            ✓ every quantifiable element  
            ✓ provisional full/cropped cell census  
            ✓ every detected cell × every element  
            ✓ whole-scan concentration statistics  
            ✓ within-cell element-pair correlations  
            ✓ concentration arrays + comparison tables  
            ✓ artifact / contamination candidate screening  
            ✓ QC flags + full provenance
            """
        )

        st.markdown(
            """
            <div class="sd-front-warning">
              <strong>Current cell masks are provisional.</strong><br>
              Change 11A uses TFY-based cell detection. SOURDOUGH does not yet
              claim these as canonical multichannel-consensus cells. Artifact
              screening is now flag-only and non-destructive; multichannel cell
              consensus comes next.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Advanced automated-analysis settings", expanded=False):
        _sd_c1, _sd_c2 = st.columns(2)
        with _sd_c1:
            _sd_methods = st.multiselect(
                "Analysis products",
                ["Fitted", "NNLS", "ROI"],
                default=["Fitted", "NNLS", "ROI"],
                key="study_analysis_methods",
            )
        with _sd_c2:
            _sd_refs = st.multiselect(
                "Quantitative references",
                ["US_IC", "DS_IC"],
                default=["US_IC", "DS_IC"],
                key="study_analysis_refs",
            )

        _sd_a1, _sd_a2, _sd_a3 = st.columns(3)
        with _sd_a1:
            _sd_floor = st.slider(
                "Low-scaler mask (%)",
                0.0,
                10.0,
                1.0,
                0.25,
                key="study_analysis_floor",
            )
        with _sd_a2:
            _sd_arrays = st.checkbox(
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

    _sd_ready = bool(_sd_discovered and _sd_methods and _sd_refs)

    if st.button(
        "Analyze entire study",
        type="primary",
        use_container_width=True,
        disabled=not _sd_ready,
        key="study_analysis_run_button",
    ):
        try:
            with st.spinner(
                "SOURDOUGH is analyzing every scan, cell, and quantifiable element..."
            ):
                _sd_result = _sd_analyze_study(
                    _sd_input,
                    output_root=_sd_output_root,
                    metadata_csv=(
                        _sd_metadata.strip() or None
                    ),
                    methods=tuple(_sd_methods),
                    references=tuple(_sd_refs),
                    floor_percentile=float(_sd_floor),
                    save_concentration_arrays=bool(_sd_arrays),
                    compute_element_pairs=bool(_sd_pairs),
                    artifact_screening=bool(
                        _sd_artifact_screening
                    ),
                )

            _sd_summary = _sd_result["summary"]
            st.session_state["sourdough_last_study_run"] = str(
                _sd_result["run_dir"]
            )

            st.markdown(
                """
                <div class="sd-front-good">
                  <strong>Study analysis complete.</strong><br>
                  The complete study dataset is now available under RESULTS.
                </div>
                """,
                unsafe_allow_html=True,
            )

            _sd_m = st.columns(6)
            _sd_m[0].metric(
                "Scans",
                _sd_summary["scans_analyzed"],
            )
            _sd_m[1].metric(
                "Cells",
                _sd_summary["detected_cells"],
            )
            _sd_m[2].metric(
                "Full cells",
                _sd_summary["full_cells"],
            )
            _sd_m[3].metric(
                "Cell × element",
                _sd_summary["cell_element_records"],
            )
            _sd_m[4].metric(
                "Element pairs",
                _sd_summary["element_pair_records"],
            )
            _sd_m[5].metric(
                "Failures",
                _sd_summary["scan_failures"],
            )

            st.code(str(_sd_result["run_dir"]), language=None)

        except Exception as _sd_exc:
            st.exception(_sd_exc)

    st.stop()


if _sd_mode == "RESULTS":
    st.markdown(
        """
        <div class="sd-front-hero">
          <div class="sd-front-kicker">Study results</div>
          <div class="sd-front-title">Cells, elements, relationships, and provenance.</div>
          <div class="sd-front-copy">
            Browse automated study outputs without opening individual analysis
            tools. EXPLORE remains available when you want to inspect a scan
            manually.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _SD_STUDIES.mkdir(parents=True, exist_ok=True)
    _sd_runs = sorted(
        [
            p
            for p in _SD_STUDIES.iterdir()
            if p.is_dir() and (p / "study_summary.json").is_file()
        ],
        reverse=True,
    )

    if not _sd_runs:
        st.info(
            "No completed automated study run was found yet. "
            "Choose ANALYZE and run a study first."
        )
        st.stop()

    _sd_preferred = st.session_state.get("sourdough_last_study_run")
    _sd_default_idx = 0
    if _sd_preferred:
        for _sd_i, _sd_path in enumerate(_sd_runs):
            if str(_sd_path) == str(_sd_preferred):
                _sd_default_idx = _sd_i
                break

    _sd_run = st.selectbox(
        "Study run",
        _sd_runs,
        index=_sd_default_idx,
        format_func=lambda p: p.name,
        key="study_results_run",
    )

    _sd_summary = _sd_json.loads(
        (_sd_run / "study_summary.json").read_text()
    )

    _sd_metrics = st.columns(7)
    _sd_metrics[0].metric("Scans", _sd_summary.get("scans_analyzed", 0))
    _sd_metrics[1].metric("Cells", _sd_summary.get("detected_cells", 0))
    _sd_metrics[2].metric("Full", _sd_summary.get("full_cells", 0))
    _sd_metrics[3].metric("Cropped", _sd_summary.get("cropped_cells", 0))
    _sd_metrics[4].metric(
        "Cell × element",
        _sd_summary.get("cell_element_records", 0),
    )
    _sd_metrics[5].metric(
        "Element pairs",
        _sd_summary.get("element_pair_records", 0),
    )
    _sd_metrics[6].metric(
        "Failures",
        _sd_summary.get("scan_failures", 0),
    )

    st.markdown(
        """
        <div class="sd-front-warning">
          <strong>Cell masks are currently provisional TFY-based masks.</strong>
          Results are quantitative, but cell delineation will be replaced by
          the upcoming multichannel-consensus engine before these become the
          canonical biological cell dataset. Artifact candidates are retained
          and flagged for review rather than automatically removed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _sd_read_table(_name):
        _path = _sd_run / f"{_name}.csv"
        if not _path.is_file():
            return _sd_pd.DataFrame()

        # Optional study tables such as failures.csv or qc_flags.csv may
        # legitimately contain no rows. Older Change-11A runs wrote a
        # completely empty (0-byte) CSV in that case. Treat it as an empty
        # table instead of allowing pandas.EmptyDataError to break RESULTS.
        if _path.stat().st_size == 0:
            return _sd_pd.DataFrame()

        try:
            return _sd_pd.read_csv(_path)
        except _sd_pd.errors.EmptyDataError:
            return _sd_pd.DataFrame()

    _sd_scans = _sd_read_table("scans")
    _sd_cells = _sd_read_table("cells")
    _sd_ce = _sd_read_table("cell_elements")
    _sd_pairs_df = _sd_read_table("element_pairs")
    _sd_artifacts = _sd_read_table("artifact_candidates")
    _sd_qc = _sd_read_table("qc_flags")
    _sd_failures = _sd_read_table("failures")

    _sd_tabs = st.tabs(
        [
            "Overview",
            "Cells",
            "Cell × Element",
            "Element Pairs",
            "QC & Failures",
            "Files & Provenance",
        ]
    )

    with _sd_tabs[0]:
        if not _sd_scans.empty:
            st.markdown("#### Scan census")
            st.dataframe(
                _sd_scans,
                use_container_width=True,
                hide_index=True,
            )

        if not _sd_ce.empty and {
            "element",
            "mean",
            "analysis_population",
        }.issubset(_sd_ce.columns):
            _sd_primary = _sd_ce[
                _sd_ce["analysis_population"].astype(str).str.lower().isin(
                    ["true", "1"]
                )
            ].copy()

            if not _sd_primary.empty:
                _sd_summary_table = (
                    _sd_primary.groupby(
                        ["element", "method", "reference"],
                        dropna=False,
                    )["mean"]
                    .agg(["count", "mean", "median", "std"])
                    .reset_index()
                )
                st.markdown("#### Study-level concentration summary")
                st.dataframe(
                    _sd_summary_table,
                    use_container_width=True,
                    hide_index=True,
                )

    with _sd_tabs[1]:
        if _sd_cells.empty:
            st.info("No cell table is available for this run.")
        else:
            _sd_filter_cols = st.columns(3)
            _sd_scan_filter = _sd_filter_cols[0].multiselect(
                "Scan",
                sorted(_sd_cells["scan"].dropna().astype(str).unique())
                if "scan" in _sd_cells
                else [],
                key="results_cells_scan_filter",
            )
            _sd_condition_filter = _sd_filter_cols[1].multiselect(
                "Condition",
                sorted(
                    x
                    for x in _sd_cells.get(
                        "condition",
                        _sd_pd.Series(dtype=str),
                    ).dropna().astype(str).unique()
                    if x
                ),
                key="results_cells_condition_filter",
            )
            _sd_population = _sd_filter_cols[2].selectbox(
                "Population",
                ["Full cells", "All detected cells", "Cropped cells"],
                key="results_cells_population",
            )

            _sd_view = _sd_cells.copy()
            if _sd_scan_filter:
                _sd_view = _sd_view[
                    _sd_view["scan"].astype(str).isin(_sd_scan_filter)
                ]
            if _sd_condition_filter and "condition" in _sd_view:
                _sd_view = _sd_view[
                    _sd_view["condition"].astype(str).isin(
                        _sd_condition_filter
                    )
                ]
            if "cropped" in _sd_view:
                _sd_cropped_bool = (
                    _sd_view["cropped"].astype(str).str.lower()
                    .isin(["true", "1"])
                )
                if _sd_population == "Full cells":
                    _sd_view = _sd_view[~_sd_cropped_bool]
                elif _sd_population == "Cropped cells":
                    _sd_view = _sd_view[_sd_cropped_bool]

            st.dataframe(
                _sd_view,
                use_container_width=True,
                hide_index=True,
            )

    with _sd_tabs[2]:
        if _sd_ce.empty:
            st.info("No cell × element table is available for this run.")
        else:
            _sd_filter = st.columns(4)
            _sd_elements = sorted(
                _sd_ce["element"].dropna().astype(str).unique()
            )
            _sd_methods_available = sorted(
                _sd_ce["method"].dropna().astype(str).unique()
            )
            _sd_refs_available = sorted(
                _sd_ce["reference"].dropna().astype(str).unique()
            )

            _sd_element_sel = _sd_filter[0].multiselect(
                "Element",
                _sd_elements,
                key="results_ce_element",
            )
            _sd_method_sel = _sd_filter[1].multiselect(
                "Method",
                _sd_methods_available,
                default=(
                    ["Fitted"]
                    if "Fitted" in _sd_methods_available
                    else []
                ),
                key="results_ce_method",
            )
            _sd_ref_sel = _sd_filter[2].multiselect(
                "Reference",
                _sd_refs_available,
                default=(
                    ["US_IC"]
                    if "US_IC" in _sd_refs_available
                    else []
                ),
                key="results_ce_ref",
            )
            _sd_full_only = _sd_filter[3].checkbox(
                "Full cells only",
                value=True,
                key="results_ce_full_only",
            )

            _sd_view = _sd_ce.copy()
            if _sd_element_sel:
                _sd_view = _sd_view[
                    _sd_view["element"].astype(str).isin(_sd_element_sel)
                ]
            if _sd_method_sel:
                _sd_view = _sd_view[
                    _sd_view["method"].astype(str).isin(_sd_method_sel)
                ]
            if _sd_ref_sel:
                _sd_view = _sd_view[
                    _sd_view["reference"].astype(str).isin(_sd_ref_sel)
                ]
            if _sd_full_only and "analysis_population" in _sd_view:
                _sd_view = _sd_view[
                    _sd_view["analysis_population"]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1"])
                ]

            st.dataframe(
                _sd_view,
                use_container_width=True,
                hide_index=True,
            )

    with _sd_tabs[3]:
        if _sd_pairs_df.empty:
            st.info("Element-pair relationships were not generated.")
        else:
            st.caption(
                "Current relationship metric: within-cell pixelwise Pearson "
                "correlation between elemental concentration maps."
            )
            st.dataframe(
                _sd_pairs_df,
                use_container_width=True,
                hide_index=True,
            )

    with _sd_tabs[4]:
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
            st.dataframe(
                _sd_qc,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No QC flags were written for this study run.")

        if not _sd_failures.empty:
            st.markdown("#### Failures")
            st.dataframe(
                _sd_failures,
                use_container_width=True,
                hide_index=True,
            )

    with _sd_tabs[5]:
        st.markdown("#### Study run directory")
        st.code(str(_sd_run), language=None)

        for _sd_file in (
            "study_manifest.csv",
            "study_summary.json",
            "study_provenance.json",
            "scans.csv",
            "cells.csv",
            "cell_elements.csv",
            "scan_elements.csv",
            "element_pairs.csv",
            "artifact_candidates.csv",
            "qc_flags.csv",
            "failures.csv",
        ):
            _sd_path = _sd_run / _sd_file
            if _sd_path.is_file():
                st.write(f"✓ `{_sd_file}`")

        _sd_prov = _sd_run / "study_provenance.json"
        if _sd_prov.is_file():
            with st.expander("Study provenance", expanded=False):
                st.json(_sd_json.loads(_sd_prov.read_text()))

    st.stop()


# EXPLORE intentionally falls through into the pre-existing SOURDOUGH
# scientific workspace below. No existing exploration capability is moved,
# renamed, or scientifically altered by Change 11B.

with st.container(border=True):
    st.markdown("#### Study controls")
    control_scan, control_method, control_context = st.columns([2.2, 1.0, 1.8])
    with control_scan:
        selected_file = st.selectbox(
            "Scan",
            files,
            format_func=lambda p: p.name,
            key="global_scan_selector",
        )
    with control_method:
        method = st.selectbox(
            "Analyzed product",
            ["Fitted", "NNLS", "ROI"],
            key="global_analysis_method",
        )
    with control_context:
        st.markdown("**Analysis contract**")
        st.caption(
            "Fitted, NNLS, and ROI stay separate. No method is silently treated "
            "as scientifically preferred."
        )

path_text = str(selected_file)
meta = load_scan_metadata(path_text)
summary = meta["summary"]
channels = meta["channels"][method]
x_axis = np.asarray(meta["x"], dtype=float)
y_axis = np.asarray(meta["y"], dtype=float)

channel_lookup = {row["name"]: row for row in channels}
channel_names = list(channel_lookup)

with st.container(border=True):
    st.markdown("#### Dataset at a glance")
    overview_metrics = st.columns(4)
    overview_metrics[0].metric(
        "Raster",
        f"{summary['geometry']['shape_yx'][0]} × {summary['geometry']['shape_yx'][1]}",
    )
    overview_metrics[1].metric(
        "Analyzed channels",
        summary["methods"][method]["channel_count"],
    )
    overview_metrics[2].metric(
        "Energy channels",
        summary["spectra"]["energy_count"],
    )
    theta_value = summary["scan"]["theta"]
    overview_metrics[3].metric(
        "Theta",
        "n/a" if theta_value is None else f"{theta_value:.6g}",
    )

st.caption(
    f"**{selected_file.name}** · {method} · exact stored X/Y coordinates · "
    "raw data remain read-only"
)


def channel_label(name: str) -> str:
    row = channel_lookup[name]
    unit = row["unit"] or "unit blank"
    return f"{name}  ·  {row['role']}  ·  {unit}"


def transform_controls(prefix: str) -> tuple[str, float, float, float]:
    mode = st.selectbox(
        "Display transform",
        ["Raw", "Percentile stretch", "Signed log1p", "Gamma"],
        key=f"{prefix}_transform",
    )
    low = 1.0
    high = 99.0
    gamma = 0.7
    if mode in {"Percentile stretch", "Gamma"}:
        c1, c2 = st.columns(2)
        low = c1.number_input(
            "Low percentile",
            min_value=0.0,
            max_value=99.0,
            value=1.0,
            step=0.5,
            key=f"{prefix}_low",
        )
        high = c2.number_input(
            "High percentile",
            min_value=1.0,
            max_value=100.0,
            value=99.0,
            step=0.5,
            key=f"{prefix}_high",
        )
        if high <= low:
            st.warning("High percentile must be greater than low percentile.")
            high = min(100.0, low + 1.0)
    if mode == "Gamma":
        gamma = st.slider(
            "Gamma",
            min_value=0.1,
            max_value=3.0,
            value=0.7,
            step=0.05,
            key=f"{prefix}_gamma",
        )
    return mode, float(low), float(high), float(gamma)


def metric_text(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


st.markdown("#### Scientific workspace")
st.caption(
    "Move from observation → QC → mathematical features → geometry → cell analysis."
)

tabs = st.tabs(
    [
        "🗺️ Map Explorer",
        "📈 Spectrum Inspector",
        "🧭 Scaler & QC",
        "⚖️ Compare",
        "🧾 Acquisition",
        "🧪 QC & Normalize",
        "🌓 Intensity & Contrast",
        "∇ Gradients & Edges",
        "∇² Hessian & Curvature",
        "🧬 2.5D Cell Landscape",
        "🧫 Cell Analyzer & 3D Model",
        "📚 Batch Study",
        "⚖️ Beam & Quantification",
    ]
)

with tabs[0]:
    top_left, top_right = st.columns([2, 1])
    with top_left:
        selected_channel = st.selectbox(
            "Channel",
            channel_names,
            format_func=channel_label,
            key="map_channel",
        )
    with top_right:
        colorscale = st.selectbox(
            "Colormap",
            ["Viridis", "Cividis", "Inferno", "Magma", "Plasma", "Turbo"],
            key="map_colorscale",
        )

    control_col, info_col = st.columns([1, 2])
    with control_col:
        mode, low, high, gamma = transform_controls("map")
    with info_col:
        info = channel_lookup[selected_channel]
        m1, m2, m3 = st.columns(3)
        m1.metric("Role", info["role"])
        m2.metric("Unit", info["unit"] or "blank")
        m3.metric("Method", method)

    raw_map = load_map(path_text, method, selected_channel)
    transformed = display_transform(
        raw_map,
        mode,
        low=low,
        high=high,
        gamma=gamma,
    )

    show_original = st.checkbox(
        "Show original raw map beside transformed view",
        value=True,
        key="show_original_raw",
        help=(
            "The original panel shows the selected HDF5 channel values with no "
            "percentile, log, or gamma transform applied."
        ),
    )

    if show_original and mode != "Raw":
        raw_col, display_col = st.columns(2)

        with raw_col:
            st.markdown("#### Original / raw")
            st.plotly_chart(
                heatmap_figure(
                    raw_map,
                    x_axis,
                    y_axis,
                    title=f"{selected_channel} · {method} · Original raw map",
                    colorscale=colorscale,
                    colorbar_title=info["unit"] or "raw value",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Direct values from the selected analyzed channel. "
                "No percentile stretch, log transform, gamma transform, or clipping "
                "has been applied by the Explorer."
            )

        with display_col:
            st.markdown("#### Current display")
            st.plotly_chart(
                heatmap_figure(
                    transformed.data,
                    x_axis,
                    y_axis,
                    title=f"{selected_channel} · {method} · {transformed.label}",
                    colorscale=colorscale,
                    colorbar_title="display value",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Visualization-only representation. The source scientific array remains "
                "unchanged."
            )
    else:
        st.markdown("#### Original / raw")
        st.plotly_chart(
            heatmap_figure(
                raw_map,
                x_axis,
                y_axis,
                title=f"{selected_channel} · {method} · Original raw map",
                colorscale=colorscale,
                colorbar_title=info["unit"] or "raw value",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
        if mode == "Raw":
            st.caption(
                "The current display mode is Raw, so this is also the untransformed "
                "source-channel view."
            )

    st.caption(
        "The heatmaps use the exact stored X and Y coordinate arrays. "
        "Coordinate units have not yet been established by metadata, so no physical "
        "unit label is invented."
    )

    with st.expander("Scan overview / original appearance", expanded=False):
        overview_channel = (
            "Total_Fluorescence_Yield"
            if "Total_Fluorescence_Yield" in channel_names
            else selected_channel
        )
        overview_info = channel_lookup[overview_channel]
        overview_raw = load_map(path_text, method, overview_channel)

        st.markdown(
            f"**Reference channel:** `{overview_channel}`  "
            f"({overview_info['role']}; {overview_info['unit'] or 'unit blank'})"
        )
        st.caption(
            "Total Fluorescence Yield is used as the overview when available because it "
            "provides a broad scan-intensity reference. This is still an XRF-derived map, "
            "not an optical microscopy photograph."
            if overview_channel == "Total_Fluorescence_Yield"
            else
            "Total Fluorescence Yield was not available, so the selected raw channel is "
            "shown as the scan reference."
        )

        overview_left, overview_right = st.columns(2)
        with overview_left:
            st.plotly_chart(
                heatmap_figure(
                    overview_raw,
                    x_axis,
                    y_axis,
                    title=f"{overview_channel} · raw reference",
                    colorscale="Cividis",
                    colorbar_title=overview_info["unit"] or "raw value",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )

        with overview_right:
            overview_display = display_transform(
                overview_raw,
                "Percentile stretch",
                low=1.0,
                high=99.0,
            )
            st.plotly_chart(
                heatmap_figure(
                    overview_display.data,
                    x_axis,
                    y_axis,
                    title=f"{overview_channel} · 1–99% reference display",
                    colorscale="Cividis",
                    colorbar_title="display value",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Right panel is contrast-enhanced only to help reveal the scan outline; "
                "the left panel is the raw reference."
            )

    stats = map_statistics(raw_map)
    st.subheader("Raw-map statistics")
    metric_cols = st.columns(6)
    for col, key, label in zip(
        metric_cols,
        ["min", "max", "mean", "median", "std", "negative_count"],
        ["Min", "Max", "Mean", "Median", "Std", "Negative pixels"],
    ):
        col.metric(label, metric_text(stats[key]))

    with st.expander("Full statistics and histogram", expanded=False):
        left, right = st.columns([1, 2])
        with left:
            stat_table = pd.DataFrame(
                [{"metric": key, "value": value} for key, value in stats.items()]
            )
            st.dataframe(stat_table, use_container_width=True, hide_index=True)
        with right:
            st.plotly_chart(
                histogram_figure(raw_map, title=f"{selected_channel} raw-value distribution"),
                use_container_width=True,
                config={"displaylogo": False},
            )

    with st.expander("Interactive 2.5D surface preview", expanded=False):
        st.warning(
            "This is a 2.5D intensity surface. Height represents display intensity, "
            "not biological depth and not a reconstructed 3D yeast cell."
        )
        st.plotly_chart(
            surface_figure(
                transformed.data,
                x_axis,
                y_axis,
                title=f"2.5D surface · {selected_channel} · {transformed.label}",
                colorscale=colorscale,
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

with tabs[1]:
    st.subheader("Single-pixel XRF spectrum")
    st.caption(
        "Select a pixel by raster index. The plot reads only that one 2048-channel "
        "spectrum from the HDF5 spectral cube."
    )

    context_channel = st.selectbox(
        "Context map",
        channel_names,
        index=channel_names.index("Zn") if "Zn" in channel_names else 0,
        format_func=channel_label,
        key="spectrum_context",
    )

    y_size, x_size = summary["geometry"]["shape_yx"]
    selector_left, selector_right = st.columns(2)
    y_index = selector_left.slider(
        "Y index",
        min_value=0,
        max_value=int(y_size) - 1,
        value=(int(y_size) - 1) // 2,
        key="spectrum_y",
    )
    x_index = selector_right.slider(
        "X index",
        min_value=0,
        max_value=int(x_size) - 1,
        value=(int(x_size) - 1) // 2,
        key="spectrum_x",
    )

    x_coord = float(x_axis[x_index])
    y_coord = float(y_axis[y_index])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("X index", x_index)
    c2.metric("Y index", y_index)
    c3.metric("X coordinate", f"{x_coord:.6g}")
    c4.metric("Y coordinate", f"{y_coord:.6g}")

    context_map = load_map(path_text, method, context_channel)
    context_display = display_transform(context_map, "Percentile stretch")
    spectrum_energy, spectrum_values = load_spectrum(path_text, y_index, x_index)

    left, right = st.columns([1, 1.25])
    with left:
        st.plotly_chart(
            heatmap_figure(
                context_display.data,
                x_axis,
                y_axis,
                title=f"Selected pixel on {context_channel}",
                colorscale="Viridis",
                colorbar_title="display value",
                marker_xy=(x_coord, y_coord),
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with right:
        st.plotly_chart(
            spectrum_figure(
                spectrum_energy,
                spectrum_values,
                title=f"Pixel spectrum · ({y_index}, {x_index})",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

with tabs[2]:
    st.subheader("Acquisition scalers and fit diagnostics")

    scaler_col, qc_col = st.columns(2)

    with scaler_col:
        st.markdown("#### Scaler map")
        scaler_family = st.radio(
            "Scaler family",
            ["primary", "legacy"],
            horizontal=True,
            key="scaler_family",
        )
        scaler_names = (
            meta["primary_scalers"]
            if scaler_family == "primary"
            else meta["legacy_scalers"]
        )
        default_scaler = (
            scaler_names.index("US_IC") if "US_IC" in scaler_names else 0
        )
        selected_scaler = st.selectbox(
            "Scaler",
            scaler_names,
            index=default_scaler,
            key="selected_scaler",
        )
        scaler_map = load_scaler(path_text, scaler_family, selected_scaler)
        scaler_display = display_transform(scaler_map, "Percentile stretch")
        st.plotly_chart(
            heatmap_figure(
                scaler_display.data,
                x_axis,
                y_axis,
                title=f"{selected_scaler} · {scaler_family}",
                colorscale="Cividis",
                colorbar_title="display value",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with qc_col:
        st.markdown("#### XRF diagnostic channel")
        qc_candidates = [
            row
            for row in channels
            if row["role"]
            in {ChannelRole.FIT_DIAGNOSTIC.value, ChannelRole.SCATTER.value}
        ]
        if qc_candidates:
            qc_names = [row["name"] for row in qc_candidates]
            selected_qc = st.selectbox("Diagnostic", qc_names, key="qc_channel")
            qc_map = load_map(path_text, method, selected_qc)
            qc_display = display_transform(qc_map, "Percentile stretch")
            st.plotly_chart(
                heatmap_figure(
                    qc_display.data,
                    x_axis,
                    y_axis,
                    title=f"{selected_qc} · {method}",
                    colorscale="Magma",
                    colorbar_title="display value",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info("No diagnostic/scatter channels were classified for this method.")

    st.info(
        "These maps are shown for inspection only. Change 04 does not automatically "
        "normalize XRF channels by any scaler, livetime, dead time, or fit diagnostic."
    )

with tabs[3]:
    st.subheader("Side-by-side comparison")
    compare_mode = st.radio(
        "Comparison type",
        ["Channels within one method", "Same channel across methods"],
        horizontal=True,
    )
    compare_transform = st.selectbox(
        "Comparison display",
        ["Raw", "Percentile stretch", "Signed log1p"],
        key="compare_transform",
    )
    compare_colorscale = st.selectbox(
        "Comparison colormap",
        ["Viridis", "Cividis", "Inferno", "Magma", "Plasma", "Turbo"],
        key="compare_colorscale",
    )

    arrays: list[np.ndarray] = []
    labels: list[str] = []

    if compare_mode == "Channels within one method":
        preferred = [name for name in ("P", "Fe", "Zn") if name in channel_names]
        if len(preferred) < 2:
            preferred = channel_names[: min(3, len(channel_names))]
        selected_compare_channels = st.multiselect(
            "Channels",
            channel_names,
            default=preferred,
            max_selections=3,
        )
        for name in selected_compare_channels:
            arrays.append(load_map(path_text, method, name))
            labels.append(f"{name} · {method}")
    else:
        compare_channel = st.selectbox(
            "Channel",
            channel_names,
            index=channel_names.index("Zn") if "Zn" in channel_names else 0,
            key="compare_method_channel",
        )
        for compare_method in ("Fitted", "NNLS", "ROI"):
            arrays.append(load_map(path_text, compare_method, compare_channel))
            labels.append(f"{compare_channel} · {compare_method}")

    use_shared_raw = False
    shared_range = None
    if compare_transform == "Raw" and arrays:
        use_shared_raw = st.checkbox(
            "Use one shared raw-value color range",
            value=True,
        )
        if use_shared_raw:
            shared_range = finite_shared_range(arrays)

    if not arrays:
        st.info("Select at least one channel.")
    else:
        cols = st.columns(len(arrays))
        for col, array, label in zip(cols, arrays, labels):
            transformed = display_transform(array, compare_transform)
            zmin = shared_range[0] if shared_range is not None else None
            zmax = shared_range[1] if shared_range is not None else None
            with col:
                st.plotly_chart(
                    heatmap_figure(
                        transformed.data,
                        x_axis,
                        y_axis,
                        title=label,
                        colorscale=compare_colorscale,
                        colorbar_title=(
                            "raw value" if compare_transform == "Raw" else "display value"
                        ),
                        zmin=zmin,
                        zmax=zmax,
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                )

with tabs[4]:
    st.subheader("Acquisition and data-model metadata")

    geometry = summary["geometry"]
    spectral = summary["spectra"]
    scan_info = summary["scan"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raster", f"{geometry['shape_yx'][0]} × {geometry['shape_yx'][1]}")
    m2.metric("Analyzed channels", summary["methods"]["Fitted"]["channel_count"])
    m3.metric("Energy channels", spectral["energy_count"])
    m4.metric("Theta", metric_text(scan_info["theta"]))

    geom_df = pd.DataFrame(
        [
            {
                "axis": "X",
                "count": geometry["x_count"],
                "min": geometry["x_min"],
                "max": geometry["x_max"],
                "median step": geometry["dx_median_abs"],
                "uniform": geometry["x_uniform"],
                "unit": geometry["coordinate_unit"] or "unknown",
            },
            {
                "axis": "Y",
                "count": geometry["y_count"],
                "min": geometry["y_min"],
                "max": geometry["y_max"],
                "median step": geometry["dy_median_abs"],
                "uniform": geometry["y_uniform"],
                "unit": geometry["coordinate_unit"] or "unknown",
            },
        ]
    )
    st.markdown("#### Geometry")
    st.dataframe(geom_df, use_container_width=True, hide_index=True)

    st.markdown("#### Spectral axis")
    spectral_df = pd.DataFrame(
        [
            {
                "energy channels": spectral["energy_count"],
                "energy min": spectral["energy_min"],
                "energy max": spectral["energy_max"],
                "calibration coefficients": spectral["energy_calibration"],
            }
        ]
    )
    st.dataframe(spectral_df, use_container_width=True, hide_index=True)

    st.markdown("#### Channel catalog")
    channel_df = pd.DataFrame(channels)
    st.dataframe(channel_df, use_container_width=True, hide_index=True)

    st.markdown("#### Extra PVs")
    extra_pv_df = pd.DataFrame(meta["extra_pvs"])
    st.dataframe(extra_pv_df, use_container_width=True, hide_index=True)

    with st.expander("Canonical scan summary JSON"):
        st.json(summary)


with tabs[5]:
    st.subheader("QC + explicit normalization")
    st.caption(
        "Normalization is opt-in and never overwrites the raw XRF map. "
        "Every result records its reference, denominator floor, masks, and scale factor."
    )

    qc_channel = st.selectbox(
        "XRF channel",
        channel_names,
        index=channel_names.index("Fe") if "Fe" in channel_names else 0,
        format_func=channel_label,
        key="qc_norm_channel",
    )
    qc_raw = load_map(path_text, method, qc_channel)

    st.markdown("#### 1. Raw-map QC")
    qc_c1, qc_c2 = st.columns(2)
    with qc_c1:
        low_signal_text = st.text_input(
            "Optional low-signal threshold",
            value="",
            help="Leave blank to avoid defining a low-signal threshold.",
            key="qc_low_signal",
        )
    with qc_c2:
        high_tail = st.number_input(
            "High-value tail candidate percentile",
            min_value=90.0,
            max_value=99.999,
            value=99.9,
            step=0.1,
            key="qc_high_tail",
            help=(
                "Flags the upper tail for inspection only. It is not called saturation "
                "because detector saturation limits are not established here."
            ),
        )

    low_signal = None
    if low_signal_text.strip():
        try:
            low_signal = float(low_signal_text)
        except ValueError:
            st.error("Low-signal threshold must be numeric or blank.")

    qc_result = build_qc_flags(
        qc_raw,
        low_signal_threshold=low_signal,
        high_value_percentile=float(high_tail),
    )

    qc_metrics = st.columns(5)
    qc_metrics[0].metric("Nonfinite", qc_result.summary["nonfinite_count"])
    qc_metrics[1].metric("Negative", qc_result.summary["negative_count"])
    qc_metrics[2].metric("Zero", qc_result.summary["zero_count"])
    qc_metrics[3].metric("Low signal", qc_result.summary["low_signal_count"])
    qc_metrics[4].metric(
        "High-tail candidates",
        qc_result.summary["high_value_candidate_count"],
    )

    st.info(
        "Negative fitted values are flagged, not discarded. They remain part of the "
        "normalization unless you explicitly choose to mask them below."
    )

    st.markdown("#### 2. Explicit reference normalization")

    ref_c1, ref_c2 = st.columns(2)
    with ref_c1:
        norm_family = st.radio(
            "Reference family",
            ["primary", "legacy"],
            horizontal=True,
            key="norm_family",
        )
    norm_scalers = (
        meta["primary_scalers"]
        if norm_family == "primary"
        else meta["legacy_scalers"]
    )
    with ref_c2:
        default_ref = norm_scalers.index("US_IC") if "US_IC" in norm_scalers else 0
        norm_reference = st.selectbox(
            "Reference / scaler",
            norm_scalers,
            index=default_ref,
            key="norm_reference",
        )

    reference_map = load_scaler(path_text, norm_family, norm_reference)

    floor_mode = st.radio(
        "Denominator floor policy",
        ["Absolute value", "Percentile of positive reference pixels"],
        horizontal=True,
        key="floor_mode",
    )
    if floor_mode == "Absolute value":
        denominator_floor = st.number_input(
            "Reference must be greater than",
            value=0.0,
            step=0.001,
            format="%.8g",
            key="absolute_floor",
        )
        floor_description = f"absolute floor = {denominator_floor:.8g}"
    else:
        floor_percentile = st.number_input(
            "Positive-reference percentile",
            min_value=0.0,
            max_value=99.0,
            value=1.0,
            step=0.5,
            key="floor_percentile",
        )
        try:
            denominator_floor = positive_reference_percentile_floor(
                reference_map,
                float(floor_percentile),
            )
            floor_description = (
                f"{floor_percentile:g}th percentile of positive finite "
                f"{norm_reference} = {denominator_floor:.8g}"
            )
        except ValueError as exc:
            st.error(str(exc))
            denominator_floor = 0.0
            floor_description = "could not calculate percentile floor"

    mask_c1, mask_c2, scale_c = st.columns(3)
    mask_negative = mask_c1.checkbox(
        "Mask negative XRF numerator",
        value=False,
        key="norm_mask_negative",
    )
    mask_zero = mask_c2.checkbox(
        "Mask zero XRF numerator",
        value=False,
        key="norm_mask_zero",
    )
    scale_factor = scale_c.number_input(
        "Explicit scale factor",
        value=1.0,
        step=1.0,
        format="%.8g",
        key="norm_scale",
    )

    norm_result = normalize_by_reference(
        qc_raw,
        reference_map,
        numerator_label=qc_channel,
        reference_label=norm_reference,
        denominator_floor=float(denominator_floor),
        scale_factor=float(scale_factor),
        mask_negative_numerator=bool(mask_negative),
        mask_zero_numerator=bool(mask_zero),
        source_identity=summary["source"],
        method=method,
    )

    st.caption(f"Denominator policy: **{floor_description}**")
    st.code(
        f"normalized = ({qc_channel} / {norm_reference}) × {scale_factor:g}",
        language="text",
    )

    norm_display_mode = st.selectbox(
        "Display for raw/normalized comparison",
        ["Raw", "Percentile stretch", "Signed log1p"],
        key="norm_display_mode",
    )
    raw_display = display_transform(qc_raw, norm_display_mode)
    normalized_display = display_transform(norm_result.data, norm_display_mode)

    raw_col, norm_col = st.columns(2)
    with raw_col:
        st.plotly_chart(
            heatmap_figure(
                raw_display.data,
                x_axis,
                y_axis,
                title=f"{qc_channel} · raw source · {raw_display.label}",
                colorscale="Viridis",
                colorbar_title=(
                    channel_lookup[qc_channel]["unit"]
                    if norm_display_mode == "Raw"
                    else "display value"
                ),
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with norm_col:
        st.plotly_chart(
            heatmap_figure(
                normalized_display.data,
                x_axis,
                y_axis,
                title=(
                    f"{qc_channel}/{norm_reference} · "
                    f"{normalized_display.label}"
                ),
                colorscale="Viridis",
                colorbar_title=(
                    "normalized value"
                    if norm_display_mode == "Raw"
                    else "display value"
                ),
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    valid_fraction = norm_result.summary["valid_fraction"]
    norm_metrics = st.columns(4)
    norm_metrics[0].metric("Valid pixels", f"{100.0 * valid_fraction:.2f}%")
    norm_metrics[1].metric(
        "Low/invalid reference",
        norm_result.summary["reference_at_or_below_floor_count"],
    )
    norm_metrics[2].metric(
        "Negative numerator masked",
        norm_result.summary["negative_numerator_masked_count"],
    )
    norm_metrics[3].metric(
        "Zero numerator masked",
        norm_result.summary["zero_numerator_masked_count"],
    )

    with st.expander("Normalization mask + provenance", expanded=False):
        invalid_counts = invalid_reason_counts(norm_result.invalid_reason)
        st.dataframe(
            pd.DataFrame(
                [
                    {"reason": key, "pixel_count": value}
                    for key, value in invalid_counts.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.json(norm_result.provenance)

    st.markdown("#### 3. Detector / acquisition diagnostics")
    primary_scalers = set(meta["primary_scalers"])

    diag_cols = st.columns(2)
    with diag_cols[0]:
        if {"ICR", "OCR"}.issubset(primary_scalers):
            icr = load_scaler(path_text, "primary", "ICR")
            ocr = load_scaler(path_text, "primary", "OCR")
            rate_diag = count_rate_diagnostics(icr, ocr)
            dead_display = display_transform(
                rate_diag["implied_dead_time_fraction"],
                "Percentile stretch",
            )
            st.plotly_chart(
                heatmap_figure(
                    dead_display.data,
                    x_axis,
                    y_axis,
                    title="Implied dead-time fraction = 1 - OCR/ICR",
                    colorscale="Magma",
                    colorbar_title="display value",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Diagnostic formula only. Values are intentionally not clipped to [0, 1]."
            )
        else:
            st.info("ICR/OCR are not both available in the primary scaler family.")

    with diag_cols[1]:
        if {"ELT", "ERT"}.issubset(primary_scalers):
            elt = load_scaler(path_text, "primary", "ELT")
            ert = load_scaler(path_text, "primary", "ERT")
            live_diag = live_time_diagnostics(elt, ert)
            live_display = display_transform(
                live_diag["live_time_fraction"],
                "Percentile stretch",
            )
            st.plotly_chart(
                heatmap_figure(
                    live_display.data,
                    x_axis,
                    y_axis,
                    title="Live-time diagnostic = ELT/ERT",
                    colorscale="Cividis",
                    colorbar_title="display value",
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Shown as an acquisition diagnostic. It is not automatically applied "
                "as a correction."
            )
        else:
            st.info("ELT/ERT are not both available in the primary scaler family.")

    st.warning(
        "US_IC, DS_IC, ELT, ERT, ICR, OCR, and Dead_Time are exposed as candidate "
        "QC/normalization information. Change 05 does not declare any one of them the "
        "scientifically preferred correction for these experiments."
    )



with tabs[6]:
    st.subheader("Intensity + contrast feature explorer")
    st.caption(
        "Display-only transforms change rendering. Derived scientific features create "
        "new numerical maps with explicit formulas, units, and parameters."
    )

    ic_channel = st.selectbox(
        "Source channel",
        channel_names,
        index=channel_names.index("Fe") if "Fe" in channel_names else 0,
        format_func=channel_label,
        key="intensity_channel",
    )
    ic_info = channel_lookup[ic_channel]
    source_unit = ic_info["unit"] or "unknown"
    ic_raw = load_map(path_text, method, ic_channel)

    st.markdown("#### Display-only")
    dc1, dc2 = st.columns([2, 1])
    with dc1:
        display_mode = st.selectbox(
            "Display transform",
            ["Raw", "Percentile stretch", "Signed log1p", "Gamma", "CLAHE"],
            key="intensity_display_mode",
        )
    with dc2:
        cmap = st.selectbox(
            "Colormap",
            ["Viridis", "Cividis", "Inferno", "Magma", "Plasma", "Turbo"],
            key="intensity_cmap",
        )

    low, high, gamma = 1.0, 99.0, 0.7
    if display_mode in {"Percentile stretch", "Gamma", "CLAHE"}:
        p1, p2 = st.columns(2)
        low = p1.number_input("Low percentile", 0.0, 99.0, 1.0, 0.5, key="ic_low")
        high = p2.number_input("High percentile", 1.0, 100.0, 99.0, 0.5, key="ic_high")
        if high <= low:
            high = min(100.0, low + 1.0)
    if display_mode == "Gamma":
        gamma = st.slider("Gamma", 0.1, 3.0, 0.7, 0.05, key="ic_gamma")

    shown = display_transform(
        ic_raw, display_mode, low=float(low), high=float(high), gamma=float(gamma)
    )
    st.plotly_chart(
        heatmap_figure(
            shown.data,
            x_axis,
            y_axis,
            title=f"{ic_channel} · Display-only · {shown.label}",
            colorscale=cmap,
            colorbar_title=source_unit if display_mode == "Raw" else "display value",
        ),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.info(
        "Display-only transforms are visualization products. They are not treated as "
        "quantitative derived feature maps."
    )

    st.markdown("#### Derived scientific feature")
    feature_name = st.selectbox(
        "Feature",
        [
            "Local mean",
            "Local std",
            "Local CV",
            "Local median",
            "Local MAD",
            "Local IQR",
            "Local contrast z",
            "Background estimate",
            "Background-subtracted",
            "Asinh intensity",
            "Robust z-score",
        ],
        key="derived_intensity_feature",
    )

    window_features = {
        "Local mean", "Local std", "Local CV", "Local median",
        "Local MAD", "Local IQR", "Local contrast z",
    }
    window, sigma, mean_floor, std_floor, asinh_scale = 7, 3.0, 0.0, 0.0, None

    if feature_name in window_features:
        window = st.slider(
            "Neighborhood window (odd pixels)", 3, 31, 7, 2, key="ic_window"
        )
        if feature_name == "Local CV":
            mean_floor = st.number_input(
                "Absolute local-mean floor", value=0.0, step=0.001,
                format="%.8g", key="ic_mean_floor"
            )
        if feature_name == "Local contrast z":
            std_floor = st.number_input(
                "Local-std floor", value=0.0, step=0.001,
                format="%.8g", key="ic_std_floor"
            )
    elif feature_name in {"Background estimate", "Background-subtracted"}:
        sigma = st.slider(
            "Gaussian background sigma (pixels)", 0.5, 20.0, 3.0, 0.5, key="ic_sigma"
        )
    elif feature_name == "Asinh intensity":
        if not st.checkbox("Use automatic robust asinh scale", True, key="ic_asinh_auto"):
            asinh_scale = st.number_input(
                "Asinh scale", min_value=1e-12, value=1.0,
                format="%.8g", key="ic_asinh_scale"
            )

    kwargs = {"input_label": ic_channel, "input_unit": source_unit}
    if feature_name == "Local mean":
        feature = local_mean(ic_raw, window=window, **kwargs)
    elif feature_name == "Local std":
        feature = local_std(ic_raw, window=window, **kwargs)
    elif feature_name == "Local CV":
        feature = local_cv(ic_raw, window=window, mean_floor=float(mean_floor), **kwargs)
    elif feature_name == "Local median":
        feature = local_median(ic_raw, window=window, **kwargs)
    elif feature_name == "Local MAD":
        feature = local_mad(ic_raw, window=window, **kwargs)
    elif feature_name == "Local IQR":
        feature = local_iqr(ic_raw, window=window, **kwargs)
    elif feature_name == "Local contrast z":
        feature = local_contrast_z(
            ic_raw, window=window, std_floor=float(std_floor), **kwargs
        )
    elif feature_name == "Background estimate":
        feature = gaussian_background(ic_raw, sigma_pixels=float(sigma), **kwargs)
    elif feature_name == "Background-subtracted":
        feature = background_subtracted(ic_raw, sigma_pixels=float(sigma), **kwargs)
    elif feature_name == "Asinh intensity":
        feature = asinh_feature(ic_raw, scale=asinh_scale, **kwargs)
    else:
        feature = robust_zscore_feature(ic_raw, **kwargs)

    derived_mode = st.selectbox(
        "Derived-map display",
        ["Raw", "Percentile stretch", "Signed log1p"],
        key="derived_intensity_display",
    )
    derived = display_transform(feature.data, derived_mode)
    reference = display_transform(ic_raw, "Percentile stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            heatmap_figure(
                reference.data,
                x_axis,
                y_axis,
                title=f"{ic_channel} · source reference",
                colorscale=cmap,
                colorbar_title="display value",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with right:
        st.plotly_chart(
            heatmap_figure(
                derived.data,
                x_axis,
                y_axis,
                title=f"{feature_name} · {derived.label}",
                colorscale=cmap,
                colorbar_title=(
                    feature.metadata.output_unit if derived_mode == "Raw"
                    else "display value"
                ),
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    mleft, mright = st.columns(2)
    with mleft:
        st.markdown("##### Feature metadata")
        st.json(
            {
                "name": feature.metadata.name,
                "kind": feature.metadata.kind,
                "description": feature.metadata.description,
                "input_label": feature.metadata.input_label,
                "output_unit": feature.metadata.output_unit,
                "parameters": feature.metadata.parameters,
                "provenance": feature.metadata.provenance,
            }
        )
    with mright:
        st.markdown("##### Derived-map statistics")
        stats = map_statistics(feature.data)
        st.dataframe(
            pd.DataFrame([{"metric": k, "value": v} for k, v in stats.items()]),
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Neighborhood windows and Gaussian sigma are currently specified in pixels. "
        "The coordinate unit is still unverified, so no physical distance unit is invented."
    )



with tabs[7]:
    st.subheader("Gradients + edge structure")
    st.caption(
        "Coordinate-aware gradients use the exact stored X/Y coordinate vectors. "
        "Sobel, Scharr, and Prewitt remain pixel-grid edge operators."
    )

    gchannel = st.selectbox(
        "Source channel", channel_names,
        index=channel_names.index("Fe") if "Fe" in channel_names else 0,
        format_func=channel_label, key="g_channel"
    )
    ginfo = channel_lookup[gchannel]
    gunit = ginfo["unit"] or "unknown"
    raw = load_map(path_text, method, gchannel)

    family = st.radio(
        "Derivative family",
        ["Coordinate-aware gradient", "Pixel-grid edge operator"],
        horizontal=True, key="g_family"
    )

    if family == "Coordinate-aware gradient":
        sigma = st.slider(
            "Gaussian pre-smoothing sigma (pixels)", 0.0, 6.0, 1.0, 0.25,
            key="g_sigma"
        )
        field = coordinate_gradient(
            raw, x_axis, y_axis,
            input_label=gchannel, input_unit=gunit, sigma_pixels=float(sigma)
        )
        quantity = st.selectbox(
            "Gradient quantity",
            ["Ix","Iy","Gradient magnitude","Gradient orientation",
             "Directional derivative","Edge mask"],
            key="g_quantity"
        )
        if quantity == "Ix":
            data, meta = field.ix, field.metadata["ix"]
        elif quantity == "Iy":
            data, meta = field.iy, field.metadata["iy"]
        elif quantity == "Gradient magnitude":
            data, meta = field.magnitude, field.metadata["magnitude"]
        elif quantity == "Gradient orientation":
            data, meta = field.orientation_deg, field.metadata["orientation_deg"]
        elif quantity == "Directional derivative":
            angle = st.slider(
                "Direction angle (0=+X, 90=+Y)", -180.0, 180.0, 45.0, 5.0,
                key="g_angle"
            )
            result = directional_derivative(
                field, angle_deg=float(angle), input_label=gchannel
            )
            data, meta = result.data, result.metadata
        else:
            perc = st.slider(
                "Edge-strength percentile", 50.0, 99.9, 90.0, 0.5,
                key="g_edge_p"
            )
            result = edge_mask(
                field.magnitude, percentile=float(perc),
                input_label=f"{gchannel} gradient magnitude"
            )
            data, meta = result.data.astype(float), result.metadata

        st.info(
            "Coordinate derivatives are reported per stored-coordinate-unit because "
            "the numeric X/Y values are known but their physical unit is not yet verified."
        )
    else:
        oplabel = st.selectbox(
            "Pixel-grid operator", ["Sobel","Scharr","Prewitt"], key="g_operator"
        )
        field = pixel_operator_gradient(
            raw, operator=oplabel.lower(),
            input_label=gchannel, input_unit=gunit
        )
        quantity = st.selectbox(
            "Edge quantity",
            ["Ix","Iy","Gradient magnitude","Gradient orientation","Edge mask"],
            key="g_pixel_quantity"
        )
        if quantity == "Ix":
            data, meta = field.ix, field.metadata["ix"]
        elif quantity == "Iy":
            data, meta = field.iy, field.metadata["iy"]
        elif quantity == "Gradient magnitude":
            data, meta = field.magnitude, field.metadata["magnitude"]
        elif quantity == "Gradient orientation":
            data, meta = field.orientation_deg, field.metadata["orientation_deg"]
        else:
            perc = st.slider(
                "Edge-strength percentile", 50.0, 99.9, 90.0, 0.5,
                key="g_pixel_edge_p"
            )
            result = edge_mask(
                field.magnitude, percentile=float(perc),
                input_label=f"{gchannel} {oplabel} magnitude"
            )
            data, meta = result.data.astype(float), result.metadata

        st.info(
            "Pixel-grid edge operators are reported per pixel, not per stored-coordinate-unit."
        )

    mode = st.selectbox(
        "Feature display", ["Raw","Percentile stretch","Signed log1p"],
        key="g_display"
    )
    if quantity == "Gradient orientation":
        mode = "Raw"

    source = display_transform(raw, "Percentile stretch")
    shown = display_transform(data, mode)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            heatmap_figure(
                source.data, x_axis, y_axis,
                title=f"{gchannel} · source reference",
                colorscale="Cividis", colorbar_title="display value"
            ),
            use_container_width=True, config={"displaylogo": False}
        )
    with right:
        st.plotly_chart(
            heatmap_figure(
                shown.data, x_axis, y_axis,
                title=f"{quantity} · {family}",
                colorscale="Turbo",
                colorbar_title=meta.output_unit if mode == "Raw" else "display value"
            ),
            use_container_width=True, config={"displaylogo": False}
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Derivative metadata")
        st.json({
            "name": meta.name,
            "kind": meta.kind,
            "description": meta.description,
            "input_label": meta.input_label,
            "output_unit": meta.output_unit,
            "parameters": meta.parameters,
            "provenance": meta.provenance,
        })
    with c2:
        st.markdown("##### Feature statistics")
        stats = map_statistics(data)
        st.dataframe(
            pd.DataFrame([{"metric":k,"value":v} for k,v in stats.items()]),
            use_container_width=True, hide_index=True
        )

    st.warning(
        "Change 07 is first-derivative only. Hessians, Laplacians, curvature, "
        "ridges, valleys, and blobs are reserved for Change 08."
    )



with tabs[8]:
    st.subheader("Hessian + curvature")
    st.caption(
        "Second derivatives use the actual stored X/Y coordinate values. "
        "Ixx and Iyy come from local quadratic coordinate fits; Ixy is symmetrized "
        "from the two mixed-derivative paths."
    )

    h_channel = st.selectbox(
        "Source channel",
        channel_names,
        index=channel_names.index("Fe") if "Fe" in channel_names else 0,
        format_func=channel_label,
        key="h_channel",
    )
    h_info = channel_lookup[h_channel]
    h_unit = h_info["unit"] or "unknown"
    h_raw = load_map(path_text, method, h_channel)

    hc1, hc2 = st.columns(2)
    sigma = hc1.slider(
        "Gaussian pre-smoothing sigma (pixels)",
        0.0, 6.0, 1.0, 0.25,
        key="h_sigma",
    )
    radius = hc2.slider(
        "Local quadratic fit radius (index steps)",
        1, 6, 2, 1,
        key="h_radius",
        help=(
            "The fitter expands the local window automatically when repeated coordinates "
            "leave fewer than three independent coordinate positions."
        ),
    )

    hfield = coordinate_hessian(
        h_raw,
        x_axis,
        y_axis,
        input_label=h_channel,
        input_unit=h_unit,
        sigma_pixels=float(sigma),
        quadratic_radius=int(radius),
    )

    quantity = st.selectbox(
        "Second-order quantity",
        [
            "Ixx",
            "Iyy",
            "Ixy",
            "Mixed derivative disagreement",
            "Laplacian",
            "Hessian trace",
            "Hessian determinant",
            "Lambda min",
            "Lambda max",
            "Principal direction",
            "Curvedness",
            "Shape index",
            "Bright ridge",
            "Dark valley",
            "Bright blob",
            "Dark blob",
        ],
        key="h_quantity",
    )

    mapping = {
        "Ixx": (hfield.ixx, hfield.metadata["ixx"]),
        "Iyy": (hfield.iyy, hfield.metadata["iyy"]),
        "Ixy": (hfield.ixy, hfield.metadata["ixy"]),
        "Mixed derivative disagreement": (
            hfield.mixed_disagreement,
            hfield.metadata["mixed_disagreement"],
        ),
        "Laplacian": (hfield.laplacian, hfield.metadata["laplacian"]),
        "Hessian trace": (hfield.trace, hfield.metadata["trace"]),
        "Hessian determinant": (
            hfield.determinant,
            hfield.metadata["determinant"],
        ),
        "Lambda min": (hfield.lambda_min, hfield.metadata["lambda_min"]),
        "Lambda max": (hfield.lambda_max, hfield.metadata["lambda_max"]),
        "Principal direction": (
            hfield.principal_orientation_deg,
            hfield.metadata["principal_orientation_deg"],
        ),
        "Curvedness": (hfield.curvedness, hfield.metadata["curvedness"]),
        "Shape index": (hfield.shape_index, hfield.metadata["shape_index"]),
        "Bright ridge": (hfield.bright_ridge, hfield.metadata["bright_ridge"]),
        "Dark valley": (hfield.dark_valley, hfield.metadata["dark_valley"]),
        "Bright blob": (hfield.bright_blob, hfield.metadata["bright_blob"]),
        "Dark blob": (hfield.dark_blob, hfield.metadata["dark_blob"]),
    }
    h_data, h_meta = mapping[quantity]

    h_mode = st.selectbox(
        "Feature display",
        ["Raw", "Percentile stretch", "Signed log1p"],
        key="h_display",
    )
    if quantity in {"Principal direction", "Shape index"}:
        h_mode = "Raw"

    source = display_transform(h_raw, "Percentile stretch")
    shown = display_transform(h_data, h_mode)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            heatmap_figure(
                source.data,
                x_axis,
                y_axis,
                title=f"{h_channel} · source reference",
                colorscale="Cividis",
                colorbar_title="display value",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="hessian_source_plot",
        )
    with right:
        st.plotly_chart(
            heatmap_figure(
                shown.data,
                x_axis,
                y_axis,
                title=f"{quantity} · Hessian",
                colorscale="Turbo",
                colorbar_title=(
                    h_meta.output_unit if h_mode == "Raw" else "display value"
                ),
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="hessian_feature_plot",
        )

    st.info(
        "Ixy and Iyx are calculated through independent derivative paths. "
        "Their absolute disagreement is exposed as numerical QC instead of being hidden."
    )

    with st.expander("Point Hessian inspector", expanded=False):
        ny, nx = h_raw.shape
        pc1, pc2 = st.columns(2)
        py = pc1.slider(
            "Y index",
            0, ny - 1, (ny - 1) // 2,
            key="h_point_y",
        )
        px = pc2.slider(
            "X index",
            0, nx - 1, (nx - 1) // 2,
            key="h_point_x",
        )

        matrix = [
            [float(hfield.ixx[py, px]), float(hfield.ixy[py, px])],
            [float(hfield.ixy[py, px]), float(hfield.iyy[py, px])],
        ]
        st.markdown(
            f"**Coordinate:** x = `{x_axis[px]:.8g}`, y = `{y_axis[py]:.8g}`"
        )
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["x", "y"],
                columns=["x", "y"],
            ),
            use_container_width=True,
        )

        pm = st.columns(6)
        pm[0].metric("λ min", f"{hfield.lambda_min[py, px]:.6g}")
        pm[1].metric("λ max", f"{hfield.lambda_max[py, px]:.6g}")
        pm[2].metric("Trace", f"{hfield.trace[py, px]:.6g}")
        pm[3].metric("Det", f"{hfield.determinant[py, px]:.6g}")
        pm[4].metric("Shape index", f"{hfield.shape_index[py, px]:.6g}")
        pm[5].metric(
            "Principal angle",
            f"{hfield.principal_orientation_deg[py, px]:.4g}°",
        )

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("##### Hessian feature metadata")
        st.json(
            {
                "name": h_meta.name,
                "kind": h_meta.kind,
                "description": h_meta.description,
                "input_label": h_meta.input_label,
                "output_unit": h_meta.output_unit,
                "parameters": h_meta.parameters,
                "provenance": h_meta.provenance,
            }
        )
    with dc2:
        st.markdown("##### Feature statistics")
        h_stats = map_statistics(h_data)
        st.dataframe(
            pd.DataFrame(
                [{"metric": k, "value": v} for k, v in h_stats.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "The ridge, valley, and blob maps in Change 08 are transparent single-scale "
        "Hessian curvature responses. They are not yet multiscale Frangi/vesselness "
        "or scale-selected blob detectors. Those belong in the later multiscale engine."
    )



with tabs[9]:
    st.subheader("Interactive 2.5D yeast-cell chemical landscape")
    st.warning(
        "This is a 2.5D chemical landscape derived from a 2D XRF raster. "
        "It is not a reconstructed physical 3D yeast cell: the current dataset has "
        "no validated physical Z axis, serial-section stack, or tomographic angular series."
    )

    default_height = (
        channel_names.index("Total_Fluorescence_Yield")
        if "Total_Fluorescence_Yield" in channel_names
        else (channel_names.index("P") if "P" in channel_names else 0)
    )
    height_channel = st.selectbox(
        "Surface height source",
        channel_names,
        index=default_height,
        format_func=channel_label,
        key="landscape_height_channel",
    )
    height_info = channel_lookup[height_channel]
    height_unit = height_info["unit"] or "unknown"
    height_raw = load_map(path_text, method, height_channel)

    color_mode = st.selectbox(
        "Surface color source",
        [
            "Chemical channel",
            "Gradient magnitude",
            "Gradient orientation",
            "Curvedness",
            "Shape index",
            "Bright ridge",
            "Dark valley",
            "Bright blob",
            "Dark blob",
        ],
        key="landscape_color_mode",
    )

    color_channel = height_channel
    if color_mode == "Chemical channel":
        default_color = (
            channel_names.index("Zn")
            if "Zn" in channel_names
            else channel_names.index(height_channel)
        )
        color_channel = st.selectbox(
            "Chemical color channel",
            channel_names,
            index=default_color,
            format_func=channel_label,
            key="landscape_color_channel",
        )

    with st.expander("Landscape geometry controls", expanded=True):
        gc1, gc2, gc3 = st.columns(3)
        z_low = gc1.number_input(
            "Height low percentile",
            min_value=0.0,
            max_value=99.0,
            value=1.0,
            step=0.5,
            key="landscape_z_low",
        )
        z_high = gc2.number_input(
            "Height high percentile",
            min_value=1.0,
            max_value=100.0,
            value=99.0,
            step=0.5,
            key="landscape_z_high",
        )
        z_exaggeration = gc3.slider(
            "Vertical exaggeration",
            min_value=0.5,
            max_value=15.0,
            value=4.0,
            step=0.5,
            key="landscape_z_exaggeration",
        )
        if z_high <= z_low:
            z_high = min(100.0, z_low + 1.0)

        gc4, gc5 = st.columns(2)
        landscape_sigma = gc4.slider(
            "Gradient/Hessian smoothing sigma (pixels)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.25,
            key="landscape_sigma",
        )
        landscape_radius = gc5.slider(
            "Hessian quadratic radius",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
            key="landscape_hessian_radius",
        )

        mask_background = st.checkbox(
            "Hide low-signal background",
            value=False,
            key="landscape_mask_background",
        )
        background_percentile = None
        if mask_background:
            background_percentile = st.slider(
                "Background visibility percentile",
                min_value=0.0,
                max_value=80.0,
                value=20.0,
                step=1.0,
                key="landscape_background_percentile",
            )
            st.caption(
                "This is a visualization mask, not cell segmentation. "
                "Individual-cell masks will be introduced later."
            )

        curvature_emphasis = st.checkbox(
            "Curvature emphasis",
            value=False,
            key="landscape_curvature_emphasis",
        )
        curvature_weight = 0.0
        if curvature_emphasis:
            curvature_weight = st.slider(
                "Curvature emphasis strength",
                min_value=0.0,
                max_value=3.0,
                value=0.75,
                step=0.05,
                key="landscape_curvature_weight",
                help=(
                    "Display-only geometry modulation using normalized Hessian curvedness. "
                    "It does not alter the XRF data."
                ),
            )

    gradient_field = coordinate_gradient(
        height_raw,
        x_axis,
        y_axis,
        input_label=height_channel,
        input_unit=height_unit,
        sigma_pixels=float(landscape_sigma),
    )
    landscape_hessian = coordinate_hessian(
        height_raw,
        x_axis,
        y_axis,
        input_label=height_channel,
        input_unit=height_unit,
        sigma_pixels=float(landscape_sigma),
        quadratic_radius=int(landscape_radius),
    )

    surface = build_landscape_height(
        height_raw,
        low_percentile=float(z_low),
        high_percentile=float(z_high),
        vertical_exaggeration=float(z_exaggeration),
        background_mask_percentile=background_percentile,
        curvature=landscape_hessian.curvedness if curvature_emphasis else None,
        curvature_weight=float(curvature_weight),
    )

    if color_mode == "Chemical channel":
        color_raw = load_map(path_text, method, color_channel)
        color_display = display_transform(
            color_raw,
            "Percentile stretch",
            low=1.0,
            high=99.0,
        ).data
        color_title = f"{color_channel} display"
    elif color_mode == "Gradient magnitude":
        color_raw = gradient_field.magnitude
        color_display = display_transform(color_raw, "Percentile stretch").data
        color_title = "|∇I| display"
    elif color_mode == "Gradient orientation":
        color_raw = gradient_field.orientation_deg
        color_display = color_raw
        color_title = "gradient orientation (degrees)"
    elif color_mode == "Curvedness":
        color_raw = landscape_hessian.curvedness
        color_display = display_transform(color_raw, "Percentile stretch").data
        color_title = "curvedness display"
    elif color_mode == "Shape index":
        color_raw = landscape_hessian.shape_index
        color_display = color_raw
        color_title = "shape index"
    elif color_mode == "Bright ridge":
        color_raw = landscape_hessian.bright_ridge
        color_display = display_transform(color_raw, "Percentile stretch").data
        color_title = "bright ridge display"
    elif color_mode == "Dark valley":
        color_raw = landscape_hessian.dark_valley
        color_display = display_transform(color_raw, "Percentile stretch").data
        color_title = "dark valley display"
    elif color_mode == "Bright blob":
        color_raw = landscape_hessian.bright_blob
        color_display = display_transform(color_raw, "Percentile stretch").data
        color_title = "bright blob display"
    else:
        color_raw = landscape_hessian.dark_blob
        color_display = display_transform(color_raw, "Percentile stretch").data
        color_title = "dark blob display"

    visual_c1, visual_c2 = st.columns([1, 1])
    with visual_c1:
        landscape_cmap = st.selectbox(
            "3D colorscale",
            ["Viridis", "Cividis", "Inferno", "Magma", "Plasma", "Turbo"],
            key="landscape_cmap",
        )
        surface_opacity = st.slider(
            "Surface opacity",
            min_value=0.25,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="landscape_opacity",
        )
    with visual_c2:
        show_contours = st.checkbox(
            "Project height contours",
            value=False,
            key="landscape_contours",
        )
        ridge_overlay = st.checkbox(
            "Bright-ridge overlay",
            value=False,
            key="landscape_ridge_overlay",
        )
        blob_overlay = st.checkbox(
            "Bright-blob overlay",
            value=False,
            key="landscape_blob_overlay",
        )

    ridge_percentile = 97.5
    blob_percentile = 97.5
    if ridge_overlay:
        ridge_percentile = st.slider(
            "Bright-ridge overlay percentile",
            min_value=90.0,
            max_value=99.9,
            value=97.5,
            step=0.5,
            key="landscape_ridge_percentile",
        )
    if blob_overlay:
        blob_percentile = st.slider(
            "Bright-blob overlay percentile",
            min_value=90.0,
            max_value=99.9,
            value=97.5,
            step=0.5,
            key="landscape_blob_percentile",
        )

    st.plotly_chart(
        landscape_figure(
            x=x_axis,
            y=y_axis,
            surface=surface,
            surface_color=color_display,
            raw_height=height_raw,
            raw_color=color_raw,
            title=(
                f"2.5D chemical landscape · height={height_channel} · "
                f"color={color_mode if color_mode != 'Chemical channel' else color_channel}"
            ),
            colorbar_title=color_title,
            colorscale=landscape_cmap,
            opacity=float(surface_opacity),
            show_z_contours=bool(show_contours),
            ridge_response=(
                landscape_hessian.bright_ridge if ridge_overlay else None
            ),
            ridge_percentile=float(ridge_percentile),
            blob_response=(
                landscape_hessian.bright_blob if blob_overlay else None
            ),
            blob_percentile=float(blob_percentile),
        ),
        use_container_width=True,
        config={"displaylogo": False},
        key="cell_landscape_3d_plot",
    )

    st.caption(
        "The Z axis is Derived/display height. Rotation, perspective, lighting, "
        "curvature emphasis, and masking are visualization tools and do not create "
        "new experimental Z information."
    )

    ref1, ref2 = st.columns(2)
    with ref1:
        height_reference = display_transform(
            height_raw,
            "Percentile stretch",
            low=float(z_low),
            high=float(z_high),
        )
        st.plotly_chart(
            heatmap_figure(
                height_reference.data,
                x_axis,
                y_axis,
                title=f"Height source · {height_channel}",
                colorscale="Cividis",
                colorbar_title="display value",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="landscape_height_reference_plot",
        )
    with ref2:
        st.plotly_chart(
            heatmap_figure(
                color_display,
                x_axis,
                y_axis,
                title=f"Surface color · {color_title}",
                colorscale=landscape_cmap,
                colorbar_title=color_title,
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="landscape_color_reference_plot",
        )

    with st.expander("Point landscape inspector", expanded=False):
        ny, nx = height_raw.shape
        ic1, ic2 = st.columns(2)
        inspect_y = ic1.slider(
            "Y index",
            0,
            ny - 1,
            (ny - 1) // 2,
            key="landscape_inspect_y",
        )
        inspect_x = ic2.slider(
            "X index",
            0,
            nx - 1,
            (nx - 1) // 2,
            key="landscape_inspect_x",
        )

        display_gradient = coordinate_gradient(
            surface.z_unmasked,
            x_axis,
            y_axis,
            input_label="Derived/display height",
            input_unit="display-height",
            sigma_pixels=0.0,
        )
        normal_x, normal_y, normal_z = display_surface_normals(
            display_gradient.ix,
            display_gradient.iy,
        )

        local_class = hessian_point_class(
            landscape_hessian.trace[inspect_y, inspect_x],
            landscape_hessian.determinant[inspect_y, inspect_x],
        )

        st.markdown(
            f"**Stored coordinate:** x=`{x_axis[inspect_x]:.8g}`, "
            f"y=`{y_axis[inspect_y]:.8g}`  \n"
            f"**Local Hessian class:** `{local_class}`"
        )

        metrics1 = st.columns(6)
        metrics1[0].metric(
            f"{height_channel} raw",
            f"{height_raw[inspect_y, inspect_x]:.6g}",
        )
        metrics1[1].metric(
            "Display height",
            f"{surface.z_unmasked[inspect_y, inspect_x]:.6g}",
        )
        metrics1[2].metric(
            "|∇I|",
            f"{gradient_field.magnitude[inspect_y, inspect_x]:.6g}",
        )
        metrics1[3].metric(
            "Gradient angle",
            f"{gradient_field.orientation_deg[inspect_y, inspect_x]:.4g}°",
        )
        metrics1[4].metric(
            "Curvedness",
            f"{landscape_hessian.curvedness[inspect_y, inspect_x]:.6g}",
        )
        metrics1[5].metric(
            "Shape index",
            f"{landscape_hessian.shape_index[inspect_y, inspect_x]:.6g}",
        )

        metrics2 = st.columns(6)
        metrics2[0].metric(
            "Ixx",
            f"{landscape_hessian.ixx[inspect_y, inspect_x]:.6g}",
        )
        metrics2[1].metric(
            "Ixy",
            f"{landscape_hessian.ixy[inspect_y, inspect_x]:.6g}",
        )
        metrics2[2].metric(
            "Iyy",
            f"{landscape_hessian.iyy[inspect_y, inspect_x]:.6g}",
        )
        metrics2[3].metric(
            "λ min",
            f"{landscape_hessian.lambda_min[inspect_y, inspect_x]:.6g}",
        )
        metrics2[4].metric(
            "λ max",
            f"{landscape_hessian.lambda_max[inspect_y, inspect_x]:.6g}",
        )
        metrics2[5].metric(
            "Bright blob",
            f"{landscape_hessian.bright_blob[inspect_y, inspect_x]:.6g}",
        )

        st.markdown("##### Display-surface normal")
        st.code(
            "n = "
            f"({normal_x[inspect_y, inspect_x]:.6g}, "
            f"{normal_y[inspect_y, inspect_x]:.6g}, "
            f"{normal_z[inspect_y, inspect_x]:.6g})",
            language="text",
        )
        st.caption(
            "This normal belongs to the derived/display landscape geometry. "
            "It is not a measured physical surface normal of the yeast cell."
        )

    with st.expander("Landscape provenance", expanded=False):
        st.json(surface.metadata)


with tabs[10]:
    st.subheader("TFY cell analyzer + inferred 3D cell model")
    st.caption(
        "Cell detection uses Total_Fluorescence_Yield. The measured X/Y footprint is "
        "preserved. Depth is inferred from an explicit geometric model, so this is "
        "not an experimental 3D reconstruction."
    )

    seg_method = st.selectbox(
        "TFY analyzed product",
        ["Fitted", "NNLS", "ROI"],
        index=0,
        key="cell_seg_method",
    )
    tfy = load_map(
        path_text,
        seg_method,
        "Total_Fluorescence_Yield",
    )

    with st.expander("Cell-detection controls", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        threshold_label = sc1.selectbox(
            "Threshold method",
            ["Otsu", "Yen", "Li", "Percentile"],
            key="cell_threshold_method",
        )
        smooth_sigma = sc2.slider(
            "TFY smoothing sigma (pixels)",
            0.0, 5.0, 1.0, 0.25,
            key="cell_seg_sigma",
        )
        min_area = sc3.number_input(
            "Minimum cell area (pixels)",
            min_value=1,
            value=25,
            step=5,
            key="cell_min_area",
        )

        percentile_threshold = 75.0
        if threshold_label == "Percentile":
            percentile_threshold = st.slider(
                "TFY threshold percentile",
                1.0, 99.0, 75.0, 1.0,
                key="cell_threshold_percentile",
            )

        sc4, sc5, sc6 = st.columns(3)
        closing_radius = sc4.slider(
            "Morphological closing radius (pixels)",
            0, 5, 1, 1,
            key="cell_closing_radius",
        )
        split_touching = sc5.checkbox(
            "Split touching cells",
            value=True,
            key="cell_split_touching",
        )
        peak_distance = sc6.slider(
            "Watershed minimum peak distance",
            1, 20, 5, 1,
            key="cell_peak_distance",
            disabled=not split_touching,
        )

    cells = segment_cells_tfy(
        tfy,
        x_axis,
        y_axis,
        threshold_method=threshold_label.lower(),
        threshold_percentile=float(percentile_threshold),
        smooth_sigma_pixels=float(smooth_sigma),
        min_area_pixels=int(min_area),
        closing_radius_pixels=int(closing_radius),
        fill_holes=True,
        split_touching=bool(split_touching),
        watershed_min_distance_pixels=int(peak_distance),
    )

    counts = st.columns(4)
    counts[0].metric("Detected cells", cells.total_cells)
    counts[1].metric("Complete cells", cells.complete_cells)
    counts[2].metric("Cropped cells", cells.cropped_cells)
    counts[3].metric(
        "TFY threshold",
        f"{cells.threshold_value:.4g}",
    )

    st.caption(
        "A cell is marked cropped when its segmented component touches any image border. "
        "Cropped cells are reported but can be excluded from quantitative cell analysis."
    )

    exclude_cropped = st.checkbox(
        "Exclude cropped cells",
        value=True,
        key="exclude_cropped_cells",
    )

    records = [record.to_dict() for record in cells.records]
    if records:
        table = pd.DataFrame(records)
        st.dataframe(
            table[
                [
                    "cell_id",
                    "cropped",
                    "area_pixels",
                    "coordinate_area_approx",
                    "equivalent_diameter_coordinate_approx",
                    "major_axis_coordinate_approx",
                    "minor_axis_coordinate_approx",
                    "orientation_deg_coordinate",
                    "eccentricity_coordinate",
                    "circularity_pixel",
                    "solidity",
                    "mean_tfy",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Coordinate sizes are currently reported in stored-coordinate units. "
            "The physical X/Y unit has not yet been verified, so these values are not "
            "labeled as micrometers."
        )
    else:
        st.warning(
            "No cell candidates were detected with the current settings. "
            "Adjust the TFY threshold and morphology controls."
        )

    label_display = cells.labels.astype(float)
    cropped_ids = {record.cell_id for record in cells.records if record.cropped}
    for cell_id in cropped_ids:
        label_display[cells.labels == cell_id] = -float(cell_id)

    cell_ref1, cell_ref2 = st.columns(2)
    with cell_ref1:
        st.plotly_chart(
            heatmap_figure(
                display_transform(tfy, "Percentile stretch").data,
                x_axis,
                y_axis,
                title="Total_Fluorescence_Yield · segmentation source",
                colorscale="Cividis",
                colorbar_title="display value",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="cell_tfy_reference_plot",
        )
    with cell_ref2:
        st.plotly_chart(
            heatmap_figure(
                label_display,
                x_axis,
                y_axis,
                title="Detected cell labels · negative IDs are cropped",
                colorscale="Turbo",
                colorbar_title="cell ID",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="cell_label_reference_plot",
        )

    eligible = [
        record
        for record in cells.records
        if (not exclude_cropped or not record.cropped)
    ]

    if eligible:
        selected_id = st.selectbox(
            "Cell to model",
            [record.cell_id for record in eligible],
            format_func=lambda cid: (
                f"Cell {cid}"
                + (
                    " · CROPPED"
                    if next(r for r in cells.records if r.cell_id == cid).cropped
                    else " · complete"
                )
            ),
            key="selected_cell_model_id",
        )
        selected = next(
            record for record in cells.records if record.cell_id == selected_id
        )
        cell_mask = cells.labels == selected_id

        st.markdown("### Inferred 3D cell model")
        st.warning(
            "The X/Y cell footprint comes from measured TFY. The inferred depth is a "
            "model assumption constrained by the measured footprint and minor-axis size. "
            "It is not an experimental 3D reconstruction."
        )

        mc1, mc2, mc3 = st.columns(3)
        depth_ratio = mc1.slider(
            "Depth / minor-radius ratio",
            0.25, 2.0, 1.0, 0.05,
            key="cell_depth_ratio",
            help=(
                "1.0 assumes the maximum half-depth is approximately the measured "
                "minor-axis radius in stored-coordinate units."
            ),
        )
        dome_power = mc2.slider(
            "Envelope roundness",
            0.25, 2.0, 0.75, 0.05,
            key="cell_dome_power",
        )
        surface_opacity = mc3.slider(
            "Envelope opacity",
            0.10, 1.0, 0.40, 0.05,
            key="cell_model_opacity",
        )

        envelope = build_mask_conforming_envelope(
            cell_mask,
            x_axis,
            y_axis,
            depth_ratio_to_minor_radius=float(depth_ratio),
            minor_axis_coordinate=selected.minor_axis_coordinate_approx,
            dome_power=float(dome_power),
        )

        color_default = (
            channel_names.index("Zn")
            if "Zn" in channel_names
            else 0
        )
        model_color_channel = st.selectbox(
            "XRF channel mapped onto the cell",
            channel_names,
            index=color_default,
            format_func=channel_label,
            key="cell_model_color_channel",
        )
        model_color_raw = load_map(
            path_text,
            method,
            model_color_channel,
        )
        model_color_masked = np.where(cell_mask, model_color_raw, np.nan)
        surface_color = display_transform(
            model_color_masked,
            "Percentile stretch",
        ).data

        st.markdown("#### Projection-conserving interior")
        st.caption(
            "For the internal model, the measured 2D XRF signal is treated as a "
            "projection through the inferred thickness. A uniform-depth density is "
            "computed so that integrating through the modeled cell returns the original "
            "2D measured XRF map."
        )

        density, thickness_floor = projection_conserving_density(
            model_color_masked,
            envelope,
            thickness_floor_fraction=0.10,
        )

        z_fraction = st.slider(
            "Internal Z slice",
            -1.0, 1.0, 0.0, 0.05,
            key="cell_internal_z",
            help="0 is the modeled equatorial plane.",
        )
        slice_data, slice_z, _ = internal_slice(
            density,
            envelope,
            z_fraction=float(z_fraction),
        )

        cut1, cut2, cut3 = st.columns(3)
        cutaway_axis = cut1.selectbox(
            "Cutaway axis",
            ["none", "x", "y"],
            key="cell_cutaway_axis",
        )
        cutaway_fraction = cut2.slider(
            "Cutaway fraction",
            0.0, 1.0, 0.50, 0.05,
            key="cell_cutaway_fraction",
            disabled=cutaway_axis == "none",
        )
        show_bottom = cut3.checkbox(
            "Show lower envelope",
            value=True,
            key="cell_show_bottom",
        )

        st.plotly_chart(
            cell_model_figure(
                x=x_axis,
                y=y_axis,
                envelope=envelope,
                surface_color=surface_color,
                raw_surface_color=model_color_masked,
                slice_data=slice_data,
                slice_z=slice_z,
                title=(
                    f"Cell {selected_id} · inferred 3D envelope · "
                    f"XRF={model_color_channel}"
                ),
                colorscale="Viridis",
                surface_opacity=float(surface_opacity),
                show_bottom=bool(show_bottom),
                cutaway_axis=cutaway_axis,
                cutaway_fraction=float(cutaway_fraction),
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="inferred_cell_model_plot",
        )

        st.caption(
            "You can rotate into the cutaway and inspect the internal model slice. "
            "The surface footprint is measured; the depth and volumetric XRF distribution "
            "are inferred under the displayed assumptions."
        )

        st.markdown("#### Selected-cell geometry")
        geom_metrics = st.columns(6)
        geom_metrics[0].metric("Area (px)", selected.area_pixels)
        geom_metrics[1].metric(
            "Area (coord²)",
            f"{selected.coordinate_area_approx:.6g}",
        )
        geom_metrics[2].metric(
            "Eq. diameter",
            f"{selected.equivalent_diameter_coordinate_approx:.6g}",
        )
        geom_metrics[3].metric(
            "Major axis",
            f"{selected.major_axis_coordinate_approx:.6g}",
        )
        geom_metrics[4].metric(
            "Minor axis",
            f"{selected.minor_axis_coordinate_approx:.6g}",
        )
        geom_metrics[5].metric(
            "Inferred max half-depth",
            f"{envelope.max_half_depth:.6g}",
        )

        st.markdown("#### Per-cell XRF summary")
        chemistry_rows = []
        for chem_name in [
            name for name in ("P", "S", "K", "Ca", "Mn", "Fe", "Cu", "Zn")
            if name in channel_names
        ]:
            chem_map = load_map(path_text, method, chem_name)
            values = chem_map[cell_mask]
            finite_values = values[np.isfinite(values)]
            chemistry_rows.append(
                {
                    "channel": chem_name,
                    "mean": (
                        float(np.mean(finite_values))
                        if finite_values.size else np.nan
                    ),
                    "median": (
                        float(np.median(finite_values))
                        if finite_values.size else np.nan
                    ),
                    "sum": (
                        float(np.sum(finite_values))
                        if finite_values.size else np.nan
                    ),
                    "max": (
                        float(np.max(finite_values))
                        if finite_values.size else np.nan
                    ),
                    "pixel_count": int(finite_values.size),
                }
            )

        st.dataframe(
            pd.DataFrame(chemistry_rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("3D model assumptions and provenance", expanded=False):
            st.json(
                {
                    "segmentation": cells.metadata,
                    "cell": selected.to_dict(),
                    "envelope": envelope.metadata,
                    "interior_model": {
                        "type": "projection-conserving uniform-depth density",
                        "formula": "density = measured_2D_XRF / inferred_thickness",
                        "thickness_floor": thickness_floor,
                        "physical_z_measured": False,
                        "measured_xy_footprint": True,
                    },
                }
            )


with tabs[11]:
    st.subheader("Analyze an entire directory")
    st.caption(
        "Build a reproducible study across every HDF5 sample in a directory. "
        "Each scan gets a TFY sample snapshot, cell census, full-cell crops, "
        "chemistry tables, structural feature summaries, and optional centroid spectra."
    )

    with st.container(border=True):
        st.markdown("#### Batch inputs")
        bc1, bc2 = st.columns([2.2, 1.8])
        with bc1:
            batch_input_dir = st.text_input(
                "Input directory",
                value=str(ROOT / "img.dat"),
                key="batch_input_dir",
            )
        with bc2:
            batch_output_root = st.text_input(
                "Output root",
                value=str(ROOT / "analysis" / "batch"),
                key="batch_output_root",
            )

        bc3, bc4, bc5 = st.columns(3)
        with bc3:
            batch_methods = st.multiselect(
                "Analyzed products",
                ["Fitted", "NNLS", "ROI"],
                default=["Fitted", "NNLS", "ROI"],
                key="batch_methods",
            )
        with bc4:
            batch_tfy_method = st.selectbox(
                "TFY segmentation product",
                ["Fitted", "NNLS", "ROI"],
                index=0,
                key="batch_tfy_method",
            )
        with bc5:
            structural_choices = [
                name for name in ("P", "Fe", "Zn") if name in channel_names
            ]
            batch_structural = st.multiselect(
                "Gradient/Hessian channels",
                structural_choices,
                default=structural_choices,
                key="batch_structural_channels",
            )

    with st.expander("Cell-detection settings", expanded=False):
        bs1, bs2, bs3 = st.columns(3)
        batch_threshold = bs1.selectbox(
            "Threshold",
            ["otsu", "yen", "li", "percentile"],
            index=0,
            key="batch_threshold",
        )
        batch_sigma = bs2.slider(
            "TFY smoothing sigma",
            0.0, 5.0, 1.0, 0.25,
            key="batch_sigma",
        )
        batch_min_area = bs3.number_input(
            "Minimum cell area (pixels)",
            min_value=1,
            value=25,
            step=5,
            key="batch_min_area",
        )

        bs4, bs5, bs6 = st.columns(3)
        batch_closing = bs4.slider(
            "Closing radius",
            0, 5, 1, 1,
            key="batch_closing",
        )
        batch_split = bs5.checkbox(
            "Split touching cells",
            value=True,
            key="batch_split",
        )
        batch_peak_distance = bs6.slider(
            "Watershed peak distance",
            1, 20, 5, 1,
            key="batch_peak_distance",
            disabled=not batch_split,
        )

        batch_percentile = 75.0
        if batch_threshold == "percentile":
            batch_percentile = st.slider(
                "Threshold percentile",
                1.0, 99.0, 75.0, 1.0,
                key="batch_percentile",
            )

    with st.expander("Structural-analysis settings", expanded=False):
        bst1, bst2, bst3 = st.columns(3)
        batch_struct_sigma = bst1.slider(
            "Gradient/Hessian sigma",
            0.0, 6.0, 1.0, 0.25,
            key="batch_struct_sigma",
        )
        batch_hessian_radius = bst2.slider(
            "Hessian quadratic radius",
            1, 6, 2, 1,
            key="batch_hessian_radius",
        )
        batch_spectra = bst3.checkbox(
            "Save centroid spectrum for every full cell",
            value=True,
            key="batch_centroid_spectra",
        )

    st.info(
        "**Counting rule:** Detected cells includes every TFY candidate. "
        "**Cropped cells excluded** are candidates touching any image edge. "
        "**Full cells analyzed** are the remaining complete cells that enter "
        "chemistry and structural-feature analysis."
    )

    if st.button(
        "Run directory analysis",
        type="primary",
        use_container_width=True,
        key="run_batch_analysis",
    ):
        if not batch_methods:
            st.error("Select at least one analyzed product.")
        else:
            with st.spinner(
                "Analyzing directory, creating sample snapshots, and building cell tables..."
            ):
                batch_result = analyze_directory(
                    batch_input_dir,
                    output_root=batch_output_root,
                    methods=tuple(batch_methods),
                    tfy_method=batch_tfy_method,
                    threshold_method=batch_threshold,
                    threshold_percentile=float(batch_percentile),
                    smooth_sigma_pixels=float(batch_sigma),
                    min_area_pixels=int(batch_min_area),
                    closing_radius_pixels=int(batch_closing),
                    split_touching=bool(batch_split),
                    watershed_min_distance_pixels=int(batch_peak_distance),
                    structural_channels=tuple(batch_structural),
                    structural_sigma_pixels=float(batch_struct_sigma),
                    hessian_radius=int(batch_hessian_radius),
                    centroid_spectra=bool(batch_spectra),
                )

            summary = batch_result["summary"]
            st.success("Directory analysis complete.")

            bm = st.columns(5)
            bm[0].metric("Scans analyzed", summary["scans_analyzed"])
            bm[1].metric("Detected cells", summary["detected_cells"])
            bm[2].metric("Full cells analyzed", summary["full_cells_analyzed"])
            bm[3].metric(
                "Cropped cells excluded",
                summary["cropped_cells_excluded"],
            )
            bm[4].metric("Scan failures", summary["scan_failures"])

            st.markdown("#### Sample census")
            st.dataframe(
                batch_result["scans"],
                use_container_width=True,
                hide_index=True,
            )

            if not batch_result["failures"].empty:
                st.markdown("#### Failures")
                st.dataframe(
                    batch_result["failures"],
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("#### Study output")
            st.code(str(batch_result["run_dir"]), language="text")
            st.caption(
                "Open index.html in this run directory for a visual sample-by-sample "
                "report. Every sample snapshot shows TFY, detected IDs, and full versus "
                "cropped cells."
            )


from yeast_xrf.analysis.maps_concentration import maps_concentration
from yeast_xrf.io.maps_concentration import quantifiable_channels

with tabs[12]:
    st.markdown(
        """
        <style>
        .sd-hero-card{
            border:1px solid rgba(49, 51, 63, 0.14);
            border-radius:18px;
            padding:1.1rem 1.2rem 1rem 1.2rem;
            background:linear-gradient(180deg, rgba(250,250,248,1) 0%, rgba(246,245,242,1) 100%);
            margin-bottom:0.9rem;
        }
        .sd-kicker{
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:0.08em;
            text-transform:uppercase;
            color:#7a6d5b;
            margin-bottom:0.25rem;
        }
        .sd-title{
            font-size:1.6rem;
            font-weight:700;
            line-height:1.2;
            color:#222222;
            margin-bottom:0.2rem;
        }
        .sd-subtitle{
            font-size:0.96rem;
            color:#5f5a52;
            margin-bottom:0.7rem;
        }
        .sd-pill-row{
            display:flex;
            flex-wrap:wrap;
            gap:0.45rem;
            margin-top:0.45rem;
        }
        .sd-pill{
            border:1px solid rgba(49, 51, 63, 0.14);
            border-radius:999px;
            padding:0.28rem 0.68rem;
            font-size:0.82rem;
            font-weight:600;
            background:#ffffff;
            color:#2e2e2e;
        }
        .sd-card{
            border:1px solid rgba(49, 51, 63, 0.12);
            border-radius:16px;
            padding:0.95rem 1rem;
            background:#faf9f7;
            margin-bottom:0.8rem;
        }
        .sd-section{
            margin-top:0.55rem;
            margin-bottom:0.4rem;
        }
        .sd-section-label{
            font-size:0.76rem;
            font-weight:700;
            letter-spacing:0.08em;
            text-transform:uppercase;
            color:#8b755d;
            margin-bottom:0.18rem;
        }
        .sd-section-title{
            font-size:1.18rem;
            font-weight:700;
            color:#252525;
            margin-bottom:0.15rem;
        }
        .sd-section-copy{
            font-size:0.92rem;
            color:#61594f;
            margin-bottom:0.25rem;
        }
        .sd-mini-grid{
            display:grid;
            grid-template-columns:repeat(3, minmax(0,1fr));
            gap:0.55rem;
            margin-top:0.35rem;
        }
        .sd-mini-item{
            border:1px solid rgba(49, 51, 63, 0.10);
            border-radius:14px;
            background:#ffffff;
            padding:0.6rem 0.7rem;
        }
        .sd-mini-label{
            font-size:0.72rem;
            color:#7b7267;
            text-transform:uppercase;
            letter-spacing:0.06em;
            margin-bottom:0.18rem;
        }
        .sd-mini-value{
            font-size:1rem;
            font-weight:700;
            color:#262626;
            overflow-wrap:anywhere;
        }
        .sd-plot-label{
            font-size:0.78rem;
            font-weight:700;
            letter-spacing:0.06em;
            text-transform:uppercase;
            color:#7c6f5e;
            margin-bottom:0.25rem;
            margin-top:0.15rem;
        }
        .sd-success{
            border:1px solid rgba(52, 105, 72, 0.22);
            border-radius:16px;
            padding:0.95rem 1rem;
            background:rgba(239, 247, 241, 0.8);
            color:#304737;
        }
        .sd-muted{
            font-size:0.86rem;
            color:#6f665a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    quant_us_channels = quantifiable_channels(
        path_text,
        method=method,
        reference="US_IC",
    )
    quant_ds_channels = quantifiable_channels(
        path_text,
        method=method,
        reference="DS_IC",
    )
    quantifiable = [
        name
        for name in channel_names
        if name in quant_us_channels and name in quant_ds_channels
    ]

    if not quantifiable:
        st.error(
            "No channels passed the validated MAPS calibration gate for "
            f"{method} in this scan."
        )
        st.stop()

    default_quant_channel = (
        "Zn"
        if "Zn" in quantifiable
        else quantifiable[0]
    )
    if st.session_state.get("u11_quant_channel") not in quantifiable:
        st.session_state["u11_quant_channel"] = default_quant_channel

    quant_channel = st.session_state.get(
        "u11_quant_channel",
        default_quant_channel,
    )
    quant_floor_percentile = float(
        st.session_state.get("u11_quant_floor_percentile", 1.0)
    )
    concentration_reference = st.session_state.get(
        "change10_concentration_reference",
        "US_IC",
    )
    if concentration_reference not in ("US_IC", "DS_IC"):
        concentration_reference = "US_IC"

    st.markdown(
        f"""
        <div class="sd-hero-card">
          <div class="sd-kicker">SOURDOUGH · Beam & Quantification</div>
          <div class="sd-title">Beam & Quantification</div>
          <div class="sd-subtitle">
            {selected_file.name} · {method} · {quant_channel}
            <br>
            <span class="sd-muted">
              Counts/s → beam normalization → validated MAPS concentration
            </span>
          </div>
          <div class="sd-pill-row">
            <span class="sd-pill">● Normalization ready</span>
            <span class="sd-pill">● MAPS calibration verified</span>
            <span class="sd-pill">● MAPS concentration ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_cols = st.columns([1.0, 2.0], gap="large")

    with top_cols[0]:
        st.markdown(
            """
            <div class="sd-card">
              <div class="sd-section-label">Controls</div>
              <div class="sd-section-title">Quantitative view</div>
              <div class="sd-section-copy">
                Only channels that pass the MAPS calibration consistency
                check are offered for automatic concentration.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        quant_channel = st.selectbox(
            "Element",
            quantifiable,
            index=quantifiable.index(quant_channel),
            format_func=channel_label,
            key="u11_quant_channel",
        )

        concentration_reference = st.radio(
            "Concentration reference",
            ["US_IC", "DS_IC"],
            index=0 if concentration_reference == "US_IC" else 1,
            horizontal=True,
            key="change10_concentration_reference",
            help=(
                "The selected reference determines which validated MAPS "
                "calibration factor is used."
            ),
        )

        quant_floor_percentile = st.slider(
            "Low-scaler mask (%)",
            0.0,
            10.0,
            float(quant_floor_percentile),
            0.25,
            key="u11_quant_floor_percentile",
        )

        display_mode = st.selectbox(
            "Signal display",
            ["Percentile stretch", "Raw"],
            index=0,
            key="u11_quant_display_mode",
            help=(
                "Display-only transform for the signal comparison panels. "
                "The concentration calculation always uses the original "
                "analyzed counts/s values."
            ),
        )

        st.caption(
            "Automatic concentration ignores display transforms and uses "
            "the stored analyzed counts/s, scaler, and verified MAPS factor."
        )

    quant_raw = load_map(path_text, method, quant_channel)
    us_ic = load_scaler(
        path_text,
        family="primary",
        name="US_IC",
    )
    ds_ic = load_scaler(
        path_text,
        family="primary",
        name="DS_IC",
    )

    source_identity = {
        "file": selected_file.name,
        "path": path_text,
    }

    us_product = beam_normalize(
        quant_raw,
        us_ic,
        numerator_label=quant_channel,
        reference_label="US_IC",
        floor_percentile=float(quant_floor_percentile),
        scale_factor=1.0,
        method=method,
        source_identity=source_identity,
    )
    ds_product = beam_normalize(
        quant_raw,
        ds_ic,
        numerator_label=quant_channel,
        reference_label="DS_IC",
        floor_percentile=float(quant_floor_percentile),
        scale_factor=1.0,
        method=method,
        source_identity=source_identity,
    )
    beam_ratio = transmission_diagnostic(
        us_ic,
        ds_ic,
        floor_percentile=float(quant_floor_percentile),
    )
    comparison = normalization_comparison(
        us_product.data,
        ds_product.data,
    )

    active_scaler = (
        us_ic
        if concentration_reference == "US_IC"
        else ds_ic
    )
    auto_product = maps_concentration(
        path_text,
        quant_raw,
        active_scaler,
        method=method,
        reference=concentration_reference,
        channel=quant_channel,
        floor_percentile=float(quant_floor_percentile),
    )

    finite_concentration = auto_product.data[
        np.isfinite(auto_product.data)
    ]

    with top_cols[1]:
        median_conc = (
            float(np.median(finite_concentration))
            if finite_concentration.size
            else float("nan")
        )
        valid_fraction = float(np.mean(auto_product.valid_mask))

        st.markdown(
            f"""
            <div class="sd-card">
              <div class="sd-section-label">Current quantitative product</div>
              <div class="sd-section-title">{quant_channel} · {concentration_reference}</div>
              <div class="sd-mini-grid">
                <div class="sd-mini-item">
                  <div class="sd-mini-label">Method</div>
                  <div class="sd-mini-value">{method}</div>
                </div>
                <div class="sd-mini-item">
                  <div class="sd-mini-label">Valid pixels</div>
                  <div class="sd-mini-value">{valid_fraction:.2%}</div>
                </div>
                <div class="sd-mini-item">
                  <div class="sd-mini-label">Median</div>
                  <div class="sd-mini-value">
                    {median_conc:.4g} µg/cm²
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="sd-section">
          <div class="sd-section-label">1 · Signal & beam normalization</div>
          <div class="sd-section-title">Compare raw signal and normalization references</div>
          <div class="sd-section-copy">
            These views are diagnostics. The quantitative product below is
            calculated from the original analyzed counts/s values.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    plot_cols_top = st.columns(2, gap="large")
    with plot_cols_top[0]:
        st.markdown(
            '<div class="sd-plot-label">Raw signal</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            heatmap_figure(
                display_transform(
                    quant_raw,
                    display_mode,
                ).data,
                x_axis,
                y_axis,
                title=f"Raw {quant_channel}",
                colorscale="Cividis",
                colorbar_title=(
                    channel_lookup[quant_channel]["unit"] or "raw"
                ),
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="u1_quant_raw_plot",
        )

    with plot_cols_top[1]:
        st.markdown(
            '<div class="sd-plot-label">US_IC-normalized</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            heatmap_figure(
                display_transform(
                    us_product.data,
                    display_mode,
                ).data,
                x_axis,
                y_axis,
                title=f"US_IC-normalized {quant_channel}",
                colorscale="Cividis",
                colorbar_title="counts/s ÷ US_IC",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="u1_quant_us_plot",
        )

    plot_cols_bottom = st.columns(2, gap="large")
    with plot_cols_bottom[0]:
        st.markdown(
            '<div class="sd-plot-label">DS_IC-normalized</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            heatmap_figure(
                display_transform(
                    ds_product.data,
                    display_mode,
                ).data,
                x_axis,
                y_axis,
                title=f"DS_IC-normalized {quant_channel}",
                colorscale="Cividis",
                colorbar_title="counts/s ÷ DS_IC",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="u1_quant_ds_plot",
        )

    with plot_cols_bottom[1]:
        st.markdown(
            '<div class="sd-plot-label">Beam ratio diagnostic</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            heatmap_figure(
                beam_ratio["data"],
                x_axis,
                y_axis,
                title="Beam ratio diagnostic · DS_IC / US_IC",
                colorscale="Viridis",
                colorbar_title="DS_IC / US_IC",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="u1_quant_beam_ratio_plot",
        )

    diag_cols = st.columns(4)
    diag_cols[0].metric(
        "US valid",
        f"{np.mean(us_product.valid_mask):.2%}",
    )
    diag_cols[1].metric(
        "DS valid",
        f"{np.mean(ds_product.valid_mask):.2%}",
    )
    diag_cols[2].metric(
        "Scaler floor",
        f"{auto_product.denominator_floor:.6g}",
    )
    diag_cols[3].metric(
        "US↔DS Pearson r",
        (
            "n/a"
            if not np.isfinite(comparison["pearson_r"])
            else f"{comparison['pearson_r']:.4f}"
        ),
    )

    with st.expander("Advanced signal diagnostics", expanded=False):
        st.json(
            {
                "US_IC normalization": us_product.provenance,
                "DS_IC normalization": ds_product.provenance,
                "US vs DS comparison": comparison,
                "beam ratio formula": beam_ratio["formula"],
            }
        )

    st.markdown(
        """
        <div class="sd-section">
          <div class="sd-section-label">2 · Calibration</div>
          <div class="sd-section-title">Check calibration readiness</div>
          <div class="sd-section-copy">
            The active MAPS factor must match the exact stored calibration
            curve entry for this element, method, and scaler reference.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    factor_relerr = auto_product.provenance[
        "factor_curve_relative_error"
    ]
    factor_row = auto_product.provenance["factor_row"]
    curve_position = auto_product.provenance[
        "calibration_curve_position"
    ]

    st.markdown(
        f"""
        <div class="sd-success">
          <strong>MAPS calibration verified.</strong><br>
          Factor: <strong>{auto_product.calibration_factor:.8g}</strong><br>
          Reference: <strong>{concentration_reference}</strong> ·
          legacy quant row <strong>{factor_row}</strong><br>
          Calibration-curve position:
          <strong>{curve_position}</strong><br>
          Factor ↔ curve relative error:
          <strong>{factor_relerr:.3g}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Calibration provenance", expanded=False):
        st.json(auto_product.provenance)

    st.markdown(
        """
        <div class="sd-section">
          <div class="sd-section-label">3 · Concentration</div>
          <div class="sd-section-title">Automatic MAPS concentration</div>
          <div class="sd-section-copy">
            Quantitative elemental areal density from the validated MAPS
            calibration relationship.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    conc_cols = st.columns([2.2, 1.0], gap="large")

    with conc_cols[0]:
        st.plotly_chart(
            heatmap_figure(
                auto_product.data,
                x_axis,
                y_axis,
                title=(
                    f"{quant_channel} concentration · "
                    f"{concentration_reference}"
                ),
                colorscale="Viridis",
                colorbar_title="µg/cm²",
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="u1_quant_areal_density_plot",
        )

    with conc_cols[1]:
        if finite_concentration.size:
            p05 = float(
                np.percentile(finite_concentration, 5)
            )
            p95 = float(
                np.percentile(finite_concentration, 95)
            )
            cmin = float(np.min(finite_concentration))
            cmax = float(np.max(finite_concentration))
            negatives = int(
                np.count_nonzero(finite_concentration < 0)
            )
        else:
            p05 = p95 = cmin = cmax = float("nan")
            negatives = 0

        st.markdown(
            f"""
            <div class="sd-card">
              <div class="sd-section-label">Quantitative summary</div>
              <div class="sd-mini-item">
                <div class="sd-mini-label">Median</div>
                <div class="sd-mini-value">
                  {median_conc:.5g} µg/cm²
                </div>
              </div>
              <div style="height:0.4rem"></div>
              <div class="sd-mini-item">
                <div class="sd-mini-label">5–95% range</div>
                <div class="sd-mini-value">
                  {p05:.4g} – {p95:.4g}
                </div>
              </div>
              <div style="height:0.4rem"></div>
              <div class="sd-mini-item">
                <div class="sd-mini-label">Negative pixels preserved</div>
                <div class="sd-mini-value">{negatives}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Negative analyzed values are retained rather than silently "
            "clipped. Low/invalid scaler pixels are masked."
        )

    with st.expander("Manual concentration factor", expanded=False):
        st.caption(
            "Advanced fallback only. Automatic MAPS concentration above "
            "uses the validated embedded calibration and is the default."
        )

        explicit_reference = st.selectbox(
            "Manual normalized source",
            ["US_IC", "DS_IC"],
            key="u11_quant_explicit_reference",
        )
        explicit_slope = st.number_input(
            "Manual slope",
            value=1.0,
            format="%.12g",
            key="u11_quant_explicit_slope",
        )
        explicit_intercept = st.number_input(
            "Manual intercept",
            value=0.0,
            format="%.12g",
            key="u11_quant_explicit_intercept",
        )
        explicit_unit = st.text_input(
            "Manual output unit",
            value="µg/cm²",
            key="u11_quant_explicit_unit",
        )
        explicit_source = st.text_input(
            "Manual calibration source",
            value="",
            placeholder="Document the trusted source",
            key="u11_quant_explicit_source",
        )

        if explicit_source.strip():
            manual_source = (
                us_product
                if explicit_reference == "US_IC"
                else ds_product
            )
            manual_product = apply_explicit_areal_density_conversion(
                manual_source.data,
                slope=float(explicit_slope),
                intercept=float(explicit_intercept),
                output_unit=explicit_unit,
                calibration_source=explicit_source,
                reference_label=explicit_reference,
                channel_label=quant_channel,
                method=method,
            )
            st.plotly_chart(
                heatmap_figure(
                    manual_product.data,
                    x_axis,
                    y_axis,
                    title=f"{quant_channel} · manual comparison",
                    colorscale="Viridis",
                    colorbar_title=manual_product.unit,
                ),
                use_container_width=True,
                config={"displaylogo": False},
                key="change10_manual_concentration_plot",
            )

    st.markdown(
        """
        <div class="sd-section">
          <div class="sd-section-label">4 · Cell results</div>
          <div class="sd-section-title">Full-cell quantitative summary</div>
          <div class="sd-section-copy">
            Automatic MAPS concentration summarized over complete detected
            cells. Border-touching cells remain excluded by default.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tfy_for_quant = load_map(
        path_text,
        method,
        "Total_Fluorescence_Yield",
    )
    quant_cells = segment_cells_tfy(
        tfy_for_quant,
        x_axis,
        y_axis,
        threshold_method="otsu",
        smooth_sigma_pixels=1.0,
        min_area_pixels=25,
        closing_radius_pixels=1,
        fill_holes=True,
        split_touching=True,
        watershed_min_distance_pixels=5,
    )

    cell_rows = summarize_full_cells(
        auto_product.data,
        quant_cells.labels,
        quant_cells.records,
    )

    if cell_rows:
        cell_df = pd.DataFrame(cell_rows)
        cell_df["unit"] = auto_product.unit
        cell_df["element"] = quant_channel
        cell_df["reference"] = concentration_reference
        cell_df["method"] = method

        summary_cols = st.columns(4)
        summary_cols[0].metric(
            "Full cells",
            str(len(cell_df)),
        )
        summary_cols[1].metric(
            "Mean cell concentration",
            f"{float(cell_df['mean'].mean()):.4g} µg/cm²",
        )
        summary_cols[2].metric(
            "Median cell concentration",
            f"{float(cell_df['median'].median()):.4g} µg/cm²",
        )
        summary_cols[3].metric(
            "Highest cell maximum",
            f"{float(cell_df['max'].max()):.4g} µg/cm²",
        )

        st.dataframe(
            cell_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "`sum_of_pixel_values` is not total elemental mass. "
            "Physical X/Y pixel area is still not assigned a verified "
            "physical unit."
        )
    else:
        st.warning(
            "No complete cells were available under the current TFY "
            "segmentation settings."
        )

    with st.expander(
        "Automatic quantification provenance",
        expanded=False,
    ):
        st.json(
            {
                "concentration": auto_product.provenance,
                "beam_normalization_US_IC": us_product.provenance,
                "beam_normalization_DS_IC": ds_product.provenance,
            }
        )

st.divider()
st.caption(
    "The Explorer preserves raw data and makes normalization explicit. Feature extraction, "
    "cell segmentation, and inferential statistics are intentionally deferred to later "
    "validated analysis modules."
)
