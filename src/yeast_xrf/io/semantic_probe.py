"""Read-only semantic probing for the MAPS-style yeast XRF HDF5 files.

This module reads only explicitly allowlisted small metadata/vector datasets so channel names,
geometry, scalers, theta, energy calibration, and acquisition metadata can be interpreted
without loading the large XRF cubes. Raw HDF5 files are never modified.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MAX_SMALL_DATASET_ELEMENTS = 20_000

SMALL_DATASETS: tuple[str, ...] = (
    "/MAPS/version",
    "/MAPS/Scan/name",
    "/MAPS/Scan/scan_type",
    "/MAPS/Scan/scan_time_stamp",
    "/MAPS/Scan/theta",
    "/MAPS/Scan/requested_rows",
    "/MAPS/Scan/requested_cols",
    "/MAPS/Scan/x_axis",
    "/MAPS/Scan/y_axis",
    "/MAPS/Scan/Extra_PVs/Names",
    "/MAPS/Scan/Extra_PVs/Description",
    "/MAPS/Scan/Extra_PVs/Unit",
    "/MAPS/Scan/Extra_PVs/Values",
    "/MAPS/XRF_Analyzed/Fitted/Channel_Names",
    "/MAPS/XRF_Analyzed/Fitted/Channel_Units",
    "/MAPS/XRF_Analyzed/NNLS/Channel_Names",
    "/MAPS/XRF_Analyzed/NNLS/Channel_Units",
    "/MAPS/XRF_Analyzed/ROI/Channel_Names",
    "/MAPS/XRF_Analyzed/ROI/Channel_Units",
    "/MAPS/Scalers/Names",
    "/MAPS/Scalers/Units",
    "/MAPS/scaler_names",
    "/MAPS/scaler_units",
    "/MAPS/Spectra/Energy",
    "/MAPS/Spectra/Energy_Calibration",
    "/MAPS/Quantification/Number_Of_Standards",
    "/MAPS/Quantification/Standard0/Standard_Name",
    "/MAPS/Quantification/Standard0/Element_Weights",
    "/MAPS/Quantification/Standard0/Element_Weights_Names",
    "/MAPS/XRF_fits_quant_names",
    "/MAPS/XRF_roi_plus_quant_names",
    "/MAPS/XRF_roi_quant_names",
)

LARGE_DATASET_METADATA: tuple[str, ...] = (
    "/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec",
    "/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec",
    "/MAPS/XRF_Analyzed/ROI/Counts_Per_Sec",
    "/MAPS/Spectra/mca_arr",
    "/MAPS/Scalers/Values",
    "/MAPS/scalers",
    "/MAPS/Spectra/Elapsed_Livetime",
    "/MAPS/Spectra/Elapsed_Realtime",
    "/MAPS/Spectra/Input_Counts",
    "/MAPS/Spectra/Output_Counts",
)

TOMO_TERMS = re.compile(
    r"(theta|angle|rotation|projection|tomo|tomography|rotary|stage.*rot)",
    flags=re.IGNORECASE,
)
Z_TERMS = re.compile(
    r"(^|[^a-z])(z|depth|slice|section)([^a-z]|$)",
    flags=re.IGNORECASE,
)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.generic):
        return _decode(value.item())
    if isinstance(value, np.ndarray):
        return [_decode(v) for v in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _decode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _flatten_singleton(value: Any) -> Any:
    value = _decode(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _as_list(value: Any) -> list[Any]:
    value = _decode(value)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_float(value: Any) -> float | None:
    value = _flatten_singleton(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _string_list(value: Any) -> list[str]:
    # Non-empty strings when positional alignment is irrelevant.
    return [str(v).strip() for v in _as_list(value) if str(v).strip()]


def _string_slots(value: Any) -> list[str]:
    # Preserve every slot, including blanks, for index-aligned metadata.
    return [str(v).strip() for v in _as_list(value)]


def _dataset_meta(handle: h5py.File, path: str) -> dict[str, Any] | None:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        return None
    ds = handle[path]
    return {
        "path": path,
        "shape": [int(v) for v in ds.shape],
        "ndim": int(ds.ndim),
        "dtype": str(ds.dtype),
        "size": int(ds.size),
        "attrs": {str(k): _decode(v) for k, v in ds.attrs.items()},
    }


def read_small_dataset(
    handle: h5py.File,
    path: str,
    *,
    max_elements: int = MAX_SMALL_DATASET_ELEMENTS,
) -> Any:
    """Read one allowlisted small dataset and refuse unexpected large payloads."""
    if path not in SMALL_DATASETS:
        raise ValueError(f"dataset is not allowlisted for small-data reads: {path}")
    if path not in handle:
        return None
    ds = handle[path]
    if not isinstance(ds, h5py.Dataset):
        return None
    if int(ds.size) > int(max_elements):
        raise ValueError(
            f"refusing to read {path}: {ds.size} elements exceeds safe limit {max_elements}"
        )
    return _decode(ds[()])


def axis_summary(values: Any) -> dict[str, Any]:
    raw = _as_list(values)
    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return {"count": len(raw), "numeric": False, "values_preview": raw[:8]}

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "numeric": True, "finite_count": 0}

    diffs = np.diff(finite)
    nonzero = diffs[np.abs(diffs) > 0]
    median_step = float(np.median(nonzero)) if nonzero.size else None
    abs_steps = np.abs(nonzero)
    step_cv = (
        float(np.std(abs_steps) / np.mean(abs_steps))
        if abs_steps.size and np.mean(abs_steps) != 0
        else 0.0 if abs_steps.size else None
    )
    return {
        "count": int(arr.size),
        "numeric": True,
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "extent": float(np.max(finite) - np.min(finite)),
        "first": float(finite[0]),
        "last": float(finite[-1]),
        "median_step_signed": median_step,
        "median_step_abs": None if median_step is None else abs(median_step),
        "step_cv_abs": step_cv,
        "approximately_uniform": None if step_cv is None else bool(step_cv <= 1e-3),
        "monotonic_increasing": bool(np.all(diffs >= 0)) if diffs.size else True,
        "monotonic_decreasing": bool(np.all(diffs <= 0)) if diffs.size else True,
    }


def _zip_metadata_columns(names: Any, descriptions: Any, units: Any, values: Any) -> list[dict[str, Any]]:
    n, d, u, v = map(_as_list, (names, descriptions, units, values))
    width = max(len(n), len(d), len(u), len(v), 0)
    return [
        {
            "index": i,
            "name": str(n[i]).strip() if i < len(n) else "",
            "description": str(d[i]).strip() if i < len(d) else "",
            "unit": str(u[i]).strip() if i < len(u) else "",
            "value": v[i] if i < len(v) else None,
        }
        for i in range(width)
    ]


def probe_scan(path: str | Path) -> dict[str, Any]:
    """Read semantic metadata from one MAPS HDF5 file."""
    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        small = {key: read_small_dataset(handle, key) for key in SMALL_DATASETS}
        large_meta = {
            key: meta
            for key in LARGE_DATASET_METADATA
            if (meta := _dataset_meta(handle, key)) is not None
        }

    x_info = axis_summary(small["/MAPS/Scan/x_axis"])
    y_info = axis_summary(small["/MAPS/Scan/y_axis"])
    fitted_meta = large_meta.get("/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec")
    if fitted_meta and len(fitted_meta["shape"]) >= 3:
        channel_count, y_pixels, x_pixels = fitted_meta["shape"][-3:]
    else:
        channel_count = y_pixels = x_pixels = None

    extra_pvs = _zip_metadata_columns(
        small["/MAPS/Scan/Extra_PVs/Names"],
        small["/MAPS/Scan/Extra_PVs/Description"],
        small["/MAPS/Scan/Extra_PVs/Unit"],
        small["/MAPS/Scan/Extra_PVs/Values"],
    )
    energy_info = axis_summary(small["/MAPS/Spectra/Energy"])

    return {
        "source": str(source),
        "source_name": source.name,
        "source_size_bytes": source.stat().st_size,
        "maps_version": _flatten_singleton(small["/MAPS/version"]),
        "scan": {
            "name": _flatten_singleton(small["/MAPS/Scan/name"]),
            "scan_type": _flatten_singleton(small["/MAPS/Scan/scan_type"]),
            "timestamp": _flatten_singleton(small["/MAPS/Scan/scan_time_stamp"]),
            "theta": _as_float(small["/MAPS/Scan/theta"]),
            "requested_rows": _flatten_singleton(small["/MAPS/Scan/requested_rows"]),
            "requested_cols": _flatten_singleton(small["/MAPS/Scan/requested_cols"]),
        },
        "geometry": {
            "x": x_info,
            "y": y_info,
            "x_axis_values": _as_list(small["/MAPS/Scan/x_axis"]),
            "y_axis_values": _as_list(small["/MAPS/Scan/y_axis"]),
            "fitted_raster_shape_yx": [int(y_pixels), int(x_pixels)] if y_pixels is not None else None,
            "fitted_channel_count": channel_count,
            "x_axis_matches_raster": x_info.get("count") == x_pixels if x_pixels is not None else None,
            "y_axis_matches_raster": y_info.get("count") == y_pixels if y_pixels is not None else None,
            "coordinate_units": "not established by current probe",
        },
        "channels": {
            method: {
                "names": _string_slots(small[f"/MAPS/XRF_Analyzed/{method}/Channel_Names"]),
                "units": _string_slots(small[f"/MAPS/XRF_Analyzed/{method}/Channel_Units"]),
                "counts_shape": large_meta.get(f"/MAPS/XRF_Analyzed/{method}/Counts_Per_Sec", {}).get("shape"),
            }
            for method in ("Fitted", "NNLS", "ROI")
        },
        "scalers": {
            "names_46": _string_slots(small["/MAPS/Scalers/Names"]),
            "units_46": _string_slots(small["/MAPS/Scalers/Units"]),
            "values_shape_46": large_meta.get("/MAPS/Scalers/Values", {}).get("shape"),
            "names_17": _string_slots(small["/MAPS/scaler_names"]),
            "units_17": _string_slots(small["/MAPS/scaler_units"]),
            "values_shape_17": large_meta.get("/MAPS/scalers", {}).get("shape"),
        },
        "spectra": {
            "energy": energy_info,
            "energy_calibration": _decode(small["/MAPS/Spectra/Energy_Calibration"]),
            "mca_shape": large_meta.get("/MAPS/Spectra/mca_arr", {}).get("shape"),
            "elapsed_livetime_shape": large_meta.get("/MAPS/Spectra/Elapsed_Livetime", {}).get("shape"),
            "elapsed_realtime_shape": large_meta.get("/MAPS/Spectra/Elapsed_Realtime", {}).get("shape"),
            "input_counts_shape": large_meta.get("/MAPS/Spectra/Input_Counts", {}).get("shape"),
            "output_counts_shape": large_meta.get("/MAPS/Spectra/Output_Counts", {}).get("shape"),
        },
        "quantification": {
            "number_of_standards": _flatten_singleton(small["/MAPS/Quantification/Number_Of_Standards"]),
            "standard_name": _flatten_singleton(small["/MAPS/Quantification/Standard0/Standard_Name"]),
            "element_weight_names": _string_list(small["/MAPS/Quantification/Standard0/Element_Weights_Names"]),
            "element_weights": _decode(small["/MAPS/Quantification/Standard0/Element_Weights"]),
            "fits_quant_names": _decode(small["/MAPS/XRF_fits_quant_names"]),
            "roi_plus_quant_names": _decode(small["/MAPS/XRF_roi_plus_quant_names"]),
            "roi_quant_names": _decode(small["/MAPS/XRF_roi_quant_names"]),
        },
        "extra_pvs": extra_pvs,
        "tomography_related_extra_pvs": [
            row for row in extra_pvs if TOMO_TERMS.search(f"{row['name']} {row['description']}")
        ],
        "z_related_extra_pvs": [
            row for row in extra_pvs if Z_TERMS.search(f"{row['name']} {row['description']}")
        ],
        "large_dataset_metadata": large_meta,
    }


def probe_directory(data_dir: str | Path) -> list[dict[str, Any]]:
    files = sorted(Path(data_dir).expanduser().resolve().glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no HDF5 files found under {data_dir}")
    return [probe_scan(path) for path in files]


def summarize_probes(probes: list[dict[str, Any]]) -> dict[str, Any]:
    name_variants = {method: Counter() for method in ("Fitted", "NNLS", "ROI")}
    unit_variants = {method: Counter() for method in ("Fitted", "NNLS", "ROI")}
    theta_values: list[float] = []
    geometry_rows: list[dict[str, Any]] = []

    for probe in probes:
        for method in ("Fitted", "NNLS", "ROI"):
            name_variants[method][tuple(probe["channels"][method]["names"])] += 1
            unit_variants[method][tuple(probe["channels"][method]["units"])] += 1
        theta = probe["scan"]["theta"]
        if theta is not None:
            theta_values.append(float(theta))
        geometry_rows.append(
            {
                "scan": probe["source_name"],
                "theta": theta,
                "raster_shape_yx": probe["geometry"]["fitted_raster_shape_yx"],
                "x_count": probe["geometry"]["x"].get("count"),
                "y_count": probe["geometry"]["y"].get("count"),
                "x_min": probe["geometry"]["x"].get("min"),
                "x_max": probe["geometry"]["x"].get("max"),
                "y_min": probe["geometry"]["y"].get("min"),
                "y_max": probe["geometry"]["y"].get("max"),
                "dx_median_abs": probe["geometry"]["x"].get("median_step_abs"),
                "dy_median_abs": probe["geometry"]["y"].get("median_step_abs"),
                "x_uniform": probe["geometry"]["x"].get("approximately_uniform"),
                "y_uniform": probe["geometry"]["y"].get("approximately_uniform"),
                "x_matches_raster": probe["geometry"]["x_axis_matches_raster"],
                "y_matches_raster": probe["geometry"]["y_axis_matches_raster"],
            }
        )

    unique_theta = sorted(set(theta_values))
    diffs = np.diff(np.asarray(theta_values, dtype=float)) if len(theta_values) > 1 else np.array([])
    theta_summary = {
        "count": len(theta_values),
        "unique_count": len(unique_theta),
        "unique_values": unique_theta,
        "min": min(theta_values) if theta_values else None,
        "max": max(theta_values) if theta_values else None,
        "span": max(theta_values) - min(theta_values) if theta_values else None,
        "monotonic_in_file_order": bool(np.all(diffs >= 0) or np.all(diffs <= 0)) if diffs.size else True,
        "all_same": len(unique_theta) <= 1,
    }

    if not theta_values:
        assessment = "No numeric theta values were read. The current files do not establish an angular tomographic series."
    elif len(unique_theta) == 1:
        assessment = "All readable theta values are identical. The presence of a theta field alone does not support tomography for these 12 scans."
    else:
        assessment = (
            "Theta varies across scans. This makes an angular-series hypothesis worth investigating, "
            "but does not prove tomography: specimen identity, projection geometry, coordinate frames, "
            "and angular coverage must still be established."
        )

    return {
        "scan_count": len(probes),
        "theta": theta_summary,
        "tomography_assessment": assessment,
        "channel_name_variants": {
            method: [{"names": list(sig), "scan_count": count} for sig, count in name_variants[method].most_common()]
            for method in name_variants
        },
        "channel_unit_variants": {
            method: [{"units": list(sig), "scan_count": count} for sig, count in unit_variants[method].most_common()]
            for method in unit_variants
        },
        "geometry": geometry_rows,
        "scaler_name_variants_46": [
            {"names": list(sig), "scan_count": count}
            for sig, count in Counter(tuple(p["scalers"]["names_46"]) for p in probes).most_common()
        ],
        "scaler_name_variants_17": [
            {"names": list(sig), "scan_count": count}
            for sig, count in Counter(tuple(p["scalers"]["names_17"]) for p in probes).most_common()
        ],
        "extra_pv_name_variants": [
            {"names": list(sig), "scan_count": count}
            for sig, count in Counter(tuple(row["name"] for row in p["extra_pvs"]) for p in probes).most_common()
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_semantic_reports(
    probes: list[dict[str, Any]], summary: dict[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_json = out / "semantic_scans.json"
    summary_json = out / "semantic_summary.json"
    geometry_csv = out / "scan_geometry_theta.csv"
    channels_csv = out / "channel_catalog.csv"
    scalers_csv = out / "scaler_catalog.csv"
    extra_pvs_csv = out / "extra_pvs.csv"
    report_md = out / "semantic_report.md"

    raw_json.write_text(json.dumps(probes, indent=2, sort_keys=True, default=str))
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))

    geometry_fields = [
        "scan", "theta", "raster_shape_yx", "x_count", "y_count", "x_min", "x_max",
        "y_min", "y_max", "dx_median_abs", "dy_median_abs", "x_uniform", "y_uniform",
        "x_matches_raster", "y_matches_raster",
    ]
    geometry_rows = []
    for row in summary["geometry"]:
        row = dict(row)
        row["raster_shape_yx"] = json.dumps(row["raster_shape_yx"])
        geometry_rows.append(row)
    _write_csv(geometry_csv, geometry_rows, geometry_fields)

    channel_rows = []
    for probe in probes:
        for method in ("Fitted", "NNLS", "ROI"):
            names = probe["channels"][method]["names"]
            units = probe["channels"][method]["units"]
            for idx, name in enumerate(names):
                channel_rows.append({
                    "scan": probe["source_name"], "method": method, "channel_index": idx,
                    "channel_name": name, "unit": units[idx] if idx < len(units) else "",
                })
    _write_csv(channels_csv, channel_rows, ["scan", "method", "channel_index", "channel_name", "unit"])

    scaler_rows = []
    for probe in probes:
        for family, name_key, unit_key in (
            ("MAPS_Scalers_46", "names_46", "units_46"),
            ("legacy_scalers_17", "names_17", "units_17"),
        ):
            names = probe["scalers"][name_key]
            units = probe["scalers"][unit_key]
            for idx, name in enumerate(names):
                scaler_rows.append({
                    "scan": probe["source_name"], "family": family, "scaler_index": idx,
                    "scaler_name": name, "unit": units[idx] if idx < len(units) else "",
                })
    _write_csv(scalers_csv, scaler_rows, ["scan", "family", "scaler_index", "scaler_name", "unit"])

    pv_rows = []
    for probe in probes:
        for row in probe["extra_pvs"]:
            pv_rows.append({
                "scan": probe["source_name"], "index": row["index"], "name": row["name"],
                "description": row["description"], "unit": row["unit"],
                "value": json.dumps(row["value"], default=str) if isinstance(row["value"], (list, dict)) else row["value"],
                "tomography_hint": bool(TOMO_TERMS.search(f"{row['name']} {row['description']}")),
                "z_hint": bool(Z_TERMS.search(f"{row['name']} {row['description']}")),
            })
    _write_csv(extra_pvs_csv, pv_rows, ["scan", "index", "name", "description", "unit", "value", "tomography_hint", "z_hint"])

    lines = [
        "# Yeast XRF semantic probe", "", f"- Scans probed: **{summary['scan_count']}**",
        "- Raw `.h5` files were opened read-only.",
        f"- Small-dataset safety limit: **{MAX_SMALL_DATASET_ELEMENTS:,} elements** per explicitly allowlisted dataset.",
        "", "## Tomography / theta", "", summary["tomography_assessment"], "",
        f"- Numeric theta values: **{summary['theta']['count']}**",
        f"- Unique theta values: **{summary['theta']['unique_count']}**",
        f"- Theta min/max: **{summary['theta']['min']} / {summary['theta']['max']}**",
        f"- Theta span: **{summary['theta']['span']}**", "",
        "| Scan | theta | raster Y×X | x range | y range | dx | dy | axes match raster |",
        "|---|---:|---|---|---|---:|---:|---|",
    ]
    for row in summary["geometry"]:
        lines.append(
            f"| `{row['scan']}` | {row['theta']} | {row['raster_shape_yx']} "
            f"| {row['x_min']} → {row['x_max']} | {row['y_min']} → {row['y_max']} "
            f"| {row['dx_median_abs']} | {row['dy_median_abs']} "
            f"| x={row['x_matches_raster']}, y={row['y_matches_raster']} |"
        )

    lines.extend(["", "## Analyzed XRF channels", ""])
    for method in ("Fitted", "NNLS", "ROI"):
        variants = summary["channel_name_variants"][method]
        lines.extend([f"### {method}", ""])
        if len(variants) == 1:
            variant = variants[0]
            lines.append(f"One channel-name list is shared by **{variant['scan_count']}** scans:")
            lines.append("")
            lines.append(", ".join(f"`{name}`" for name in variant["names"]))
        else:
            lines.append(f"Found **{len(variants)}** channel-name variants across the scans.")
            for i, variant in enumerate(variants, 1):
                lines.append(f"- Variant {i}, {variant['scan_count']} scan(s): " + ", ".join(f"`{name}`" for name in variant["names"]))
        lines.append("")

    lines.extend(["## Spectral structure", ""])
    for probe in probes:
        energy = probe["spectra"]["energy"]
        lines.append(
            f"- `{probe['source_name']}`: MCA `{probe['spectra']['mca_shape']}`, "
            f"energy {energy.get('min')} → {energy.get('max')}, median ΔE={energy.get('median_step_abs')}"
        )

    lines.extend(["", "## Scalers", ""])
    for label, key in (("46-channel MAPS scaler set", "scaler_name_variants_46"), ("17-channel scaler set", "scaler_name_variants_17")):
        variants = summary[key]
        lines.extend([f"### {label}", ""])
        if variants:
            lines.append(f"Most common list ({variants[0]['scan_count']} scan(s)): " + ", ".join(f"`{name}`" for name in variants[0]["names"]))
        lines.append("")

    lines.extend([
        "## Extra PV metadata", "",
        "The full Extra-PV table is in `extra_pvs.csv`. Angular/tomography-like and z/depth-like names are retained as hints only.",
        "", "## Interpretation boundary", "",
        "This probe establishes semantic metadata and array-axis relationships. It does not yet choose Fitted vs NNLS vs ROI as the preferred scientific map, assign every channel to a chemical element, or claim a tomographic/serial-section volume.", "",
    ])
    report_md.write_text("\n".join(lines))

    return {
        "semantic_scans_json": raw_json,
        "semantic_summary_json": summary_json,
        "scan_geometry_theta_csv": geometry_csv,
        "channel_catalog_csv": channels_csv,
        "scaler_catalog_csv": scalers_csv,
        "extra_pvs_csv": extra_pvs_csv,
        "semantic_report_md": report_md,
    }
