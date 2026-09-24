#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from yeast_xrf.io.maps_calibration_semantics import (
    inspect_calibration_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT_JSON = ROOT / "analysis" / "quantification" / "change10_mapping_probe.json"
OUT_TXT = ROOT / "analysis" / "quantification" / "change10_mapping_probe.txt"
OUT_CSV = ROOT / "analysis" / "quantification" / "change10_mapping_probe_channels.csv"

METHODS = ("Fitted", "NNLS", "ROI")
REFERENCES = ("US_IC", "DS_IC", "SR_Current", "US_FM")
FOCUS_CHANNELS = (
    "P", "S", "K", "Ca", "Ti", "Cr", "Mn", "Fe", "Ni", "Cu", "Zn", "La_L"
)


def element_symbol(channel: str) -> str | None:
    if channel in {"P", "S", "K", "Ca", "Ti", "Cr", "Mn", "Fe", "Ni", "Cu", "Zn"}:
        return channel
    if channel.endswith("_L"):
        return channel[:-2]
    return None


def as_array(record):
    if not record:
        return None
    values = record.get("values")
    if values is None:
        return None
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None


def labels_array(record):
    if not record:
        return None
    values = record.get("values")
    if values is None:
        return None
    return np.asarray(values, dtype=object)


def safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def relative_error(a, b):
    a = safe_float(a)
    b = safe_float(b)
    if a is None or b is None:
        return None
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


def standard_records(report):
    names_rec = report["calibration"]["element_info_names"]
    values_rec = report["calibration"]["element_info_values"]
    index_rec = report["calibration"]["element_info_index"]

    if not names_rec or not values_rec:
        return []

    names = [str(x) for x in names_rec["values"]]
    values = np.asarray(values_rec["values"], dtype=float)
    indices = (
        list(index_rec["values"])
        if index_rec and index_rec.get("values") is not None
        else [None] * values.shape[0]
    )

    rows = []
    for i in range(values.shape[0]):
        row = {"row": i, "element_info_index": indices[i] if i < len(indices) else None}
        for j, name in enumerate(names):
            if j < values.shape[1]:
                row[name] = safe_float(values[i, j])
        rows.append(row)
    return rows


def curve_candidates_for_symbol(report, symbol):
    labels = labels_array(report["calibration"]["curve_labels"])
    curve = as_array(report["calibration"]["curve"])
    if labels is None or curve is None:
        return []

    if labels.shape != curve.shape:
        return []

    hits = []
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            if str(labels[row, col]).strip() == symbol:
                hits.append(
                    {
                        "curve_row": row,
                        "curve_col": col,
                        "label": symbol,
                        "value": safe_float(curve[row, col]),
                    }
                )
    return hits


def infer_standard_curve_matches(report):
    matches = []
    for rec in standard_records(report):
        z = rec.get("Z")
        calib = rec.get("calib_curve_val")
        candidates = []

        labels = labels_array(report["calibration"]["curve_labels"])
        curve = as_array(report["calibration"]["curve"])
        if labels is not None and curve is not None and z is not None:
            zint = int(round(z))
            for col in (zint - 1, zint):
                if 0 <= col < curve.shape[1]:
                    for row in range(curve.shape[0]):
                        candidates.append(
                            {
                                "curve_row": row,
                                "curve_col": col,
                                "label": str(labels[row, col]),
                                "value": safe_float(curve[row, col]),
                                "relative_error_to_calib_curve_val": relative_error(
                                    curve[row, col], calib
                                ),
                            }
                        )

        candidates = sorted(
            candidates,
            key=lambda x: (
                1e99
                if x["relative_error_to_calib_curve_val"] is None
                else x["relative_error_to_calib_curve_val"]
            ),
        )

        matches.append(
            {
                "element_info_record": rec,
                "closest_curve_candidates": candidates[:4],
            }
        )
    return matches


def channel_rows(report, scan_name, method, reference):
    channels_rec = report["channel_names"]
    quant_rec = report["quantification"]["values"]
    if not channels_rec or not quant_rec:
        return []

    channels = [str(x) for x in channels_rec["values"]]
    quant = np.asarray(quant_rec["values"], dtype=float)

    # Legacy arrays in these data are expected to be (3, 1, channel),
    # but keep this probe robust to singleton or flattened middle axes.
    if quant.ndim == 3 and quant.shape[1] == 1:
        q = quant[:, 0, :]
    elif quant.ndim == 2:
        q = quant
    else:
        return []

    rows = []
    for channel in FOCUS_CHANNELS:
        if channel not in channels:
            continue
        idx = channels.index(channel)
        symbol = element_symbol(channel)
        candidates = curve_candidates_for_symbol(report, symbol) if symbol else []

        row = {
            "scan": scan_name,
            "method": method,
            "reference": reference,
            "channel": channel,
            "channel_index": idx,
            "element_symbol": symbol,
        }

        for qi in range(q.shape[0]):
            row[f"legacy_quant_row_{qi}"] = safe_float(q[qi, idx])

        for ci, candidate in enumerate(candidates):
            row[f"curve_candidate_{ci}_row"] = candidate["curve_row"]
            row[f"curve_candidate_{ci}_col"] = candidate["curve_col"]
            row[f"curve_candidate_{ci}_value"] = candidate["value"]

        rows.append(row)
    return rows


def main():
    files = sorted(DATA.glob("*.h5"))
    if not files:
        raise SystemExit(f"No HDF5 files found in {DATA}")

    first = files[0]
    payload = {
        "representative_scan": first.name,
        "automatic_concentration_enabled": False,
        "reason": (
            "This is a mapping probe. It identifies how standard records, "
            "calibration-curve rows, and legacy quantification arrays relate. "
            "It does not yet assert a conversion formula."
        ),
        "representative": {},
        "channel_rows": [],
        "cross_scan_standard_matches": [],
    }

    print("SOURDOUGH Change 10 · calibration mapping probe")
    print("=" * 118)
    print(f"Representative scan: {first.name}")
    print()

    for method in METHODS:
        payload["representative"][method] = {}
        for reference in REFERENCES:
            report = inspect_calibration_semantics(
                first,
                method=method,
                reference=reference,
            )
            matches = infer_standard_curve_matches(report)
            crows = channel_rows(
                report,
                first.name,
                method,
                reference,
            )

            payload["representative"][method][reference] = {
                "element_info_names": (
                    report["calibration"]["element_info_names"]["values"]
                    if report["calibration"]["element_info_names"]
                    else None
                ),
                "element_info_index": (
                    report["calibration"]["element_info_index"]["values"]
                    if report["calibration"]["element_info_index"]
                    else None
                ),
                "standard_records": standard_records(report),
                "standard_curve_matches": matches,
                "quant_names": (
                    report["quantification"]["names"]["values"]
                    if report["quantification"]["names"]
                    else None
                ),
                "quant_shape": (
                    report["quantification"]["values"]["shape"]
                    if report["quantification"]["values"]
                    else None
                ),
                "channel_rows": crows,
            }
            payload["channel_rows"].extend(crows)

    # Cross-scan check: do standard calib_curve_val records consistently map
    # to the same calibration-curve row/column logic?
    for path in files:
        for method in METHODS:
            for reference in REFERENCES:
                report = inspect_calibration_semantics(
                    path,
                    method=method,
                    reference=reference,
                )
                matches = infer_standard_curve_matches(report)
                payload["cross_scan_standard_matches"].append(
                    {
                        "scan": path.name,
                        "method": method,
                        "reference": reference,
                        "matches": matches,
                    }
                )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    )

    if payload["channel_rows"]:
        fields = sorted(
            {
                key
                for row in payload["channel_rows"]
                for key in row.keys()
            }
        )
        with OUT_CSV.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(payload["channel_rows"])

    lines = []
    lines.append("SOURDOUGH Change 10 · calibration mapping probe")
    lines.append("=" * 118)
    lines.append(f"Representative scan: {first.name}")
    lines.append("")

    for method in METHODS:
        lines.append(f"[{method}]")
        for reference in REFERENCES:
            block = payload["representative"][method][reference]
            lines.append(f"  {reference}")
            lines.append(f"    element_info_index = {block['element_info_index']}")
            lines.append(f"    quant_names        = {block['quant_names']}")
            lines.append(f"    quant_shape        = {block['quant_shape']}")
            lines.append("    standard records + closest curve matches:")
            for item in block["standard_curve_matches"]:
                rec = item["element_info_record"]
                closest = item["closest_curve_candidates"]
                lines.append(
                    "      "
                    + f"row={rec.get('row')} "
                    + f"Z={rec.get('Z')} "
                    + f"weight={rec.get('weight')} "
                    + f"e_cal_ratio={rec.get('e_cal_ratio')} "
                    + f"calib_curve_val={rec.get('calib_curve_val')}"
                )
                for cand in closest[:3]:
                    lines.append(
                        "        -> "
                        + f"curve[{cand['curve_row']},{cand['curve_col']}] "
                        + f"label={cand['label']} "
                        + f"value={cand['value']} "
                        + f"relerr={cand['relative_error_to_calib_curve_val']}"
                    )

            lines.append("    selected analyzed channels:")
            for row in block["channel_rows"]:
                qvals = [
                    row.get("legacy_quant_row_0"),
                    row.get("legacy_quant_row_1"),
                    row.get("legacy_quant_row_2"),
                ]
                cvals = [
                    row.get("curve_candidate_0_value"),
                    row.get("curve_candidate_1_value"),
                    row.get("curve_candidate_2_value"),
                ]
                lines.append(
                    "      "
                    + f"{row['channel']:<5} "
                    + f"quant={qvals} "
                    + f"curve_candidates={cvals}"
                )
        lines.append("")

    OUT_TXT.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[:220]))
    if len(lines) > 220:
        print(f"... output truncated in terminal; full text has {len(lines)} lines")

    print()
    print("-" * 118)
    print(f"JSON: {OUT_JSON.relative_to(ROOT)}")
    print(f"Text: {OUT_TXT.relative_to(ROOT)}")
    print(f"CSV:  {OUT_CSV.relative_to(ROOT)}")
    print()
    print("Automatic concentration conversion remains LOCKED.")


if __name__ == "__main__":
    main()
