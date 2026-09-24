"""Read-only forensic inspection of MAPS quantification/calibration datasets.

Change 10 starts by decoding the stored MAPS semantics from representative files.
This module intentionally does NOT unlock automatic concentration conversion.

Direct HDF5 access stays in the IO layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


VALID_METHODS = ("Fitted", "NNLS", "ROI")
VALID_REFERENCES = ("US_IC", "DS_IC", "SR_Current", "US_FM")


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pythonize(value: Any) -> Any:
    """Convert HDF5/NumPy values to JSON-safe Python objects.

    Sequence handling must happen before np.asarray(); otherwise a list of
    byte strings becomes another NumPy byte-string array and can recurse
    forever when converted back to a list.
    """
    if isinstance(value, (list, tuple)):
        return [_pythonize(v) for v in value]

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _pythonize(value.item())
        return [_pythonize(v) for v in value.tolist()]

    if isinstance(value, (bytes, np.bytes_, np.generic)):
        return _decode(value)

    return value


def _pythonize_sequence(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_pythonize_sequence(v) for v in value]
    return _decode(value)


def _attrs(dataset: h5py.Dataset) -> dict[str, Any]:
    return {
        str(k): _pythonize_sequence(v.tolist() if hasattr(v, "tolist") else v)
        for k, v in dataset.attrs.items()
    }


def _dataset_record(h5: h5py.File, path: str, *, include_values: bool = True):
    if path not in h5:
        return None
    ds = h5[path]
    record = {
        "path": path,
        "shape": list(ds.shape),
        "dtype": str(ds.dtype),
        "attrs": _attrs(ds),
    }
    if include_values:
        record["values"] = _pythonize(ds[()])
    return record


def _numeric_summary(values: Any) -> dict[str, Any]:
    if values is None:
        return {
            "finite_count": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return {
            "finite_count": 0,
            "min": None,
            "median": None,
            "max": None,
        }

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _row_summaries(values: Any) -> list[dict[str, Any]]:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return []

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(arr.shape[0], -1)

    out = []
    for i, row in enumerate(arr):
        finite = row[np.isfinite(row)]
        out.append(
            {
                "row": int(i),
                "summary": _numeric_summary(row),
                "first_values": [
                    float(v) if np.isfinite(v) else None
                    for v in row[: min(24, row.size)]
                ],
                "nonzero_count": int(np.count_nonzero(finite))
                if finite.size
                else 0,
            }
        )
    return out


def inspect_calibration_semantics(
    path,
    *,
    method: str = "Fitted",
    reference: str = "US_IC",
) -> dict[str, Any]:
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}")
    if reference not in VALID_REFERENCES:
        raise ValueError(f"reference must be one of {VALID_REFERENCES}")

    path = Path(path)
    base = f"/MAPS/Quantification/Calibration/{method}"
    ref = f"{base}/{reference}"

    with h5py.File(path, "r") as h5:
        channel_names = _dataset_record(
            h5,
            f"/MAPS/XRF_Analyzed/{method}/Channel_Names",
        )
        channel_units = _dataset_record(
            h5,
            f"/MAPS/XRF_Analyzed/{method}/Channel_Units",
        )

        standard_paths = (
            "/MAPS/Quantification/Number_Of_Standards",
            "/MAPS/Quantification/Standard0/Standard_Name",
            "/MAPS/Quantification/Standard0/Element_Weights_Names",
            "/MAPS/Quantification/Standard0/Element_Weights",
            "/MAPS/Quantification/Standard0/Scalers/US_IC",
            "/MAPS/Quantification/Standard0/Scalers/DS_IC",
            "/MAPS/Quantification/Standard0/Scalers/SR_Current",
            "/MAPS/Quantification/Standard0/Scalers/US_FM",
        )

        standard = {
            p.rsplit("/", 1)[-1]: _dataset_record(h5, p)
            for p in standard_paths
            if p in h5
        }

        calibration = {
            "curve_labels": _dataset_record(
                h5,
                f"{base}/Calibration_Curve_Labels",
            ),
            "curve": _dataset_record(
                h5,
                f"{base}/Calibration_Curve_{reference}",
            ),
            "element_info_index": _dataset_record(
                h5,
                f"{ref}_Element_Info_Index",
            ),
            "element_info_names": _dataset_record(
                h5,
                f"{ref}_Element_Info_Names",
            ),
            "element_info_values": _dataset_record(
                h5,
                f"{ref}_Element_Info_Values",
            ),
        }

        quant_map = {
            "Fitted": (
                "/MAPS/XRF_fits_quant_names",
                "/MAPS/XRF_fits_quant",
            ),
            "NNLS": (
                "/MAPS/XRF_roi_plus_quant_names",
                "/MAPS/XRF_roi_plus_quant",
            ),
            "ROI": (
                "/MAPS/XRF_roi_quant_names",
                "/MAPS/XRF_roi_quant",
            ),
        }
        quant_names_path, quant_values_path = quant_map[method]
        quantification = {
            "names": _dataset_record(h5, quant_names_path),
            "values": _dataset_record(h5, quant_values_path),
        }

    result = {
        "source_path": str(path),
        "read_only": True,
        "method": method,
        "reference": reference,
        "channel_names": channel_names,
        "channel_units": channel_units,
        "standard": standard,
        "calibration": calibration,
        "quantification": quantification,
    }

    curve_values = (
        calibration["curve"]["values"]
        if calibration["curve"] is not None
        else None
    )
    info_values = (
        calibration["element_info_values"]["values"]
        if calibration["element_info_values"] is not None
        else None
    )
    quant_values = (
        quantification["values"]["values"]
        if quantification["values"] is not None
        else None
    )

    result["numeric_fingerprints"] = {
        "calibration_curve": {
            "summary": _numeric_summary(curve_values),
            "rows": _row_summaries(curve_values),
        },
        "element_info_values": {
            "summary": _numeric_summary(info_values),
            "rows": _row_summaries(info_values),
        },
        "quantification_values": {
            "summary": _numeric_summary(quant_values),
            "rows": _row_summaries(quant_values),
        },
    }
    return result


def compact_signature(report: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, JSON-friendly signature for cross-scan comparison."""
    calibration = report["calibration"]
    quantification = report["quantification"]
    return {
        "method": report["method"],
        "reference": report["reference"],
        "channel_names": (
            report["channel_names"]["values"]
            if report["channel_names"] is not None
            else None
        ),
        "channel_units": (
            report["channel_units"]["values"]
            if report["channel_units"] is not None
            else None
        ),
        "curve_labels": (
            calibration["curve_labels"]["values"]
            if calibration["curve_labels"] is not None
            else None
        ),
        "element_info_index": (
            calibration["element_info_index"]["values"]
            if calibration["element_info_index"] is not None
            else None
        ),
        "element_info_names": (
            calibration["element_info_names"]["values"]
            if calibration["element_info_names"] is not None
            else None
        ),
        "quant_names": (
            quantification["names"]["values"]
            if quantification["names"] is not None
            else None
        ),
        "curve_shape": (
            calibration["curve"]["shape"]
            if calibration["curve"] is not None
            else None
        ),
        "element_info_shape": (
            calibration["element_info_values"]["shape"]
            if calibration["element_info_values"] is not None
            else None
        ),
        "quant_shape": (
            quantification["values"]["shape"]
            if quantification["values"] is not None
            else None
        ),
    }
