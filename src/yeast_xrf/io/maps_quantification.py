'''Read-only MAPS quantification metadata inspection.

This module intentionally does not turn calibration-looking arrays into concentration maps
until their semantics are explicitly established. Direct HDF5 access stays in the IO layer.
'''

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


VALID_METHODS = ("Fitted", "NNLS", "ROI")
VALID_REFERENCES = ("US_IC", "DS_IC", "SR_Current", "US_FM")


def _decode_scalar(value: Any):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_python(value: Any):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _decode_scalar(arr.item())
    if arr.dtype.kind in {"S", "O", "U"}:
        return [_to_python(item) for item in arr.tolist()]
    return arr.tolist()


def _read_optional(h5: h5py.File, path: str):
    if path not in h5:
        return None
    return _to_python(h5[path][()])


def _flatten_text(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes, np.bytes_)):
        return [str(_decode_scalar(value))]
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(_flatten_text(key))
            out.extend(_flatten_text(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_text(item))
        return out
    return [str(value)]


def _shape_of(value):
    if value is None:
        return None
    return list(np.asarray(value, dtype=object).shape)


def _unit_candidates(report):
    text = " | ".join(_flatten_text(report))
    candidates = []
    patterns = (
        (r"(?:µ|μ|u)\s*g\s*/\s*cm(?:\^?2|²)", "µg/cm²"),
        (r"micrograms?\s+per\s+square\s+centimeter", "µg/cm²"),
        (r"mg\s*/\s*cm(?:\^?2|²)", "mg/cm²"),
        (r"ng\s*/\s*cm(?:\^?2|²)", "ng/cm²"),
    )
    for pattern, canonical in patterns:
        if re.search(pattern, text, flags=re.I):
            candidates.append(canonical)
    return sorted(set(candidates))


def inspect_maps_quantification(
    path,
    *,
    method="Fitted",
    reference="US_IC",
):
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}")
    if reference not in VALID_REFERENCES:
        raise ValueError(f"reference must be one of {VALID_REFERENCES}")

    path = Path(path)
    base = f"/MAPS/Quantification/Calibration/{method}"
    ref_base = f"{base}/{reference}"

    with h5py.File(path, "r") as h5:
        standard = {
            "number_of_standards": _read_optional(
                h5, "/MAPS/Quantification/Number_Of_Standards"
            ),
            "standard_name": _read_optional(
                h5, "/MAPS/Quantification/Standard0/Standard_Name"
            ),
            "element_weight_names": _read_optional(
                h5, "/MAPS/Quantification/Standard0/Element_Weights_Names"
            ),
            "element_weights": _read_optional(
                h5, "/MAPS/Quantification/Standard0/Element_Weights"
            ),
            "standard_us_ic": _read_optional(
                h5, "/MAPS/Quantification/Standard0/Scalers/US_IC"
            ),
            "standard_ds_ic": _read_optional(
                h5, "/MAPS/Quantification/Standard0/Scalers/DS_IC"
            ),
        }

        calibration = {
            "method": method,
            "reference": reference,
            "curve_labels": _read_optional(
                h5, f"{base}/Calibration_Curve_Labels"
            ),
            "curve": _read_optional(
                h5, f"{base}/Calibration_Curve_{reference}"
            ),
            "element_info_index": _read_optional(
                h5, f"{ref_base}_Element_Info_Index"
            ),
            "element_info_names": _read_optional(
                h5, f"{ref_base}_Element_Info_Names"
            ),
            "element_info_values": _read_optional(
                h5, f"{ref_base}_Element_Info_Values"
            ),
        }

        quant_candidates = {
            "fits_quant_names": _read_optional(h5, "/MAPS/XRF_fits_quant_names"),
            "fits_quant": _read_optional(h5, "/MAPS/XRF_fits_quant"),
            "roi_plus_quant_names": _read_optional(
                h5, "/MAPS/XRF_roi_plus_quant_names"
            ),
            "roi_plus_quant": _read_optional(h5, "/MAPS/XRF_roi_plus_quant"),
            "roi_quant_names": _read_optional(h5, "/MAPS/XRF_roi_quant_names"),
            "roi_quant": _read_optional(h5, "/MAPS/XRF_roi_quant"),
        }

    report = {
        "source_path": str(path),
        "read_only": True,
        "standard": standard,
        "calibration": calibration,
        "quantification_candidates": quant_candidates,
        "shapes": {
            "calibration_curve": _shape_of(calibration["curve"]),
            "calibration_labels": _shape_of(calibration["curve_labels"]),
            "element_info_values": _shape_of(calibration["element_info_values"]),
            "fits_quant": _shape_of(quant_candidates["fits_quant"]),
            "roi_plus_quant": _shape_of(quant_candidates["roi_plus_quant"]),
            "roi_quant": _shape_of(quant_candidates["roi_quant"]),
        },
    }

    units = _unit_candidates(report)
    report["unit_candidates"] = units

    has_standard = (
        standard["number_of_standards"] is not None
        and standard["element_weight_names"] is not None
        and standard["element_weights"] is not None
    )
    has_curve = (
        calibration["curve"] is not None
        and calibration["curve_labels"] is not None
    )
    has_element_info = (
        calibration["element_info_names"] is not None
        and calibration["element_info_values"] is not None
    )

    report["readiness"] = {
        "standard_metadata_present": bool(has_standard),
        "calibration_curve_present": bool(has_curve),
        "element_info_present": bool(has_element_info),
        "areal_density_unit_detected": bool(units),
        "automatic_areal_density_conversion_ready": False,
        "reason": (
            "Calibration metadata is available for inspection, but Change 09 does not "
            "assume which stored coefficient converts the selected fitted channel to "
            "areal density. Automatic conversion remains locked until that mapping is "
            "verified against MAPS semantics or a trusted representative export."
        ),
    }
    return report
