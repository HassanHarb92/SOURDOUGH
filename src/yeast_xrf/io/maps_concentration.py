"""Read-only access to verified MAPS calibration factors.

The legacy MAPS quantification arrays in this dataset encode three scaler
normalizations per analyzed channel:

    row 0 -> SR_Current
    row 1 -> US_IC
    row 2 -> DS_IC

Change 10 does not trust that mapping blindly. Each requested factor is
cross-checked against the corresponding MAPS calibration curve using the
exact analyzed channel label (e.g. Zn or La_L).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


METHOD_QUANT_DATASETS = {
    "Fitted": "/MAPS/XRF_fits_quant",
    "NNLS": "/MAPS/XRF_roi_plus_quant",
    "ROI": "/MAPS/XRF_roi_quant",
}

REFERENCE_ROWS = {
    "SR_Current": 0,
    "US_IC": 1,
    "DS_IC": 2,
}

SUPPORTED_METHODS = tuple(METHOD_QUANT_DATASETS)
SUPPORTED_REFERENCES = tuple(REFERENCE_ROWS)


@dataclass(frozen=True)
class MapsCalibrationFactor:
    factor: float
    unit: str
    method: str
    reference: str
    channel: str
    channel_index: int
    quant_dataset: str
    quant_row: int
    curve_dataset: str
    curve_row: int
    curve_col: int
    curve_value: float
    relative_error: float
    read_only: bool = True


def _decode_list(values):
    arr = np.asarray(values)
    result = []
    for value in arr.tolist():
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="replace").strip("\x00 "))
        else:
            result.append(str(value).strip("\x00 "))
    return result


def _relative_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


def read_maps_calibration_factor(
    path,
    *,
    method: str,
    reference: str,
    channel: str,
    rtol: float = 5e-5,
    atol: float = 1e-12,
) -> MapsCalibrationFactor:
    if method not in METHOD_QUANT_DATASETS:
        raise ValueError(
            f"method must be one of {tuple(METHOD_QUANT_DATASETS)}"
        )
    if reference not in REFERENCE_ROWS:
        raise ValueError(
            f"reference must be one of {tuple(REFERENCE_ROWS)}"
        )

    path = Path(path)
    quant_path = METHOD_QUANT_DATASETS[method]
    quant_row = REFERENCE_ROWS[reference]
    channel_path = f"/MAPS/XRF_Analyzed/{method}/Channel_Names"
    base = f"/MAPS/Quantification/Calibration/{method}"
    labels_path = f"{base}/Calibration_Curve_Labels"
    curve_path = f"{base}/Calibration_Curve_{reference}"

    with h5py.File(path, "r") as h5:
        required = (
            quant_path,
            channel_path,
            labels_path,
            curve_path,
        )
        missing = [p for p in required if p not in h5]
        if missing:
            raise KeyError(f"Missing MAPS quantification datasets: {missing}")

        channels = _decode_list(h5[channel_path][()])
        if channel not in channels:
            raise KeyError(
                f"Channel {channel!r} is not present for method {method!r}"
            )
        channel_index = channels.index(channel)

        quant = np.asarray(h5[quant_path][()], dtype=float)
        if quant.ndim != 3 or quant.shape[0] < 3:
            raise ValueError(
                f"Unexpected quantification shape {quant.shape} at {quant_path}"
            )
        if quant.shape[1] != 1:
            raise ValueError(
                f"Expected singleton standard axis in {quant_path}; "
                f"got shape {quant.shape}"
            )
        if channel_index >= quant.shape[2]:
            raise ValueError(
                f"Channel index {channel_index} exceeds quant array "
                f"shape {quant.shape}"
            )

        factor = float(quant[quant_row, 0, channel_index])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"No positive finite MAPS calibration factor for "
                f"{method}/{reference}/{channel}: {factor}"
            )

        labels_raw = np.asarray(h5[labels_path][()])
        curve = np.asarray(h5[curve_path][()], dtype=float)
        if labels_raw.shape != curve.shape:
            raise ValueError(
                f"Calibration labels shape {labels_raw.shape} does not "
                f"match curve shape {curve.shape}"
            )

        labels = np.empty(labels_raw.shape, dtype=object)
        it = np.nditer(
            labels_raw,
            flags=["multi_index", "refs_ok"],
            op_flags=["readonly"],
        )
        for item in it:
            value = item.item()
            if isinstance(value, bytes):
                value = value.decode(
                    "utf-8",
                    errors="replace",
                ).strip("\x00 ")
            else:
                value = str(value).strip("\x00 ")
            labels[it.multi_index] = value

        positions = np.argwhere(labels == channel)
        if positions.shape[0] != 1:
            raise ValueError(
                f"Expected exactly one calibration-curve label for "
                f"{channel!r}; found {positions.shape[0]}"
            )
        curve_row, curve_col = map(int, positions[0])
        curve_value = float(curve[curve_row, curve_col])
        relerr = _relative_error(factor, curve_value)

        if not np.isclose(
            factor,
            curve_value,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                "Legacy MAPS quant factor disagrees with calibration curve: "
                f"{method}/{reference}/{channel} factor={factor} "
                f"curve={curve_value} relative_error={relerr}"
            )

    return MapsCalibrationFactor(
        factor=factor,
        unit="normalized_signal_per_ug_cm2",
        method=method,
        reference=reference,
        channel=channel,
        channel_index=channel_index,
        quant_dataset=quant_path,
        quant_row=quant_row,
        curve_dataset=curve_path,
        curve_row=curve_row,
        curve_col=curve_col,
        curve_value=curve_value,
        relative_error=relerr,
        read_only=True,
    )


def quantifiable_channels(
    path,
    *,
    method: str,
    reference: str,
) -> list[str]:
    path = Path(path)
    channel_path = f"/MAPS/XRF_Analyzed/{method}/Channel_Names"
    labels_path = (
        f"/MAPS/Quantification/Calibration/{method}/"
        "Calibration_Curve_Labels"
    )

    with h5py.File(path, "r") as h5:
        channels = _decode_list(h5[channel_path][()])
        labels_raw = np.asarray(h5[labels_path][()])

    label_set = set()
    for value in labels_raw.reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        label_set.add(str(value).strip("\x00 "))

    out = []
    for channel in channels:
        if channel not in label_set:
            continue
        try:
            read_maps_calibration_factor(
                path,
                method=method,
                reference=reference,
                channel=channel,
            )
        except (KeyError, ValueError):
            continue
        out.append(channel)
    return out
