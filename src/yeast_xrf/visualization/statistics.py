"""Descriptive statistics for XRF maps."""

from __future__ import annotations

from typing import Any

import numpy as np


def map_statistics(data: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(data, dtype=float)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]

    result: dict[str, Any] = {
        "shape": list(arr.shape),
        "pixel_count": int(arr.size),
        "finite_count": int(finite.size),
        "nan_or_inf_count": int(arr.size - finite.size),
    }
    if finite.size == 0:
        result.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "zero_count": 0,
                "negative_count": 0,
                "p01": None,
                "p05": None,
                "p25": None,
                "p75": None,
                "p95": None,
                "p99": None,
            }
        )
        return result

    p01, p05, p25, p75, p95, p99 = np.percentile(
        finite, [1, 5, 25, 75, 95, 99]
    )
    result.update(
        {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std": float(np.std(finite)),
            "zero_count": int(np.count_nonzero(finite == 0)),
            "negative_count": int(np.count_nonzero(finite < 0)),
            "p01": float(p01),
            "p05": float(p05),
            "p25": float(p25),
            "p75": float(p75),
            "p95": float(p95),
            "p99": float(p99),
        }
    )
    return result


def finite_shared_range(arrays: list[np.ndarray]) -> tuple[float, float] | None:
    finite_parts = [
        np.asarray(arr, dtype=float)[np.isfinite(np.asarray(arr, dtype=float))]
        for arr in arrays
    ]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return None
    values = np.concatenate(finite_parts)
    return float(np.min(values)), float(np.max(values))
