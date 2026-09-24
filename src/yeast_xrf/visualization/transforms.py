"""Display-only transforms for scalar XRF maps.

These functions are for visualization. They never replace or mutate the scientific source
array. Derived scientific features belong in the future feature-analysis modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import exposure


@dataclass(frozen=True)
class DisplayTransformResult:
    data: np.ndarray
    label: str
    parameters: dict[str, float | str]


def _finite(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return arr[np.isfinite(arr)]


def percentile_stretch(
    data: np.ndarray,
    *,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Clip to robust percentiles and scale to [0, 1]."""
    arr = np.asarray(data, dtype=float)
    finite = _finite(arr)
    if finite.size == 0:
        return np.full(arr.shape, np.nan, dtype=float)
    if not 0 <= low < high <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros(arr.shape, dtype=float)
        out[~np.isfinite(arr)] = np.nan
        return out
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def signed_log1p(data: np.ndarray) -> np.ndarray:
    """Signed log transform that preserves negative fitted values."""
    arr = np.asarray(data, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def gamma_display(
    data: np.ndarray,
    *,
    gamma: float = 0.7,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Gamma transform applied to a percentile-normalized display copy."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    stretched = percentile_stretch(data, low=low, high=high)
    return np.power(stretched, gamma)



def clahe_display(
    data: np.ndarray,
    *,
    clip_limit: float = 0.01,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    '''Display-only CLAHE after robust percentile scaling.'''
    if clip_limit <= 0:
        raise ValueError("clip_limit must be > 0")
    stretched = percentile_stretch(data, low=low, high=high)
    finite = np.isfinite(stretched)
    work = np.where(finite, stretched, 0.0)
    out = np.asarray(
        exposure.equalize_adapthist(work, clip_limit=float(clip_limit)),
        dtype=float,
    )
    out[~finite] = np.nan
    return out


def display_transform(
    data: np.ndarray,
    mode: str,
    *,
    low: float = 1.0,
    high: float = 99.0,
    gamma: float = 0.7,
) -> DisplayTransformResult:
    """Apply a named display transform without altering the input."""
    mode_key = mode.strip().lower()
    arr = np.asarray(data, dtype=float)

    if mode_key == "raw":
        return DisplayTransformResult(
            data=arr.copy(),
            label="Raw",
            parameters={"mode": "raw"},
        )
    if mode_key in {"percentile", "percentile stretch"}:
        return DisplayTransformResult(
            data=percentile_stretch(arr, low=low, high=high),
            label=f"Percentile stretch ({low:g}-{high:g}%)",
            parameters={"mode": "percentile", "low": low, "high": high},
        )
    if mode_key in {"signed log1p", "log", "log1p"}:
        return DisplayTransformResult(
            data=signed_log1p(arr),
            label="Signed log1p",
            parameters={"mode": "signed_log1p"},
        )
    if mode_key == "gamma":
        return DisplayTransformResult(
            data=gamma_display(arr, gamma=gamma, low=low, high=high),
            label=f"Gamma display (γ={gamma:g}; {low:g}-{high:g}%)",
            parameters={
                "mode": "gamma",
                "gamma": gamma,
                "low": low,
                "high": high,
            },
        )
    if mode_key in {"clahe", "adaptive histogram"}:
        return DisplayTransformResult(
            data=clahe_display(arr, low=low, high=high),
            label=f"CLAHE display ({low:g}-{high:g}%)",
            parameters={"mode": "clahe", "low": low, "high": high},
        )
    raise ValueError(f"unknown display transform: {mode}")
