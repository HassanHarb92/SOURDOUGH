'''Canonical lazy-access model for MAPS-style XRF HDF5 scans.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import h5py
import numpy as np
import xarray as xr

from yeast_xrf.io.channel_roles import ChannelRole, channel_role
from yeast_xrf.provenance import file_identity


AnalysisMethod = Literal["Fitted", "NNLS", "ROI"]
ScalerFamily = Literal["primary", "legacy"]

_METHODS = ("Fitted", "NNLS", "ROI")
_METHOD_ALIASES = {"fitted": "Fitted", "fit": "Fitted", "nnls": "NNLS", "roi": "ROI"}

_PRIMARY = (
    "/MAPS/Scalers/Names",
    "/MAPS/Scalers/Units",
    "/MAPS/Scalers/Values",
)
_LEGACY = (
    "/MAPS/scaler_names",
    "/MAPS/scaler_units",
    "/MAPS/scalers",
)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.generic):
        return _decode(value.item())
    return value


def _slots(dataset: h5py.Dataset) -> tuple[str, ...]:
    raw = dataset[()]
    values = [raw] if np.ndim(raw) == 0 else np.asarray(raw).reshape(-1).tolist()
    return tuple(str(_decode(value)).strip() for value in values)


def _scalar(dataset: h5py.Dataset) -> Any:
    value = dataset[()]
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return _decode(value)


def normalize_method(method: str) -> AnalysisMethod:
    text = str(method).strip()
    if text in _METHODS:
        return text  # type: ignore[return-value]
    normalized = _METHOD_ALIASES.get(text.lower())
    if normalized is None:
        raise ValueError(f"unknown XRF analysis method {method!r}; expected {_METHODS}")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class ChannelInfo:
    index: int
    name: str
    unit: str
    role: ChannelRole


@dataclass(frozen=True)
class GeometryInfo:
    shape_yx: tuple[int, int]
    x_count: int
    y_count: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx_median_abs: float | None
    dy_median_abs: float | None
    x_uniform: bool
    y_uniform: bool
    coordinate_unit: str | None = None


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


class XRFScan:
    '''Lazy, read-only access to one MAPS-style XRF scan.'''

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._h5: h5py.File | None = None
        self._channels_cache: dict[str, tuple[ChannelInfo, ...]] = {}
        self._scaler_cache: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

    @classmethod
    def open(cls, path: str | Path) -> "XRFScan":
        return cls(path)

    def __enter__(self) -> "XRFScan":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._h5 is not None

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    @property
    def h5(self) -> h5py.File:
        return self._ensure_open()

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def source_name(self) -> str:
        return self.path.name

    @property
    def scan_name(self) -> str:
        return str(_scalar(self.h5["/MAPS/Scan/name"]))

    @property
    def scan_type(self) -> str:
        return str(_scalar(self.h5["/MAPS/Scan/scan_type"]))

    @property
    def timestamp(self) -> str:
        return str(_scalar(self.h5["/MAPS/Scan/scan_time_stamp"]))

    @property
    def theta(self) -> float:
        return float(_scalar(self.h5["/MAPS/Scan/theta"]))

    def lightweight_identity(self) -> dict[str, Any]:
        stat = self.path.stat()
        return {
            "path": str(self.path),
            "name": self.path.name,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def strong_identity(self) -> dict[str, Any]:
        return file_identity(self.path)

    @property
    def x(self) -> np.ndarray:
        return np.asarray(self.h5["/MAPS/Scan/x_axis"][...], dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.h5["/MAPS/Scan/y_axis"][...], dtype=float)

    @staticmethod
    def _axis_step(values: np.ndarray) -> tuple[float | None, bool]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size < 2:
            return None, True
        diffs = np.abs(np.diff(arr))
        finite = diffs[np.isfinite(diffs) & (diffs > 0)]
        if finite.size == 0:
            return None, True
        median = float(np.median(finite))
        cv = float(np.std(finite) / np.mean(finite)) if np.mean(finite) else 0.0
        return median, bool(cv <= 1e-3)

    @property
    def shape(self) -> tuple[int, int]:
        ds = self.h5["/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec"]
        return int(ds.shape[1]), int(ds.shape[2])

    @property
    def spatial_dims(self) -> tuple[str, str]:
        return ("y", "x")

    @property
    def geometry(self) -> GeometryInfo:
        x = self.x
        y = self.y
        dx, x_uniform = self._axis_step(x)
        dy, y_uniform = self._axis_step(y)
        return GeometryInfo(
            shape_yx=self.shape,
            x_count=int(x.size),
            y_count=int(y.size),
            x_min=float(np.nanmin(x)),
            x_max=float(np.nanmax(x)),
            y_min=float(np.nanmin(y)),
            y_max=float(np.nanmax(y)),
            dx_median_abs=dx,
            dy_median_abs=dy,
            x_uniform=x_uniform,
            y_uniform=y_uniform,
            coordinate_unit=None,
        )

    def nearest_pixel(self, *, x: float, y: float) -> tuple[int, int]:
        xi = int(np.nanargmin(np.abs(self.x - float(x))))
        yi = int(np.nanargmin(np.abs(self.y - float(y))))
        return yi, xi

    def channels(self, method: str = "Fitted") -> tuple[ChannelInfo, ...]:
        method = normalize_method(method)
        cached = self._channels_cache.get(method)
        if cached is not None:
            return cached
        names = _slots(self.h5[f"/MAPS/XRF_Analyzed/{method}/Channel_Names"])
        units = _slots(self.h5[f"/MAPS/XRF_Analyzed/{method}/Channel_Units"])
        if len(names) != len(units):
            raise ValueError(
                f"{method} channel metadata is misaligned: {len(names)} names vs {len(units)} units"
            )
        result = tuple(
            ChannelInfo(i, name, units[i], channel_role(name))
            for i, name in enumerate(names)
        )
        self._channels_cache[method] = result
        return result

    def channel_names(self, method: str = "Fitted") -> tuple[str, ...]:
        return tuple(item.name for item in self.channels(method))

    def channel_index(self, name: str, method: str = "Fitted") -> int:
        matches = [item.index for item in self.channels(method) if item.name == name]
        if not matches:
            raise KeyError(
                f"channel {name!r} not found for {normalize_method(method)}; "
                f"available: {', '.join(self.channel_names(method))}"
            )
        if len(matches) > 1:
            raise ValueError(f"channel name {name!r} is duplicated")
        return matches[0]

    def channel_info(self, name: str, method: str = "Fitted") -> ChannelInfo:
        idx = self.channel_index(name, method)
        return self.channels(method)[idx]

    def map(self, name: str, method: str = "Fitted") -> np.ndarray:
        '''Read exactly one [channel, :, :] analyzed-map slice.'''
        method = normalize_method(method)
        idx = self.channel_index(name, method)
        return np.asarray(
            self.h5[f"/MAPS/XRF_Analyzed/{method}/Counts_Per_Sec"][idx, :, :]
        )

    def map_xarray(self, name: str, method: str = "Fitted") -> xr.DataArray:
        method = normalize_method(method)
        info = self.channel_info(name, method)
        return xr.DataArray(
            self.map(name, method),
            dims=("y", "x"),
            coords={"y": self.y, "x": self.x},
            name=name,
            attrs={
                "unit": info.unit,
                "role": info.role.value,
                "analysis_method": method,
                "source_file": str(self.path),
                "coordinate_unit": None,
            },
        )

    def map_stack(self, names: Iterable[str], method: str = "Fitted") -> xr.DataArray:
        method = normalize_method(method)
        selected = tuple(names)
        data = np.stack([self.map(name, method) for name in selected], axis=0)
        return xr.DataArray(
            data,
            dims=("channel", "y", "x"),
            coords={"channel": selected, "y": self.y, "x": self.x},
            attrs={"analysis_method": method, "source_file": str(self.path)},
        )

    @property
    def energy(self) -> np.ndarray:
        return np.asarray(self.h5["/MAPS/Spectra/Energy"][...], dtype=float)

    @property
    def energy_calibration(self) -> np.ndarray:
        return np.asarray(self.h5["/MAPS/Spectra/Energy_Calibration"][...], dtype=float)

    @property
    def spectral_shape(self) -> tuple[int, int, int]:
        shape = self.h5["/MAPS/Spectra/mca_arr"].shape
        return int(shape[0]), int(shape[1]), int(shape[2])

    def spectrum(self, *, y: int, x: int) -> np.ndarray:
        '''Read exactly one [energy, y, x] spectrum vector.'''
        y_size, x_size = self.shape
        if not (0 <= int(y) < y_size and 0 <= int(x) < x_size):
            raise IndexError(f"pixel (y={y}, x={x}) outside raster shape {self.shape}")
        return np.asarray(self.h5["/MAPS/Spectra/mca_arr"][:, int(y), int(x)])

    def spectrum_xarray(self, *, y: int, x: int) -> xr.DataArray:
        return xr.DataArray(
            self.spectrum(y=y, x=x),
            dims=("energy",),
            coords={"energy": self.energy},
            name="xrf_spectrum",
            attrs={
                "source_file": str(self.path),
                "y_index": int(y),
                "x_index": int(x),
                "y_coordinate": float(self.y[int(y)]),
                "x_coordinate": float(self.x[int(x)]),
            },
        )

    def spectrum_at_coordinate(self, *, x: float, y: float) -> xr.DataArray:
        yi, xi = self.nearest_pixel(x=x, y=y)
        return self.spectrum_xarray(y=yi, x=xi)

    @property
    def integrated_spectrum(self) -> np.ndarray:
        return np.asarray(self.h5["/MAPS/Spectra/Integrated_Spectra/Spectra"][...])

    def _scaler_paths(self, family: ScalerFamily) -> tuple[str, str, str]:
        if family == "primary":
            return _PRIMARY
        if family == "legacy":
            return _LEGACY
        raise ValueError("scaler family must be 'primary' or 'legacy'")

    def scaler_metadata(
        self, family: ScalerFamily = "primary"
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        key = str(family)
        cached = self._scaler_cache.get(key)
        if cached is not None:
            return cached
        names_path, units_path, _ = self._scaler_paths(family)
        names = _slots(self.h5[names_path])
        units = _slots(self.h5[units_path])
        if len(names) != len(units):
            raise ValueError(
                f"{family} scaler metadata is misaligned: {len(names)} names vs {len(units)} units"
            )
        result = names, units
        self._scaler_cache[key] = result
        return result

    def scaler_names(self, family: ScalerFamily = "primary") -> tuple[str, ...]:
        return self.scaler_metadata(family)[0]

    def scaler_index(self, name: str, family: ScalerFamily = "primary") -> int:
        names = self.scaler_names(family)
        matches = [i for i, value in enumerate(names) if value == name]
        if not matches:
            raise KeyError(f"scaler {name!r} not found in {family} family")
        if len(matches) > 1:
            raise ValueError(f"scaler name {name!r} is duplicated")
        return matches[0]

    def scaler(self, name: str, family: ScalerFamily = "primary") -> np.ndarray:
        _, _, values_path = self._scaler_paths(family)
        idx = self.scaler_index(name, family)
        return np.asarray(self.h5[values_path][idx, :, :])

    def scaler_xarray(
        self, name: str, family: ScalerFamily = "primary"
    ) -> xr.DataArray:
        names, units = self.scaler_metadata(family)
        idx = self.scaler_index(name, family)
        return xr.DataArray(
            self.scaler(name, family),
            dims=("y", "x"),
            coords={"y": self.y, "x": self.x},
            name=name,
            attrs={
                "unit": units[idx],
                "scaler_family": family,
                "source_file": str(self.path),
                "coordinate_unit": None,
            },
        )

    def acquisition_summary(self) -> dict[str, Any]:
        return {
            "scan_name": self.scan_name,
            "scan_type": self.scan_type,
            "timestamp": self.timestamp,
            "theta": self.theta,
            "requested_rows": _scalar(self.h5["/MAPS/Scan/requested_rows"]),
            "requested_cols": _scalar(self.h5["/MAPS/Scan/requested_cols"]),
        }

    def extra_pvs(self) -> tuple[dict[str, Any], ...]:
        group = self.h5["/MAPS/Scan/Extra_PVs"]
        names = _slots(group["Names"])
        descriptions = _slots(group["Description"])
        units = _slots(group["Unit"])
        raw = group["Values"][()]
        values_raw = [raw] if np.ndim(raw) == 0 else np.asarray(raw).reshape(-1).tolist()
        values = tuple(_decode(v) for v in values_raw)
        width = max(len(names), len(descriptions), len(units), len(values))
        return tuple(
            {
                "index": i,
                "name": names[i] if i < len(names) else "",
                "description": descriptions[i] if i < len(descriptions) else "",
                "unit": units[i] if i < len(units) else "",
                "value": values[i] if i < len(values) else None,
            }
            for i in range(width)
        )

    def qc_channels(self, method: str = "Fitted") -> tuple[ChannelInfo, ...]:
        return tuple(
            item
            for item in self.channels(method)
            if item.role in {ChannelRole.FIT_DIAGNOSTIC, ChannelRole.SCATTER}
        )

    def validate(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        y_size, x_size = self.shape

        if self.x.size != x_size:
            issues.append(
                ValidationIssue("error", "x_axis_length", f"x axis {self.x.size} != raster x {x_size}")
            )
        if self.y.size != y_size:
            issues.append(
                ValidationIssue("error", "y_axis_length", f"y axis {self.y.size} != raster y {y_size}")
            )

        reference_names = self.channel_names("Fitted")
        for method in _METHODS:
            ds = self.h5[f"/MAPS/XRF_Analyzed/{method}/Counts_Per_Sec"]
            if tuple(ds.shape[1:]) != self.shape:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{method.lower()}_raster_shape",
                        f"{method} raster {tuple(ds.shape[1:])} != canonical {self.shape}",
                    )
                )
            if int(ds.shape[0]) != len(self.channels(method)):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{method.lower()}_channel_count",
                        f"{method} cube has {ds.shape[0]} channels, metadata {len(self.channels(method))}",
                    )
                )
            if self.channel_names(method) != reference_names:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"{method.lower()}_channel_names",
                        f"{method} channel names differ from Fitted",
                    )
                )

        e, sy, sx = self.spectral_shape
        if e != self.energy.size:
            issues.append(
                ValidationIssue(
                    "error",
                    "energy_length",
                    f"MCA energy axis {e} != Energy dataset {self.energy.size}",
                )
            )
        if (sy, sx) != self.shape:
            issues.append(
                ValidationIssue(
                    "error",
                    "spectral_raster_shape",
                    f"MCA raster {(sy, sx)} != canonical {self.shape}",
                )
            )

        for family in ("primary", "legacy"):
            names, units = self.scaler_metadata(family)  # type: ignore[arg-type]
            _, _, values_path = self._scaler_paths(family)  # type: ignore[arg-type]
            ds = self.h5[values_path]
            if int(ds.shape[0]) != len(names):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{family}_scaler_count",
                        f"{family} scaler cube has {ds.shape[0]}, metadata {len(names)}",
                    )
                )
            if len(names) != len(units):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{family}_scaler_units",
                        f"{family} has {len(names)} names vs {len(units)} units",
                    )
                )
            if tuple(ds.shape[1:]) != self.shape:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{family}_scaler_shape",
                        f"{family} scaler raster {tuple(ds.shape[1:])} != {self.shape}",
                    )
                )
        return tuple(issues)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.lightweight_identity(),
            "scan": self.acquisition_summary(),
            "spatial_dims": list(self.spatial_dims),
            "geometry": asdict(self.geometry),
            "methods": {
                method: {
                    "channel_count": len(self.channels(method)),
                    "channels": [
                        {
                            "index": item.index,
                            "name": item.name,
                            "unit": item.unit,
                            "role": item.role.value,
                        }
                        for item in self.channels(method)
                    ],
                }
                for method in _METHODS
            },
            "spectra": {
                "shape": list(self.spectral_shape),
                "energy_count": int(self.energy.size),
                "energy_min": float(np.nanmin(self.energy)),
                "energy_max": float(np.nanmax(self.energy)),
                "energy_calibration": self.energy_calibration.tolist(),
            },
            "scalers": {
                "primary_count": len(self.scaler_names("primary")),
                "legacy_count": len(self.scaler_names("legacy")),
            },
            "validation": [asdict(issue) for issue in self.validate()],
        }
