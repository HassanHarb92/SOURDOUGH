'''TFY-driven cell detection and 2D morphometry.

The segmentation source is Total_Fluorescence_Yield when used by the Explorer. Components
touching any image boundary are flagged as cropped and are excluded from complete-cell
analysis by default.

Coordinate-derived sizes are reported in stored-coordinate units because the physical X/Y unit
has not yet been verified.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max


ThresholdMethod = Literal["otsu", "yen", "li", "percentile"]


@dataclass(frozen=True)
class CellRecord:
    cell_id: int
    cropped: bool
    touches_top: bool
    touches_bottom: bool
    touches_left: bool
    touches_right: bool
    area_pixels: int
    coordinate_area_approx: float
    centroid_x: float
    centroid_y: float
    equivalent_diameter_coordinate_approx: float
    major_axis_coordinate_approx: float
    minor_axis_coordinate_approx: float
    orientation_deg_coordinate: float
    eccentricity_coordinate: float
    perimeter_pixels: float
    circularity_pixel: float
    solidity: float
    mean_tfy: float
    median_tfy: float
    max_tfy: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CellSegmentationResult:
    labels: np.ndarray
    foreground_mask: np.ndarray
    normalized_tfy: np.ndarray
    records: tuple[CellRecord, ...]
    threshold_value: float
    metadata: dict

    @property
    def total_cells(self):
        return len(self.records)

    @property
    def cropped_cells(self):
        return sum(record.cropped for record in self.records)

    @property
    def complete_cells(self):
        return self.total_cells - self.cropped_cells


def _robust_normalize(data):
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, np.nan)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros(arr.shape, dtype=float)
        out[~np.isfinite(arr)] = np.nan
        return out
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out


def _nonzero_median_step(axis):
    axis = np.asarray(axis, dtype=float)
    d = np.abs(np.diff(axis))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 1.0


def _coordinate_geometry(rows, cols, x, y):
    xs = np.asarray(x, dtype=float)[cols]
    ys = np.asarray(y, dtype=float)[rows]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    points = np.column_stack([xs - cx, ys - cy])
    if points.shape[0] >= 2:
        cov = np.cov(points, rowvar=False, bias=True)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        major = 4.0 * np.sqrt(eigvals[0])
        minor = 4.0 * np.sqrt(eigvals[1])
        vec = eigvecs[:, 0]
        angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
        eccentricity = (
            float(np.sqrt(max(0.0, 1.0 - eigvals[1] / eigvals[0])))
            if eigvals[0] > 0 else 0.0
        )
    else:
        major = minor = angle = eccentricity = 0.0

    dx = _nonzero_median_step(x)
    dy = _nonzero_median_step(y)
    area = float(rows.size * dx * dy)
    equivalent = float(2.0 * np.sqrt(area / np.pi)) if area > 0 else 0.0

    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "major_axis": float(major),
        "minor_axis": float(minor),
        "orientation_deg": angle,
        "eccentricity": eccentricity,
        "area": area,
        "equivalent_diameter": equivalent,
        "median_dx": dx,
        "median_dy": dy,
    }


def segment_cells_tfy(
    tfy,
    x,
    y,
    *,
    threshold_method: ThresholdMethod = "otsu",
    threshold_percentile: float = 75.0,
    smooth_sigma_pixels: float = 1.0,
    min_area_pixels: int = 25,
    closing_radius_pixels: int = 1,
    fill_holes: bool = True,
    split_touching: bool = True,
    watershed_min_distance_pixels: int = 5,
):
    arr = np.asarray(tfy, dtype=float)
    if arr.ndim != 2:
        raise ValueError("TFY source must be a 2D map")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != arr.shape[1] or y.size != arr.shape[0]:
        raise ValueError("coordinate lengths do not match TFY map")

    normalized = _robust_normalize(arr)
    finite = np.isfinite(normalized)
    work = np.where(finite, normalized, 0.0)

    sigma = float(smooth_sigma_pixels)
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError("smooth_sigma_pixels must be finite and >= 0")
    smoothed = ndi.gaussian_filter(work, sigma=sigma, mode="nearest") if sigma > 0 else work

    values = smoothed[finite]
    if values.size == 0:
        raise ValueError("TFY map has no finite pixels")

    method = str(threshold_method).lower()
    if method == "otsu":
        threshold = float(filters.threshold_otsu(values))
    elif method == "yen":
        threshold = float(filters.threshold_yen(values))
    elif method == "li":
        threshold = float(filters.threshold_li(values))
    elif method == "percentile":
        p = float(threshold_percentile)
        if not 0 < p < 100:
            raise ValueError("threshold_percentile must be between 0 and 100")
        threshold = float(np.percentile(values, p))
    else:
        raise ValueError("threshold_method must be otsu, yen, li, or percentile")

    mask = finite & (smoothed >= threshold)
    mask = morphology.remove_small_objects(mask, min_size=max(1, int(min_area_pixels)))

    radius = int(closing_radius_pixels)
    if radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(radius))

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    mask = morphology.remove_small_objects(mask, min_size=max(1, int(min_area_pixels)))

    if split_touching and np.any(mask):
        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(
            distance,
            labels=mask,
            min_distance=max(1, int(watershed_min_distance_pixels)),
            exclude_border=False,
        )
        markers = np.zeros(mask.shape, dtype=np.int32)
        for marker_id, (row, col) in enumerate(coords, start=1):
            markers[row, col] = marker_id

        if marker_id if len(coords) else 0:
            markers = measure.label(markers > 0)
            labels = segmentation.watershed(-distance, markers, mask=mask)
        else:
            labels = measure.label(mask)
    else:
        labels = measure.label(mask)

    records = []
    nrows, ncols = arr.shape

    for region in measure.regionprops(labels, intensity_image=arr):
        cell_id = int(region.label)
        rows = region.coords[:, 0]
        cols = region.coords[:, 1]
        touches_top = bool(np.any(rows == 0))
        touches_bottom = bool(np.any(rows == nrows - 1))
        touches_left = bool(np.any(cols == 0))
        touches_right = bool(np.any(cols == ncols - 1))
        cropped = touches_top or touches_bottom or touches_left or touches_right

        geom = _coordinate_geometry(rows, cols, x, y)
        area_pixels = int(region.area)
        perimeter = float(region.perimeter)
        circularity = (
            float(4.0 * np.pi * area_pixels / (perimeter * perimeter))
            if perimeter > 0 else 0.0
        )

        values_cell = arr[rows, cols]
        finite_cell = values_cell[np.isfinite(values_cell)]

        minr, minc, maxr, maxc = region.bbox
        records.append(
            CellRecord(
                cell_id=cell_id,
                cropped=cropped,
                touches_top=touches_top,
                touches_bottom=touches_bottom,
                touches_left=touches_left,
                touches_right=touches_right,
                area_pixels=area_pixels,
                coordinate_area_approx=geom["area"],
                centroid_x=geom["centroid_x"],
                centroid_y=geom["centroid_y"],
                equivalent_diameter_coordinate_approx=geom["equivalent_diameter"],
                major_axis_coordinate_approx=geom["major_axis"],
                minor_axis_coordinate_approx=geom["minor_axis"],
                orientation_deg_coordinate=geom["orientation_deg"],
                eccentricity_coordinate=geom["eccentricity"],
                perimeter_pixels=perimeter,
                circularity_pixel=circularity,
                solidity=float(region.solidity),
                mean_tfy=float(np.mean(finite_cell)) if finite_cell.size else np.nan,
                median_tfy=float(np.median(finite_cell)) if finite_cell.size else np.nan,
                max_tfy=float(np.max(finite_cell)) if finite_cell.size else np.nan,
                bbox_y0=int(minr),
                bbox_x0=int(minc),
                bbox_y1=int(maxr),
                bbox_x1=int(maxc),
            )
        )

    metadata = {
        "source": "Total_Fluorescence_Yield",
        "threshold_method": method,
        "threshold_value_normalized": threshold,
        "threshold_percentile": float(threshold_percentile),
        "smooth_sigma_pixels": sigma,
        "min_area_pixels": int(min_area_pixels),
        "closing_radius_pixels": radius,
        "fill_holes": bool(fill_holes),
        "split_touching": bool(split_touching),
        "watershed_min_distance_pixels": int(watershed_min_distance_pixels),
        "cropped_rule": "cell touches any image border",
        "complete_cell_rule": "cell touches no image border",
        "coordinate_size_unit": "stored-coordinate-unit",
        "coordinate_area_method": (
            "pixel count multiplied by median nonzero dx and dy; "
            "major/minor axes from covariance of stored pixel-center coordinates"
        ),
        "physical_unit_verified": False,
    }

    return CellSegmentationResult(
        labels=np.asarray(labels, dtype=np.int32),
        foreground_mask=np.asarray(mask, dtype=bool),
        normalized_tfy=normalized,
        records=tuple(records),
        threshold_value=threshold,
        metadata=metadata,
    )
