"""Cross-file HDF5 inventory comparison and conservative 3D-readiness analysis."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_COORD_TOKENS = {
    "x": ("x_pos", "xpos", "x_axis", "xaxis", "x_coord", "xcoord", "motor_x"),
    "y": ("y_pos", "ypos", "y_axis", "yaxis", "y_coord", "ycoord", "motor_y"),
    "z": ("z_pos", "zpos", "z_axis", "zaxis", "z_coord", "zcoord", "depth", "slice"),
}

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("coordinate", ("position", "coord", "axis", "motor", "x_pos", "y_pos", "z_pos")),
    ("scaler", ("scaler", "i0", "i1", "ion_chamber", "dwell", "live_time", "real_time")),
    ("energy_calibration", ("energy", "calib", "ev_per", "kev", "incident_energy")),
    ("spectrum_or_mca", ("spectrum", "spectra", "mca", "channel", "detector")),
    ("fitted_or_element_map", ("fit", "xrfmap", "xrf_map", "element", "roi", "map")),
    ("tomography_hint", ("theta", "angle", "projection", "tomo", "rotation")),
)


def _norm(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _norm(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_norm(v) for v in value)
    return value


def _text_blob(dataset: dict[str, Any]) -> str:
    return json.dumps(
        {"path": dataset.get("path", ""), "attrs": dataset.get("attrs", {})},
        sort_keys=True,
        default=str,
    ).lower()


def candidate_categories(dataset: dict[str, Any]) -> list[str]:
    """Return weak semantic hints only; these are not scientific classifications."""
    text = _text_blob(dataset)
    found: list[str] = []
    for category, tokens in _CATEGORY_RULES:
        if any(token in text for token in tokens):
            found.append(category)
    return found or ["unclassified"]


def axis_hints(dataset: dict[str, Any]) -> list[str]:
    text = _text_blob(dataset)
    hints: list[str] = []
    for axis, tokens in _COORD_TOKENS.items():
        if any(token in text for token in tokens):
            hints.append(axis)
    return hints


def load_inventories(directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    files = sorted(directory.glob("*.inventory.json"))
    if not files:
        raise FileNotFoundError(f"no *.inventory.json files under {directory}")
    inventories: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text())
        payload["_inventory_path"] = str(path)
        payload["_scan_name"] = Path(payload.get("source", path.name)).name
        inventories.append(payload)
    return inventories


def compare_inventories(inventories: Iterable[dict[str, Any]]) -> dict[str, Any]:
    invs = list(inventories)
    scan_names = [inv["_scan_name"] for inv in invs]
    all_paths = sorted({ds["path"] for inv in invs for ds in inv.get("datasets", [])})
    by_scan = {
        inv["_scan_name"]: {ds["path"]: ds for ds in inv.get("datasets", [])}
        for inv in invs
    }

    per_path: dict[str, dict[str, Any]] = {}
    for path in all_paths:
        present: list[str] = []
        shapes: Counter[Any] = Counter()
        dtypes: Counter[str] = Counter()
        ndims: Counter[int] = Counter()
        categories: Counter[str] = Counter()
        axes: Counter[str] = Counter()
        attr_keys: Counter[str] = Counter()

        for scan in scan_names:
            ds = by_scan[scan].get(path)
            if ds is None:
                continue
            present.append(scan)
            shapes[_norm(ds.get("shape", []))] += 1
            dtypes[str(ds.get("dtype", ""))] += 1
            ndims[int(ds.get("ndim", len(ds.get("shape", []))))] += 1
            for cat in candidate_categories(ds):
                categories[cat] += 1
            for axis in axis_hints(ds):
                axes[axis] += 1
            for key in ds.get("attrs", {}):
                attr_keys[str(key)] += 1

        per_path[path] = {
            "present_count": len(present),
            "present_fraction": len(present) / len(scan_names) if scan_names else 0.0,
            "present_scans": present,
            "missing_scans": [s for s in scan_names if s not in present],
            "shape_signatures": [
                {"shape": list(shape), "count": n} for shape, n in shapes.most_common()
            ],
            "dtype_signatures": dict(dtypes),
            "ndim_signatures": {str(k): v for k, v in sorted(ndims.items())},
            "candidate_categories": dict(categories),
            "axis_hints": dict(axes),
            "attribute_keys": dict(attr_keys),
        }

    root_attr_presence: dict[str, int] = defaultdict(int)
    root_attr_values: dict[str, Counter[Any]] = defaultdict(Counter)
    for inv in invs:
        for key, value in inv.get("root_attrs", {}).items():
            root_attr_presence[str(key)] += 1
            root_attr_values[str(key)][_norm(value)] += 1

    root_attrs = {
        key: {
            "present_count": root_attr_presence[key],
            "value_variants": len(root_attr_values[key]),
            "constant_across_present": len(root_attr_values[key]) == 1,
        }
        for key in sorted(root_attr_presence)
    }

    common_paths = [p for p, info in per_path.items() if info["present_count"] == len(scan_names)]
    variable_paths = [p for p, info in per_path.items() if 0 < info["present_count"] < len(scan_names)]

    return {
        "scan_count": len(scan_names),
        "scans": scan_names,
        "dataset_path_count": len(all_paths),
        "common_dataset_paths": common_paths,
        "variable_dataset_paths": variable_paths,
        "root_attributes": root_attrs,
        "datasets": per_path,
    }


def analyze_3d_readiness(
    inventories: Iterable[dict[str, Any]], comparison: dict[str, Any]
) -> dict[str, Any]:
    invs = list(inventories)
    candidate_native: list[dict[str, Any]] = []
    coordinate_hints: dict[str, list[dict[str, str]]] = {"x": [], "y": [], "z": []}
    tomography_hints: list[dict[str, str]] = []

    for inv in invs:
        scan = inv["_scan_name"]
        for ds in inv.get("datasets", []):
            ndim = int(ds.get("ndim", len(ds.get("shape", []))))
            cats = candidate_categories(ds)
            axes = axis_hints(ds)
            text = _text_blob(ds)

            if ndim >= 3:
                candidate_native.append(
                    {
                        "scan": scan,
                        "path": ds["path"],
                        "shape": ds.get("shape", []),
                        "ndim": ndim,
                        "candidate_categories": cats,
                        "axis_hints": axes,
                        "warning": (
                            "Rank >=3 does not prove a z-y-x volume; this may be y-x-energy, "
                            "channels-y-x, detector data, or another cube."
                        ),
                    }
                )

            for axis in axes:
                coordinate_hints[axis].append({"scan": scan, "path": ds["path"]})

            if "tomography_hint" in cats or any(
                token in text for token in ("theta", "projection", "rotation_angle")
            ):
                tomography_hints.append({"scan": scan, "path": ds["path"]})

    two_d_shapes: Counter[Any] = Counter()
    two_d_examples: dict[Any, list[dict[str, str]]] = defaultdict(list)
    for inv in invs:
        scan = inv["_scan_name"]
        for ds in inv.get("datasets", []):
            shape = tuple(ds.get("shape", []))
            ndim = int(ds.get("ndim", len(shape)))
            if ndim == 2 and len(shape) == 2:
                two_d_shapes[shape] += 1
                if len(two_d_examples[shape]) < 12:
                    two_d_examples[shape].append({"scan": scan, "path": ds["path"]})

    repeated_2d = [
        {
            "shape": list(shape),
            "dataset_occurrences": n,
            "examples": two_d_examples[shape],
        }
        for shape, n in two_d_shapes.most_common()
        if n >= 2
    ]

    z_evidence = coordinate_hints["z"]
    native_has_z_metadata = any("z" in item["axis_hints"] for item in candidate_native)

    routes = {
        "native_volume": {
            "status": "plausible" if native_has_z_metadata else ("inspect" if candidate_native else "not_evident"),
            "reason": (
                "At least one rank>=3 dataset also carries a z/depth/slice hint."
                if native_has_z_metadata
                else (
                    "Rank>=3 datasets exist, but none can yet be proven to be z-y-x."
                    if candidate_native
                    else "No rank>=3 dataset was found in the inventories."
                )
            ),
        },
        "serial_section_stack": {
            "status": "plausible" if z_evidence and repeated_2d else ("inspect" if repeated_2d else "not_evident"),
            "reason": (
                "Z/depth hints and repeatable 2D geometries exist."
                if z_evidence and repeated_2d
                else (
                    "Repeatable 2D geometries exist, but physical z ordering/spacing is not established."
                    if repeated_2d
                    else "No repeatable 2D geometry was identified."
                )
            ),
        },
        "tomography": {
            "status": "plausible" if tomography_hints else "not_evident",
            "reason": (
                "Projection/angle/tomography-like metadata or dataset names were found."
                if tomography_hints
                else "No projection-angle/tomography hints were found in inventory metadata."
            ),
        },
        "pseudo_3d_2p5d": {
            "status": "available",
            "reason": (
                "Any 2D scalar map can be rendered as an intensity surface, but this is a "
                "visualization and must not be described as a reconstructed cell volume."
            ),
        },
    }

    return {
        "important_warning": (
            "Do not infer a biological 3D cell volume from a third array dimension alone. "
            "A real reconstruction requires physical z/depth information, registered serial "
            "sections, or tomographic projection geometry."
        ),
        "candidate_rank3plus_datasets": candidate_native,
        "coordinate_hints": coordinate_hints,
        "tomography_hints": tomography_hints,
        "repeated_2d_geometries": repeated_2d,
        "routes": routes,
        "comparison_scan_count": comparison["scan_count"],
    }


def write_reports(
    inventories: list[dict[str, Any]],
    comparison: dict[str, Any],
    readiness: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_json = out / "schema_summary.json"
    readiness_json = out / "three_d_readiness.json"
    matrix_csv = out / "dataset_matrix.csv"
    report_md = out / "schema_report.md"

    summary_json.write_text(json.dumps(comparison, indent=2, sort_keys=True, default=str))
    readiness_json.write_text(json.dumps(readiness, indent=2, sort_keys=True, default=str))

    scans = comparison["scans"]
    with matrix_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_path",
                "present_count",
                "present_fraction",
                "shape_signatures",
                "dtype_signatures",
                "ndim_signatures",
                "candidate_categories",
                *scans,
            ]
        )
        for path, info in comparison["datasets"].items():
            present = set(info["present_scans"])
            writer.writerow(
                [
                    path,
                    info["present_count"],
                    f'{info["present_fraction"]:.6f}',
                    json.dumps(info["shape_signatures"], sort_keys=True),
                    json.dumps(info["dtype_signatures"], sort_keys=True),
                    json.dumps(info["ndim_signatures"], sort_keys=True),
                    json.dumps(info["candidate_categories"], sort_keys=True),
                    *["yes" if scan in present else "" for scan in scans],
                ]
            )

    lines: list[str] = [
        "# Yeast XRF HDF5 schema comparison",
        "",
        f"- Scans compared: **{comparison['scan_count']}**",
        f"- Unique dataset paths: **{comparison['dataset_path_count']}**",
        f"- Dataset paths present in every scan: **{len(comparison['common_dataset_paths'])}**",
        f"- Dataset paths present in only some scans: **{len(comparison['variable_dataset_paths'])}**",
        "",
        "## 3D readiness",
        "",
        "> " + readiness["important_warning"],
        "",
    ]
    for route, info in readiness["routes"].items():
        lines.append(f"- **{route}** — `{info['status']}`: {info['reason']}")

    lines.extend(
        [
            "",
            "## Dataset paths",
            "",
            "| Path | Present | Shapes | ndim | Candidate hints |",
            "|---|---:|---|---|---|",
        ]
    )
    for path, info in comparison["datasets"].items():
        shape_text = "; ".join(
            f"{tuple(item['shape'])} ×{item['count']}" for item in info["shape_signatures"]
        )
        ndim_text = ", ".join(f"{k}D ×{v}" for k, v in info["ndim_signatures"].items())
        cat_text = ", ".join(info["candidate_categories"]) or "unclassified"
        lines.append(
            f"| `{path}` | {info['present_count']}/{comparison['scan_count']} | "
            f"{shape_text} | {ndim_text} | {cat_text} |"
        )

    lines.extend(["", "## Candidate rank >= 3 datasets", ""])
    if readiness["candidate_rank3plus_datasets"]:
        for item in readiness["candidate_rank3plus_datasets"]:
            lines.append(
                f"- `{item['scan']}` `{item['path']}` shape={tuple(item['shape'])}; "
                f"hints={item['candidate_categories']}; axes={item['axis_hints']}"
            )
    else:
        lines.append("- None found.")

    lines.extend(
        [
            "",
            "## Next interpretation step",
            "",
            "Dataset-name heuristics in this report are intentionally weak hints. The next step is "
            "to inspect the actual candidate datasets, labels, coordinate arrays, and metadata "
            "before assigning scientific roles.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n")

    return {
        "summary_json": summary_json,
        "readiness_json": readiness_json,
        "matrix_csv": matrix_csv,
        "report_md": report_md,
    }
