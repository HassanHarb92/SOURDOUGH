#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yeast_xrf.io.maps_calibration_semantics import (
    VALID_METHODS,
    VALID_REFERENCES,
    compact_signature,
    inspect_calibration_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "img.dat"
OUT = ROOT / "analysis" / "quantification" / "change10_semantics_probe.json"
TEXT = ROOT / "analysis" / "quantification" / "change10_semantics_probe.txt"


def stable_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def short(value, limit=180):
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main():
    files = sorted(DATA.glob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 files found in {DATA}")

    records = []
    failures = []
    signature_groups = {}

    print("SOURDOUGH Change 10 · MAPS calibration semantics probe")
    print("=" * 112)
    print(
        "Goal: decode the stored MAPS calibration contract before automatic "
        "concentration conversion is enabled."
    )
    print()

    for path in files:
        scan_ok = True
        for method in VALID_METHODS:
            for reference in VALID_REFERENCES:
                try:
                    report = inspect_calibration_semantics(
                        path,
                        method=method,
                        reference=reference,
                    )
                    sig = compact_signature(report)
                    sig_hash = stable_hash(sig)
                    signature_groups.setdefault(
                        f"{method}:{reference}:{sig_hash}",
                        [],
                    ).append(path.name)
                    records.append(
                        {
                            "scan": path.name,
                            "signature_hash": sig_hash,
                            "report": report,
                        }
                    )
                except Exception as exc:
                    scan_ok = False
                    failures.append(
                        {
                            "scan": path.name,
                            "method": method,
                            "reference": reference,
                            "error": repr(exc),
                        }
                    )

        print(
            f"{'PASS' if scan_ok else 'WARN'}  "
            f"{path.name:<23} "
            f"{len(VALID_METHODS) * len(VALID_REFERENCES)} "
            "method/reference combinations inspected"
        )

    # Representative detailed view from the first file.
    representative = {}
    first = files[0]
    for method in VALID_METHODS:
        representative[method] = {}
        for reference in VALID_REFERENCES:
            representative[method][reference] = inspect_calibration_semantics(
                first,
                method=method,
                reference=reference,
            )

    payload = {
        "scan_count": len(files),
        "method_count": len(VALID_METHODS),
        "reference_count": len(VALID_REFERENCES),
        "record_count": len(records),
        "failure_count": len(failures),
        "scientific_gate": {
            "automatic_concentration_enabled": False,
            "reason": (
                "This probe is intentionally forensic. Automatic concentration "
                "conversion must remain locked until the row/column semantics and "
                "conversion formula are verified from the stored labels/values and "
                "checked against trusted MAPS behavior."
            ),
        },
        "signature_groups": signature_groups,
        "representative_scan": first.name,
        "representative": representative,
        "records": records,
        "failures": failures,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    )

    lines = []
    lines.append("SOURDOUGH Change 10 · MAPS calibration semantics probe")
    lines.append("=" * 112)
    lines.append(f"Representative scan: {first.name}")
    lines.append(f"Scans inspected: {len(files)}")
    lines.append(f"Records: {len(records)}")
    lines.append(f"Failures: {len(failures)}")
    lines.append("")

    for method in VALID_METHODS:
        lines.append(f"[{method}]")
        for reference in VALID_REFERENCES:
            r = representative[method][reference]
            c = r["calibration"]
            q = r["quantification"]
            lines.append(f"  {reference}")
            lines.append(
                "    curve labels: "
                + short(
                    c["curve_labels"]["values"]
                    if c["curve_labels"]
                    else None
                )
            )
            lines.append(
                "    element info index: "
                + short(
                    c["element_info_index"]["values"]
                    if c["element_info_index"]
                    else None
                )
            )
            lines.append(
                "    element info names: "
                + short(
                    c["element_info_names"]["values"]
                    if c["element_info_names"]
                    else None
                )
            )
            lines.append(
                "    quant names: "
                + short(q["names"]["values"] if q["names"] else None)
            )
            lines.append(
                "    curve row fingerprints: "
                + short(r["numeric_fingerprints"]["calibration_curve"]["rows"])
            )
            lines.append(
                "    element-info row fingerprints: "
                + short(r["numeric_fingerprints"]["element_info_values"]["rows"])
            )
            lines.append(
                "    quant row fingerprints: "
                + short(r["numeric_fingerprints"]["quantification_values"]["rows"])
            )
        lines.append("")

    lines.append("Cross-scan signature groups")
    lines.append("-" * 112)
    for key in sorted(signature_groups):
        scans = signature_groups[key]
        lines.append(f"{key} -> {len(scans)} scan(s): {', '.join(scans)}")

    if failures:
        lines.append("")
        lines.append("Failures")
        lines.append("-" * 112)
        for row in failures:
            lines.append(short(row, limit=500))

    TEXT.write_text("\n".join(lines) + "\n")

    print("-" * 112)
    print(f"Records written: {len(records)}")
    print(f"Failures:        {len(failures)}")
    print(f"JSON: {OUT.relative_to(ROOT)}")
    print(f"Text: {TEXT.relative_to(ROOT)}")
    print()
    print("Representative semantics:")
    for method in VALID_METHODS:
        r = representative[method]["US_IC"]
        c = r["calibration"]
        q = r["quantification"]
        print(f"  {method}")
        print(
            "    curve labels     = "
            + short(
                c["curve_labels"]["values"]
                if c["curve_labels"]
                else None
            )
        )
        print(
            "    element names    = "
            + short(
                c["element_info_names"]["values"]
                if c["element_info_names"]
                else None
            )
        )
        print(
            "    quant names      = "
            + short(q["names"]["values"] if q["names"] else None)
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
