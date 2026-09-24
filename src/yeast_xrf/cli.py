"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from yeast_xrf.io.discovery import discover_h5
from yeast_xrf.io.h5_inventory import inventory_h5, inventory_to_dict, write_inventory

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def list_files(
    data_dir: Path = typer.Option(Path("img.dat"), exists=True, file_okay=False),
) -> None:
    """List HDF5 files without opening array payloads."""
    files = discover_h5(data_dir)
    table = Table("File", "Size (MB)")
    for path in files:
        table.add_row(path.name, f"{path.stat().st_size / 1e6:.2f}")
    print(table)
    print(f"[bold]{len(files)} HDF5 file(s)[/bold]")


@app.command()
def inventory(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, help="JSON output path"),
) -> None:
    """Inspect one HDF5 file without assuming its internal layout."""
    inv = inventory_h5(path)
    payload = inventory_to_dict(inv)
    if output:
        write_inventory(inv, output)
        print(f"[green]Wrote[/green] {output}")
    else:
        print(json.dumps(payload, indent=2, default=str))


@app.command("inventory-all")
def inventory_all(
    data_dir: Path = typer.Option(Path("img.dat"), exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("analysis/inventory")),
) -> None:
    """Write one inventory JSON per HDF5 input."""
    files = discover_h5(data_dir)
    if not files:
        raise typer.Exit("No HDF5 files found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        destination = output_dir / f"{path.name}.inventory.json"
        write_inventory(inventory_h5(path), destination)
        print(f"[green]✓[/green] {path.name} -> {destination}")
    print(f"[bold green]{len(files)} inventories written[/bold green]")


@app.command("schema-report")
def schema_report(
    inventory_dir: Path = typer.Option(Path("analysis/inventory"), exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("analysis/schema")),
) -> None:
    """Compare inventory structure across scans and assess conservative 3D readiness."""
    from yeast_xrf.io.schema_compare import (
        analyze_3d_readiness,
        compare_inventories,
        load_inventories,
        write_reports,
    )

    inventories = load_inventories(inventory_dir)
    comparison = compare_inventories(inventories)
    readiness = analyze_3d_readiness(inventories, comparison)
    written = write_reports(inventories, comparison, readiness, output_dir)
    print(f"[bold]Scans:[/bold] {comparison['scan_count']}")
    print(f"[bold]Unique dataset paths:[/bold] {comparison['dataset_path_count']}")
    for route, info in readiness["routes"].items():
        print(f"[cyan]{route}[/cyan]: {info['status']} — {info['reason']}")
    for out_path in written.values():
        print(f"[green]Wrote[/green] {out_path}")


@app.command("semantic-probe")
def semantic_probe(
    data_dir: Path = typer.Option(Path("img.dat"), exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("analysis/semantic")),
) -> None:
    """Read safe semantic metadata from all XRF scans and write comparison reports."""
    from yeast_xrf.io.semantic_probe import probe_directory, summarize_probes, write_semantic_reports

    probes = probe_directory(data_dir)
    summary = summarize_probes(probes)
    written = write_semantic_reports(probes, summary, output_dir)
    print(f"[bold]Scans:[/bold] {summary['scan_count']}")
    print(
        f"[bold]Theta:[/bold] {summary['theta']['unique_count']} unique value(s), "
        f"span={summary['theta']['span']}"
    )
    print(summary["tomography_assessment"])
    for output_path in written.values():
        print(f"[green]Wrote[/green] {output_path}")


@app.command("scan-summary")
def scan_summary(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    # Print canonical metadata and validation for one scan.
    from yeast_xrf.io.xrf_scan import XRFScan

    with XRFScan.open(path) as scan:
        print(json.dumps(scan.summary(), indent=2, default=str))


if __name__ == "__main__":
    app()
