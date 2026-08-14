"""
atlas clean command — Dataset Cleaning Engine entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from atlas_cli.agents.dataset_intelligence.cleaner import clean_dataset
from atlas_cli.core.config import settings

console = Console()


def clean(
    file_path: Path = typer.Argument(..., help="Path to raw dataset file (CSV, Parquet, JSON)"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column to protect from cleaning"),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run ID for artifact saving"),
    ignore: Optional[str] = typer.Option(None, "--ignore", "-i", help="Comma-separated column names to ignore/drop"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Explicit output file path or directory for exported cleaned dataset"),
    no_vif_drop: bool = typer.Option(False, "--no-vif-drop", help="Disable automatic high-VIF column dropping"),
    no_outlier_cap: bool = typer.Option(False, "--no-outlier-cap", help="Disable IQR outlier Winsorization/capping"),
) -> None:
    """
    Clean dataset: apply IQR Winsorization, remove high-VIF collinear features,
    and filter extreme Isolation Forest anomaly noise rows.

    Example:
        atlas clean Iris.csv --target Species --ignore Id -o cleaned_iris.csv
    """
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    effective_run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = settings.workspace_dir / "runs" / effective_run_id

    ignore_cols = [c.strip() for c in ignore.split(",") if c.strip()] if ignore else []

    with console.status(f"[cyan]Cleaning dataset: {file_path.name}...[/cyan]"):
        try:
            df_clean, report = clean_dataset(
                file_path=file_path,
                target_col=target,
                output_dir=output or run_dir,
                ignore_cols=ignore_cols,
                cap_outliers=not no_outlier_cap,
                drop_high_vif=not no_vif_drop,
                filter_anomalies=True,
            )
            # Ensure copy is stored in run_dir as well if custom output path was given
            if output and report.cleaned_file_path:
                run_dir.mkdir(parents=True, exist_ok=True)
                run_cleaned_path = run_dir / "cleaned_data.csv"
                df_clean.to_csv(run_cleaned_path, index=False)
        except Exception as e:
            console.print(f"[bold red]Failed to clean dataset:[/bold red] {e}")
            raise typer.Exit(code=1)

    console.rule("[bold cyan]Dataset Cleaning Report[/bold cyan]")
    
    console.print(Panel(
        f"[bold]Original Shape:[/bold] {report.original_rows:,} rows × {report.original_cols} columns\n"
        f"[bold]Cleaned Shape:[/bold]  {report.cleaned_rows:,} rows × {report.cleaned_cols} columns\n"
        f"[bold]Noise Rows Removed:[/bold] {report.rows_removed}\n"
        f"[bold]Columns Dropped:[/bold]   {len(report.cols_dropped)}\n"
        f"[bold]Columns Capped:[/bold]    {len(report.capped_columns)}",
        title="[bold blue]🧹 Cleaning Overview[/bold blue]",
        border_style="blue",
    ))

    if report.cols_dropped:
        table = Table(title="🗑️ Dropped Features", border_style="yellow", header_style="bold yellow")
        table.add_column("Dropped Column & Reason", style="white")
        for col_reason in report.cols_dropped:
            table.add_row(col_reason)
        console.print(table)

    if report.capped_columns:
        table = Table(title="✂️ Outlier Capped Columns (IQR Bounds)", border_style="magenta", header_style="bold magenta")
        table.add_column("Capped Column & Outlier %", style="white")
        for cap_info in report.capped_columns:
            table.add_row(cap_info)
        console.print(table)

    if report.cleaned_file_path:
        console.print(Panel(
            f"[cyan]→[/cyan] [dim]{report.cleaned_file_path.resolve()}[/dim]",
            title="[bold green]✅ Cleaned File Exported[/bold green]",
            border_style="green",
        ))

    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
