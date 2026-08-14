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
    no_vif_drop: bool = typer.Option(True, "--no-vif-drop", help="Disable automatic high-VIF column dropping (default: True, VIF is diagnostic only)"),
    outlier_action: str = typer.Option("report", "--outlier-action", help="Outlier handling: 'report' (default, diagnostic only) or 'cap' (IQR Winsorization)"),
    anomaly_action: str = typer.Option("report", "--anomaly-action", "-a", help="Anomaly handling: 'report' (default, keep all rows), 'flag' (add column), 'remove' (delete rows)"),
    no_outlier_cap: bool = typer.Option(True, "--no-outlier-cap", help="Legacy flag: disable outlier capping (default: True)"),
) -> None:
    """
    Clean dataset: report or cap IQR outliers, report high-VIF collinear features,
    and configurable anomaly handling (report/flag/remove).

    Example:
        atlas clean Iris.csv --target Species --ignore Id -o cleaned_iris.csv
    """
    # Normalize potential Typer default wrapper objects if called programmatically
    if not isinstance(anomaly_action, str):
        anomaly_action = getattr(anomaly_action, "default", "report")
        if not isinstance(anomaly_action, str):
            anomaly_action = "report"

    if not isinstance(outlier_action, str):
        outlier_action = getattr(outlier_action, "default", "report")
        if not isinstance(outlier_action, str):
            outlier_action = "report"

    if not isinstance(no_vif_drop, bool):
        no_vif_drop = getattr(no_vif_drop, "default", True)
        if not isinstance(no_vif_drop, bool):
            no_vif_drop = True

    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    effective_run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = settings.workspace_dir / "runs" / effective_run_id

    ignore_cols = [c.strip() for c in ignore.split(",") if c.strip()] if ignore else []

    # Validate actions
    valid_anomaly_actions = {"report", "flag", "remove"}
    if anomaly_action not in valid_anomaly_actions:
        console.print(f"[bold red]Error:[/bold red] --anomaly-action must be one of: {valid_anomaly_actions}")
        raise typer.Exit(code=1)

    valid_outlier_actions = {"report", "cap"}
    if outlier_action not in valid_outlier_actions:
        console.print(f"[bold red]Error:[/bold red] --outlier-action must be one of: {valid_outlier_actions}")
        raise typer.Exit(code=1)

    with console.status(f"[cyan]Cleaning dataset: {file_path.name}...[/cyan]"):
        try:
            df_clean, report = clean_dataset(
                file_path=file_path,
                target_col=target,
                output_dir=output or run_dir,
                ignore_cols=ignore_cols,
                outlier_action=outlier_action,
                drop_high_vif=not no_vif_drop,
                anomaly_action=anomaly_action,
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

    outlier_status_str = "Report only (values preserved)" if report.outlier_action == "report" else f"IQR Capped ({len(report.capped_columns)} cols)"
    anomaly_status_str = "Report only (all rows preserved)" if report.anomaly_action == "report" else report.anomaly_action

    console.print(Panel(
        f"[bold]Original Shape:[/bold]    {report.original_rows:,} rows × {report.original_cols} columns\n"
        f"[bold]Cleaned Shape:[/bold]     {report.cleaned_rows:,} rows × {report.cleaned_cols} columns\n"
        f"[bold]Noise Rows Removed:[/bold]{report.rows_removed}\n"
        f"[bold]Columns Dropped:[/bold]   {len(report.cols_dropped)}\n"
        f"[bold]Outlier Strategy:[/bold]  {outlier_status_str}\n"
        f"[bold]Anomaly Strategy:[/bold]  {anomaly_status_str}",
        title="[bold blue]🧹 Cleaning Overview[/bold blue]",
        border_style="blue",
    ))

    if report.cols_dropped:
        table = Table(title="🗑️ Dropped Features", border_style="yellow", header_style="bold yellow")
        table.add_column("Dropped Column & Reason", style="white")
        for col_reason in report.cols_dropped:
            table.add_row(col_reason)
        console.print(table)

    if report.vif_diagnostics and no_vif_drop:
        table = Table(title="🔍 Multicollinearity Diagnostic (VIF >= 10, Kept)", border_style="cyan", header_style="bold cyan")
        table.add_column("Column (VIF)", style="white")
        for vif_info in report.vif_diagnostics:
            table.add_row(vif_info)
        console.print(table)

    if report.capped_columns:
        table = Table(title="✂️ Outlier Capped Columns (IQR Bounds)", border_style="magenta", header_style="bold magenta")
        table.add_column("Capped Column & Outlier %", style="white")
        for cap_info in report.capped_columns:
            table.add_row(cap_info)
        console.print(table)
    elif report.outlier_diagnostics:
        table = Table(title="📊 Univariate IQR Outlier Diagnostic (Unmodified)", border_style="cyan", header_style="bold cyan")
        table.add_column("Column Diagnostic", style="white")
        for diag in report.outlier_diagnostics:
            table.add_row(diag)
        console.print(table)

    if report.anomaly_report:
        ar = report.anomaly_report
        console.print(Panel(
            f"[bold]Isolation Forest Outliers:[/bold] {ar.isolation_forest_count} ({ar.isolation_forest_pct:.1f}%)\n"
            f"[bold]Local Outlier Factor (LOF):[/bold] {ar.lof_count} ({ar.lof_pct:.1f}%)\n"
            f"[bold]Consensus Outliers (Both):[/bold]  {ar.consensus_count} ({ar.consensus_pct:.1f}%)\n"
            f"[bold]Action Executed:[/bold]            {ar.action_taken} ({'no rows removed' if ar.action_taken == 'report' else f'{len(ar.removed_indices)} rows removed'})",
            title="[bold cyan]🔬 Anomaly Detection Diagnostic[/bold cyan]",
            border_style="cyan",
        ))

    if report.cleaned_file_path:
        console.print(Panel(
            f"[cyan]→[/cyan] [dim]{report.cleaned_file_path.resolve()}[/dim]",
            title="[bold green]✅ Cleaned File Exported[/bold green]",
            border_style="green",
        ))

    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
