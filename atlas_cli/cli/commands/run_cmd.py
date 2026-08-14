"""
atlas run command — End-to-End Autonomous Data Science Pipeline.
Executes analyze -> clean -> plan -> experiment -> report -> export in one command
with complete workspace isolation per run ID.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from atlas_cli.cli.commands.analyze_cmd import analyze
from atlas_cli.cli.commands.clean_cmd import clean
from atlas_cli.cli.commands.plan_cmd import plan
from atlas_cli.cli.commands.experiment_cmd import experiment
from atlas_cli.cli.commands.report_cmd import report
from atlas_cli.cli.commands.export_cmd import export
from atlas_cli.agents.experimentation.consistency import run_consistency_checks
from atlas_cli.core.config import settings

console = Console()


def run_pipeline(
    file_path: Path = typer.Argument(..., help="Path to raw dataset file (CSV, Parquet, JSON)"),
    goal: str = typer.Option(..., "--goal", "-g", help="High-level goal or target column description"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name override"),
    ignore: Optional[str] = typer.Option(None, "--ignore", "-i", help="Comma-separated column names to ignore/drop"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Target output directory for exported datasets"),
    output_format: str = typer.Option("csv", "--format", "-f", help="Output format for datasets: 'csv' or 'parquet'"),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run ID for workspace isolation (auto-generated if omitted)"),
    test_size: Optional[float] = typer.Option(None, "--test-size", "-ts", help="Custom test split ratio override (e.g. 0.2 for 20% test split)"),
    outlier_action: str = typer.Option("report", "--outlier-action", help="Outlier handling: 'report' (default, diagnostic only) or 'cap' (IQR Winsorization)"),
    anomaly_action: str = typer.Option("report", "--anomaly-action", "-a", help="Anomaly handling: 'report' (default, keep all rows), 'flag' (add flag column), 'remove' (delete rows)"),
    no_vif_drop: bool = typer.Option(True, "--no-vif-drop", help="Disable automatic high-VIF column dropping (default: True, VIF is diagnostic only)"),
    no_outlier_cap: bool = typer.Option(True, "--no-outlier-cap", help="Legacy flag: disable outlier capping (default: True)"),
) -> None:
    """
    Execute the entire Autonomous Data Science Pipeline end-to-end.

    Stages executed sequentially:
      1. Analyze   (Dataset Intelligence & Profiling)
      2. Clean     (Winsorization, Collinearity & Anomaly Filtering)
      3. Plan      (LLM Strategy & Preprocessing Planning)
      4. Experiment(Leakage-Free Feature Engineering & Model Training)
      5. Report    (Executive Markdown & HTML Report Generation)
      6. Export    (Cleaned, Feature Engineered, Train/Test Split Exports)

    Example:
      atlas run Iris.csv -g "Predict species" -i Id
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

    if not isinstance(output_format, str):
        output_format = getattr(output_format, "default", "csv")
        if not isinstance(output_format, str):
            output_format = "csv"

    if not isinstance(no_vif_drop, bool):
        no_vif_drop = getattr(no_vif_drop, "default", True)
        if not isinstance(no_vif_drop, bool):
            no_vif_drop = True

    if not isinstance(no_outlier_cap, bool):
        no_outlier_cap = getattr(no_outlier_cap, "default", True)
        if not isinstance(no_outlier_cap, bool):
            no_outlier_cap = True

    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    effective_run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = settings.workspace_dir / "runs" / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold green]🚀 Atlas Pipeline Execution — Run ID: {effective_run_id}[/bold green]")
    console.print(Panel(
        f"[bold]Dataset:[/bold]          {file_path.resolve()}\n"
        f"[bold]Goal:[/bold]             {goal}\n"
        f"[bold]Workspace:[/bold]        {run_dir.resolve()}\n"
        f"[bold]Target Override:[/bold]  {target or 'Auto-detect'}\n"
        f"[bold]Ignored Cols:[/bold]     {ignore or 'None'}\n"
        f"[bold]Outlier Action:[/bold]   {outlier_action}\n"
        f"[bold]Anomaly Action:[/bold]   {anomaly_action}\n"
        f"[bold]VIF Strategy:[/bold]     {'Diagnostic only (keep features)' if no_vif_drop else 'Drop collinear features (VIF >= 10)'}",
        title="[bold cyan]📋 Pipeline Context[/bold cyan]",
        border_style="cyan",
    ))

    # Stage 1: Analyze
    console.rule("[bold cyan]Stage 1/6: Dataset Intelligence & Risk Assessment[/bold cyan]")
    try:
        analyze(file_path=file_path, target=target, ignore=ignore, run_id=effective_run_id)
    except SystemExit as exc:
        if exc.code != 0:
            console.print("[bold red]Pipeline halted at Stage 1 (Analyze).[/bold red]")
            raise exc

    # Stage 2: Clean
    console.rule("[bold cyan]Stage 2/6: Dataset Hygiene & Outlier Cleaning[/bold cyan]")
    try:
        clean(
            file_path=file_path,
            target=target,
            ignore=ignore,
            output=None,
            run_id=effective_run_id,
            no_vif_drop=no_vif_drop,
            outlier_action=outlier_action,
            anomaly_action=anomaly_action,
        )
    except SystemExit as exc:
        if exc.code != 0:
            console.print("[bold red]Pipeline halted at Stage 2 (Clean).[/bold red]")
            raise exc

    # Stage 3: Plan
    console.rule("[bold cyan]Stage 3/6: LLM Strategy & Execution Planning[/bold cyan]")
    try:
        plan(file_path=file_path, goal=goal, target=target, ignore=ignore, run_id=effective_run_id, model=None)
    except SystemExit as exc:
        if exc.code != 0:
            console.print("[bold red]Pipeline halted at Stage 3 (Plan).[/bold red]")
            raise exc

    # Stage 4: Experiment
    # NOTE: file_path is intentionally NOT passed here — the experiment runner
    # must load cleaned_data.csv from the run directory to preserve the
    # cleaning stage's output (correct row count, removed anomalies, etc.).
    console.rule("[bold cyan]Stage 4/6: Leakage-Free Experimentation & Model Training[/bold cyan]")
    try:
        experiment(run_id=effective_run_id, file_path=None, ignore=ignore, parallel=4, seed=42, test_size=test_size)
    except SystemExit as exc:
        if exc.code != 0:
            console.print("[bold red]Pipeline halted at Stage 4 (Experiment).[/bold red]")
            raise exc

    # Stage 5: Report
    console.rule("[bold cyan]Stage 5/6: Executive Report Generation[/bold cyan]")
    try:
        report(run_id=effective_run_id, output_dir=str(run_dir / "reports"))
    except SystemExit as exc:
        if exc.code != 0:
            console.print("[dim yellow]Warning: Report generation encountered an issue.[/dim yellow]")

    # Stage 6: Export
    console.rule("[bold cyan]Stage 6/6: Dataset Export & Packaging[/bold cyan]")
    export_target_dir = output_dir or (run_dir / "exports")
    try:
        export(run_id=effective_run_id, dataset_type="all", output_dir=export_target_dir, output_format=output_format)
    except SystemExit as exc:
        if exc.code != 0:
            console.print("[dim yellow]Warning: Dataset export encountered an issue.[/dim yellow]")

    # ── Automated End-of-Pipeline Consistency Validation Gate ────────────
    console.rule("[bold cyan]🛡️ Pipeline Consistency & Integrity Validation Gate[/bold cyan]")
    all_passed, passed_checks, warning_checks, fail_checks = run_consistency_checks(effective_run_id)

    chk_table = Table(title="Validation Gate Results", border_style="cyan", header_style="bold cyan")
    chk_table.add_column("Check Description", style="white", min_width=40)
    chk_table.add_column("Status", justify="center", min_width=10)

    for p in passed_checks:
        chk_table.add_row(p, "[bold green]PASS[/bold green]")
    for w in warning_checks:
        chk_table.add_row(w, "[bold yellow]WARN[/bold yellow]")
    for f in fail_checks:
        chk_table.add_row(f, "[bold red]FAIL[/bold red]")

    console.print(chk_table)

    # Summary Tree of Isolated Run Workspace
    tree = Tree(f"[bold green]📂 Isolated Run Workspace: {effective_run_id}[/bold green]")
    tree.add("[cyan]analysis/[/cyan] (dataset_summary.json, quality_report.json, risk_assessment.json)")
    tree.add("[cyan]cleaned/[/cyan] (cleaned_data.csv)")
    tree.add("[cyan]plan/[/cyan] (execution_plan.json)")
    tree.add("[cyan]features/[/cyan] (features_meta.json, feature_engineered_data.csv, pipeline.joblib)")
    tree.add("[cyan]splits/[/cyan] (train.csv, val.csv, test.csv, *.npy)")
    tree.add("[cyan]models/[/cyan] (*.joblib, experiment_results.json)")
    tree.add("[cyan]reports/[/cyan] (REPORT.html, REPORT.md)")
    tree.add("[cyan]exports/[/cyan] (cleaned_dataset.csv, feature_engineered_dataset.csv, train_split.csv, val_split.csv, test_split.csv)")

    console.rule("[bold green]🎉 End-to-End Pipeline Execution Complete[/bold green]")
    console.print(tree)

    if not all_passed:
        fail_reasons = "\n".join(f"  • {f}" for f in fail_checks)
        console.print(Panel(
            f"[bold red]Pipeline failed integrity validation gate:[/bold red]\n{fail_reasons}\n\n"
            f"[bold]Run Workspace Directory:[/bold]\n"
            f"[dim]{run_dir.resolve()}[/dim]\n"
            f"[bold]Consistency Gate:[/bold] [bold red]FAILED[/bold red]",
            title="[bold red]❌ Integrity Gate Failed[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    gate_status = "[bold green]PASSED[/bold green]" if not warning_checks else "[bold yellow]PASSED WITH WARNINGS[/bold yellow]"
    console.print(Panel(
        f"[bold green]Successfully completed all 6 pipeline stages![/bold green]\n"
        f"[bold]Run Workspace Directory:[/bold]\n"
        f"[dim]{run_dir.resolve()}[/dim]\n"
        f"[bold]Consistency Gate:[/bold] {gate_status}",
        title="[bold green]✅ Pipeline Success[/bold green]",
        border_style="green",
    ))
