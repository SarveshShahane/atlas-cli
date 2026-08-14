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
from rich.tree import Tree

from atlas_cli.cli.commands.analyze_cmd import analyze
from atlas_cli.cli.commands.clean_cmd import clean
from atlas_cli.cli.commands.plan_cmd import plan
from atlas_cli.cli.commands.experiment_cmd import experiment
from atlas_cli.cli.commands.report_cmd import report
from atlas_cli.cli.commands.export_cmd import export
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
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    effective_run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = settings.workspace_dir / "runs" / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold green]🚀 Atlas Pipeline Execution — Run ID: {effective_run_id}[/bold green]")
    console.print(Panel(
        f"[bold]Dataset:[/bold]     {file_path.resolve()}\n"
        f"[bold]Goal:[/bold]        {goal}\n"
        f"[bold]Workspace:[/bold]   {run_dir.resolve()}\n"
        f"[bold]Target Override:[/bold] {target or 'Auto-detect'}\n"
        f"[bold]Ignored Cols:[/bold]    {ignore or 'None'}",
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
        clean(file_path=file_path, target=target, ignore=ignore, output=None, run_id=effective_run_id)
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
    console.rule("[bold cyan]Stage 4/6: Leakage-Free Experimentation & Model Training[/bold cyan]")
    try:
        experiment(run_id=effective_run_id, file_path=file_path, ignore=ignore, parallel=4, seed=42, test_size=test_size)
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
    console.print(Panel(
        f"[bold green]Successfully completed all 6 pipeline stages![/bold green]\n"
        f"[bold]Run Workspace Directory:[/bold]\n"
        f"[dim]{run_dir.resolve()}[/dim]",
        title="[bold green]✅ Pipeline Success[/bold green]",
        border_style="green",
    ))
