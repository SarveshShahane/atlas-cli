"""
atlas analyze command — Dataset Intelligence Engine entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid

import typer

from atlas_cli.agents.dataset_intelligence.loader import load_dataset
from atlas_cli.agents.dataset_intelligence.profiler import profile_dataset
from atlas_cli.agents.dataset_intelligence.reporter import (
    console,
    export_json_artifacts,
    render_artifacts_summary,
    render_overview,
    render_profile_table,
    render_risk_table,
    render_schema_table,
    render_target_imbalance,
)
from atlas_cli.agents.dataset_intelligence.risk import assess_risks
from atlas_cli.agents.dataset_intelligence.schema import infer_schema
from atlas_cli.core.config import settings


def analyze(
    file_path: Path = typer.Argument(..., help="Path to the dataset file (CSV, Parquet, JSON)"),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column name for imbalance and leakage analysis"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r", help="Run ID to associate this analysis with (auto-generated if omitted)"
    ),
) -> None:
    """
    Run the Dataset Intelligence Engine on a dataset file.

    Performs schema inference, statistical profiling, and risk assessment,
    then exports structured JSON artifacts and prints rich terminal tables.

    Example:
        atlas analyze data/titanic.csv --target Survived
    """
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    with console.status(f"[cyan]Loading dataset: {file_path.name}[/cyan]"):
        try:
            df, meta = load_dataset(file_path)
        except Exception as e:
            console.print(f"[bold red]Failed to load dataset:[/bold red] {e}")
            raise typer.Exit(code=1)

    with console.status("[cyan]Inferring schema and column types...[/cyan]"):
        schema = infer_schema(df)

    with console.status("[cyan]Running statistical profiler...[/cyan]"):
        profile = profile_dataset(df, target_col=target)

    with console.status("[cyan]Assessing data quality risks...[/cyan]"):
        risk = assess_risks(df, schema, profile, target_col=target)

    console.rule("[bold cyan]Dataset Intelligence Report[/bold cyan]")
    render_overview(meta, profile)
    render_schema_table(schema)
    render_profile_table(profile)
    if target:
        render_target_imbalance(profile)
    render_risk_table(risk)

    effective_run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = settings.workspace_dir / "runs" / effective_run_id

    with console.status("[cyan]Saving JSON artifacts...[/cyan]"):
        paths = export_json_artifacts(run_dir, meta, schema, profile, risk)

    render_artifacts_summary(paths)
    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
