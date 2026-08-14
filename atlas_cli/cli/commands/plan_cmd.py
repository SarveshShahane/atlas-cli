"""
atlas plan command — Pipeline Planner entry point.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import uuid

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.dataset_intelligence.loader import load_dataset
from atlas_cli.agents.dataset_intelligence.profiler import profile_dataset
from atlas_cli.agents.dataset_intelligence.reporter import export_json_artifacts
from atlas_cli.agents.dataset_intelligence.risk import assess_risks
from atlas_cli.agents.dataset_intelligence.schema import infer_schema
from atlas_cli.agents.pipeline_planner import llm_client
from atlas_cli.agents.pipeline_planner.planner import run_planner
from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan
from atlas_cli.core.config import settings

console = Console()


def plan(
    file_path: Path = typer.Argument(..., help="Path to the dataset file (CSV, Parquet, JSON)"),
    goal: str = typer.Option(..., "--goal", "-g", help="Natural-language prediction/analysis goal"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    ignore: Optional[str] = typer.Option(None, "--ignore", "-i", help="Comma-separated list of column names to ignore/exclude"),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Reuse an existing run ID (skips re-analysis)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model string override"),
) -> None:
    """
    Generate an LLM-driven ML execution plan for a dataset.

    If Phase 2 artifacts already exist for the run ID they are reused;
    otherwise the Dataset Intelligence pipeline runs inline first.

    Examples:
        atlas plan Iris.csv --goal "classify iris species" --target Species --ignore Id
        atlas plan sales.csv --goal "predict monthly revenue" --target revenue --model groq/llama-3.3-70b-versatile
    """
    effective_run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = settings.workspace_dir / "runs" / effective_run_id

    ignore_cols = [c.strip() for c in ignore.split(",") if c.strip()] if ignore else []
    needs_analyze = not (run_dir / "dataset_summary.json").exists()

    if needs_analyze:
        if not file_path.exists():
            console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
            raise typer.Exit(code=1)

        console.print(
            f"[dim]No existing analysis found for run [bold]{effective_run_id}[/bold]. "
            "Running Dataset Intelligence pipeline first...[/dim]"
        )
        _run_analyze_inline(file_path, target, ignore_cols, run_dir, effective_run_id)

    try:
        provider = llm_client.validate_api_keys()
    except RuntimeError as exc:
        console.print(Panel(
            f"[bold red]{exc}[/bold red]",
            title="[bold red]❌ No LLM API Key Configured[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    console.print(f"[dim]LLM Provider: [bold cyan]{provider}[/bold cyan]  Model: [bold]{model or settings.llm_model}[/bold][/dim]")

    with console.status("[cyan]Running Pipeline Planner (LLM reasoning)...[/cyan]"):
        try:
            plan_obj: ExecutionPlan = run_planner(
                goal=goal,
                run_dir=run_dir,
                model=model,
                temperature=0.2,
            )
            if ignore_cols:
                drop_set = set(plan_obj.preprocessing.drop_columns or [])
                drop_set.update(ignore_cols)
                plan_obj.preprocessing.drop_columns = list(drop_set)
                (run_dir / "execution_plan.json").write_text(
                    json.dumps(plan_obj.to_dict(), indent=2, default=str),
                    encoding="utf-8"
                )
        except RuntimeError as exc:
            console.print(f"[bold red]Planner failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    console.rule("[bold cyan]Execution Plan[/bold cyan]")
    _render_plan(plan_obj, run_dir)
    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")


def _run_analyze_inline(file_path: Path, target: Optional[str], ignore_cols: list[str], run_dir: Path, run_id: str) -> None:
    """Run the Dataset Intelligence pipeline and export artifacts to run_dir."""
    with console.status(f"[cyan]Loading {file_path.name}...[/cyan]"):
        df, meta = load_dataset(file_path)
        if ignore_cols:
            df = df.drop(columns=[c for c in ignore_cols if c in df.columns], errors="ignore")
    with console.status("[cyan]Inferring schema...[/cyan]"):
        schema = infer_schema(df)
    with console.status("[cyan]Profiling dataset...[/cyan]"):
        profile = profile_dataset(df, target_col=target)
    with console.status("[cyan]Assessing risks...[/cyan]"):
        risk = assess_risks(df, schema, profile, target_col=target)
    export_json_artifacts(run_dir, meta, schema, profile, risk)
    console.print("[green][OK][/green] Dataset analysis complete.")


def _render_plan(plan: ExecutionPlan, run_dir: Path) -> None:
    """Render the ExecutionPlan as rich terminal output."""
    task_style = {
        "binary_classification": "cyan",
        "multiclass_classification": "blue",
        "regression": "magenta",
        "clustering": "yellow",
        "time_series_forecasting": "orange3",
    }.get(plan.task_type, "white")

    console.print(Panel(
        f"[bold]Task Type:[/bold]     [{task_style}]{plan.task_type.replace('_', ' ').title()}[/{task_style}]\n"
        f"[bold]Target Column:[/bold] [green]{plan.target_column}[/green]\n\n"
        f"[bold]Reasoning:[/bold]\n[dim]{plan.reasoning}[/dim]",
        title="[bold blue]🎯 ML Task & Strategy[/bold blue]",
        border_style="blue",
    ))

    pre = plan.preprocessing
    pre_table = Table(title="🔧 Preprocessing Strategy", border_style="cyan", header_style="bold cyan", show_lines=False)
    pre_table.add_column("Decision")
    pre_table.add_column("Value", style="white")
    pre_table.add_row("Missing Values",   pre.missing_strategy)
    pre_table.add_row("Outlier Handling", pre.outlier_strategy)
    pre_table.add_row("Scaling",          pre.scale_strategy)
    pre_table.add_row("Drop Columns",     ", ".join(pre.drop_columns) if pre.drop_columns else "—")
    if pre.notes:
        pre_table.add_row("Notes", Text(pre.notes, style="dim"))
    console.print(pre_table)

    fe = plan.feature_engineering
    fe_table = Table(title="⚙️  Feature Engineering", border_style="magenta", header_style="bold magenta", show_lines=False)
    fe_table.add_column("Decision")
    fe_table.add_column("Value", style="white")
    fe_table.add_row("Numeric Transforms",    ", ".join(fe.numeric_transforms) if fe.numeric_transforms else "—")
    fe_table.add_row("Categorical Encoding",  fe.categorical_strategy)
    fe_table.add_row("Datetime Features",     ", ".join(fe.datetime_features) if fe.datetime_features else "—")
    fe_table.add_row("Interaction Features",  "Yes" if fe.interaction_features else "No")
    fe_table.add_row("Text Vectorizer",       fe.text_vectorizer or "—")
    console.print(fe_table)

    mc_table = Table(title="🤖 Model Candidates", border_style="green", header_style="bold green", show_lines=True)
    mc_table.add_column("#", justify="center", width=4)
    mc_table.add_column("Model", style="bold white")
    mc_table.add_column("Library", style="dim")
    mc_table.add_column("Rationale", overflow="fold")

    for candidate in sorted(plan.model_candidates, key=lambda m: m.priority):
        mc_table.add_row(str(candidate.priority), candidate.name, candidate.library, candidate.rationale)
    console.print(mc_table)

    ev = plan.evaluation
    imbalance_text = "[bold yellow]Yes — SMOTE / class_weight[/bold yellow]" if ev.handle_imbalance else "[dim]No[/dim]"
    console.print(Panel(
        f"[bold]Primary Metric:[/bold]  [cyan]{ev.primary_metric}[/cyan]\n"
        f"[bold]Secondary Metrics:[/bold] {', '.join(ev.secondary_metrics) if ev.secondary_metrics else '—'}\n"
        f"[bold]CV Strategy:[/bold]    {ev.cv_strategy}  (folds: {ev.n_folds})\n"
        f"[bold]Test Split:[/bold]     {ev.test_size * 100:.0f}%\n"
        f"[bold]Handle Imbalance:[/bold] {imbalance_text}",
        title="[bold yellow]📐 Evaluation Configuration[/bold yellow]",
        border_style="yellow",
    ))

    plan_path = run_dir / "execution_plan.json"
    plan_dir = run_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    sub_plan_path = plan_dir / "execution_plan.json"

    plan_json_data = json.dumps(plan.model_dump(), indent=2)
    plan_path.write_text(plan_json_data, encoding="utf-8")
    sub_plan_path.write_text(plan_json_data, encoding="utf-8")

    console.print(Panel(
        f"[cyan]→[/cyan] [dim]{plan_path.resolve()}[/dim]\n"
        f"[cyan]→[/cyan] [dim]{sub_plan_path.resolve()}[/dim]",
        title="[bold green]✅ Plan Saved[/bold green]",
        border_style="green",
    ))
