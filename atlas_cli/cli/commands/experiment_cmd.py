"""
atlas experiment command — Parallel Experimentation Engine entry point.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.experimentation.runner import run_experiments
from atlas_cli.agents.experimentation.worker import ExperimentResult
from atlas_cli.core.config import settings

console = Console()


def _find_latest_run_id() -> Optional[str]:
    """Find the most recently created run directory."""
    runs_dir = settings.workspace_dir / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0].name if run_dirs else None


def _render_results(results: list[ExperimentResult], run_id: str, primary_metric: str) -> None:
    """Render experiment results as a rich summary table."""
    table = Table(
        title="⚡ Experiment Results",
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", justify="center", width=4)
    table.add_column("Model", style="bold white", min_width=18)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column(primary_metric.upper(), justify="right", min_width=10)
    table.add_column("Top Feature", style="dim", justify="center")
    table.add_column("Duration", justify="right", min_width=10)

    successful = sorted(
        [r for r in results if r.status == "success"],
        key=lambda r: r.metrics.get(primary_metric, 0),
        reverse=True,
    )
    failed = [r for r in results if r.status == "failed"]
    ordered = successful + failed

    for i, result in enumerate(ordered, 1):
        is_ensemble = result.model_name == "Weighted Ensemble"
        status_text = (
            Text("✓ OK", style="bold green")
            if result.status == "success"
            else Text("✗ FAIL", style="bold red")
        )
        metric_val = (
            f"{result.metrics.get(primary_metric, 0):.4f}"
            if result.status == "success"
            else "—"
        )
        duration = f"{result.duration_seconds:.2f}s"

        top_feat = "—"
        if result.feature_importances:
            top_name, top_val = next(iter(result.feature_importances.items()))
            top_feat = f"{top_name} ({top_val:.2f})"
        elif is_ensemble:
            top_feat = "Soft Voting Blend"

        model_display = Text(result.model_name, style="bold green" if is_ensemble else "bold white")
        table.add_row(str(i), model_display, status_text, Text(metric_val, style="bold cyan" if is_ensemble else "white"), top_feat, duration)

    console.print(table)

    if successful:
        best = successful[0]
        metrics_lines = "\n".join(
            f"  [cyan]{k}:[/cyan] {v:.4f}" for k, v in sorted(best.metrics.items())
        )
        console.print(Panel(
            f"[bold green]{best.model_name}[/bold green]  ({best.library})\n"
            f"[bold]{primary_metric}:[/bold] [cyan]{best.metrics.get(primary_metric, 0):.4f}[/cyan]\n\n"
            f"[dim]All metrics:[/dim]\n{metrics_lines}",
            title="[bold green]🏆 Best Model / Winner[/bold green]",
            border_style="green",
        ))

    if failed:
        for result in failed:
            error_snippet = (result.error_message or "Unknown error")[:300]
            console.print(Panel(
                f"[bold red]{result.model_name}[/bold red]  ({result.library})\n"
                f"[dim]{error_snippet}[/dim]",
                title="[bold red]❌ Failed Model[/bold red]",
                border_style="red",
            ))

    run_dir = settings.workspace_dir / "runs" / run_id
    console.print(Panel(
        f"[cyan]→[/cyan] Models:        [dim]{(run_dir / 'models').resolve()}[/dim]\n"
        f"[cyan]→[/cyan] Summary:       [dim]{(run_dir / 'experiment_results.json').resolve()}[/dim]\n"
        f"[cyan]→[/cyan] Feature Data:  [dim]{(run_dir / 'feature_engineered_data.csv').resolve()}[/dim]\n"
        f"[cyan]→[/cyan] Train/Test:    [dim]{(run_dir / 'splits').resolve()}[/dim]\n"
        f"[dim italic](Tip: Run 'atlas export' to copy cleaned/FE/split datasets anywhere in CSV/Parquet)[/dim italic]",
        title="[bold blue]📁 Artifacts Saved & Dataset Outputs[/bold blue]",
        border_style="blue",
    ))


def experiment(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r",
        help="Run ID of the execution plan. Defaults to the most recent run.",
    ),
    parallel: int = typer.Option(
        4, "--parallel", "-p",
        help="Maximum number of concurrent training workers.",
    ),
    file_path: Optional[Path] = typer.Option(
        None, "--file-path", "-f",
        help="Dataset file path override (if different from the original plan run).",
    ),
    ignore: Optional[str] = typer.Option(
        None, "--ignore", "-i",
        help="Comma-separated column names to ignore/exclude from features.",
    ),
    seed: int = typer.Option(
        42, "--seed", "-s",
        help="Random seed for reproducibility.",
    ),
    test_size: Optional[float] = typer.Option(
        None, "--test-size", "-ts",
        help="Custom test split ratio override (e.g. 0.2 for 20% test split).",
    ),
) -> None:
    """Execute parallel multi-model training experiments for a given run plan."""

    effective_run_id = run_id or _find_latest_run_id()
    if not effective_run_id:
        console.print(Panel(
            "[bold red]No runs found.[/bold red]\n"
            "[dim]Run 'atlas plan <dataset> --goal <goal>' first to create an execution plan.[/dim]",
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    run_dir = settings.workspace_dir / "runs" / effective_run_id
    if not run_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Run directory not found: {run_dir}")
        raise typer.Exit(code=1)

    plan_file = run_dir / "execution_plan.json"
    if not plan_file.exists():
        plan_file = run_dir / "plan" / "execution_plan.json"
    if not plan_file.exists():
        console.print(
            f"[bold red]Error:[/bold red] No execution_plan.json in run {effective_run_id}. "
            "Run 'atlas plan' first."
        )
        raise typer.Exit(code=1)

    plan_modified = False
    plan_dict = json.loads(plan_file.read_text(encoding="utf-8"))

    if ignore and isinstance(ignore, str):
        ignore_cols = [c.strip() for c in ignore.split(",") if c.strip()]
        pre = plan_dict.setdefault("preprocessing", {})
        drop_cols = set(pre.get("drop_columns", []))
        drop_cols.update(ignore_cols)
        pre["drop_columns"] = list(drop_cols)
        plan_modified = True

    if test_size is not None and isinstance(test_size, (float, int)):
        eval_cfg = plan_dict.setdefault("evaluation", {})
        eval_cfg["test_size"] = float(test_size)
        plan_modified = True

    if plan_modified:
        plan_bytes = json.dumps(plan_dict, indent=2)
        (run_dir / "execution_plan.json").write_text(plan_bytes, encoding="utf-8")
        plan_sub_dir = run_dir / "plan"
        plan_sub_dir.mkdir(parents=True, exist_ok=True)
        (plan_sub_dir / "execution_plan.json").write_text(plan_bytes, encoding="utf-8")

    parallel_val = parallel if isinstance(parallel, int) else 4
    seed_val = seed if isinstance(seed, int) else 42

    console.print(Panel(
        f"[bold cyan]Parallel Experimentation Engine[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow]       {effective_run_id}\n"
        f"[yellow]Max Workers:[/yellow]  {parallel_val}\n"
        f"[yellow]Random Seed:[/yellow]  {seed_val}",
        title="[bold yellow]⚡ Experimentation Context[/bold yellow]",
        border_style="yellow",
    ))

    try:
        with console.status("[cyan]Running feature engineering, early stopping & model training...[/cyan]"):
            results = run_experiments(
                effective_run_id,
                max_workers=parallel_val,
                file_path=file_path,
                random_seed=seed_val,
            )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]Experiment engine failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
    primary_metric = plan_data.get("evaluation", {}).get("primary_metric", "accuracy")

    succeeded = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    console.rule("[bold cyan]Experiment Results[/bold cyan]")
    console.print(
        f"  [green]{succeeded} succeeded[/green]  |  "
        f"[red]{failed} failed[/red]  |  "
        f"[dim]{len(results)} total[/dim]"
    )
    console.print()

    _render_results(results, effective_run_id, primary_metric)
    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
