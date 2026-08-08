"""
atlas experiment command — Parallel Experimentation Engine entry point.
"""
from __future__ import annotations

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
    table.add_column("Duration", justify="right", min_width=10)

    successful = sorted(
        [r for r in results if r.status == "success"],
        key=lambda r: r.metrics.get(primary_metric, 0),
        reverse=True,
    )
    failed = [r for r in results if r.status == "failed"]
    ordered = successful + failed

    for i, result in enumerate(ordered, 1):
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
        table.add_row(str(i), result.model_name, status_text, metric_val, duration)

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
            title="[bold green]🏆 Best Model[/bold green]",
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
        f"[cyan]→[/cyan] Models:  [dim]{(run_dir / 'models').resolve()}[/dim]\n"
        f"[cyan]→[/cyan] Summary: [dim]{(run_dir / 'experiment_results.json').resolve()}[/dim]\n"
        f"[cyan]→[/cyan] Splits:  [dim]{(run_dir / 'splits').resolve()}[/dim]",
        title="[bold blue]📁 Artifacts Saved[/bold blue]",
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
    seed: int = typer.Option(
        42, "--seed", "-s",
        help="Random seed for reproducibility.",
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

    if not (run_dir / "execution_plan.json").exists():
        console.print(
            f"[bold red]Error:[/bold red] No execution_plan.json in run {effective_run_id}. "
            "Run 'atlas plan' first."
        )
        raise typer.Exit(code=1)

    console.print(Panel(
        f"[bold cyan]Parallel Experimentation Engine[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow]       {effective_run_id}\n"
        f"[yellow]Max Workers:[/yellow]  {parallel}\n"
        f"[yellow]Random Seed:[/yellow]  {seed}",
        title="[bold blue]atlas experiment[/bold blue]",
        border_style="cyan",
    ))

    try:
        with console.status("[cyan]Running feature engineering & model training...[/cyan]"):
            results = run_experiments(
                effective_run_id,
                max_workers=parallel,
                file_path=file_path,
                random_seed=seed,
            )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]Experiment engine failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    import json
    plan_data = json.loads((run_dir / "execution_plan.json").read_text(encoding="utf-8"))
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
