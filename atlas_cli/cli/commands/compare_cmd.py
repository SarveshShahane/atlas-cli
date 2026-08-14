"""
atlas compare command — Experiment Comparator & Metric Evaluator entry point.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from atlas_cli.agents.comparator.inference import generate_inference_script
from atlas_cli.agents.comparator.renderer import (
    render_comparison_table,
    render_feature_consensus_table,
    render_tradeoff_notes,
    render_winner_panel,
    save_comparison_json,
)
from atlas_cli.agents.comparator.scorer import PRESETS, rank_experiments
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


def _parse_weights(weights_str: Optional[str], preset_name: str) -> tuple[float, float, float, float]:
    """Parse custom weights string or resolve preset weights."""
    if weights_str:
        try:
            parts = [float(p.strip()) for p in weights_str.split(",")]
            if len(parts) == 4:
                total = sum(parts)
                return tuple(p / total for p in parts)  # type: ignore
        except Exception:
            pass

    return PRESETS.get(preset_name.lower(), PRESETS["balanced"])


def compare(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r",
        help="Run ID of experiments to evaluate. Defaults to the most recent run.",
    ),
    preset: str = typer.Option(
        "balanced", "--preset", "-p",
        help="Scoring preset: 'balanced', 'performance', 'edge', or 'fast'.",
    ),
    weights: Optional[str] = typer.Option(
        None, "--weights", "-w",
        help="Custom multi-objective weights 'w_perf,w_lat,w_size,w_cost' (e.g. '0.7,0.1,0.1,0.1').",
    ),
) -> None:
    """Compare experiments with multi-objective scoring, feature consensus, and export production inference package."""

    effective_run_id = run_id or _find_latest_run_id()
    if not effective_run_id:
        console.print(Panel(
            "[bold red]No runs found.[/bold red]\n"
            "[dim]Run 'atlas plan <dataset> --goal <goal>' and 'atlas experiment' first.[/dim]",
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    run_dir = settings.workspace_dir / "runs" / effective_run_id
    if not run_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Run directory not found: {run_dir}")
        raise typer.Exit(code=1)

    plan_path = run_dir / "execution_plan.json"
    if not plan_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] No execution_plan.json in run {effective_run_id}. "
            "Run 'atlas plan' first."
        )
        raise typer.Exit(code=1)

    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    task_type = plan_data.get("task_type", "binary_classification")
    primary_metric = plan_data.get("evaluation", {}).get("primary_metric", "accuracy")

    results_path = run_dir / "experiment_results.json"
    if not results_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] No experiment_results.json in run {effective_run_id}. "
            "Run 'atlas experiment' first."
        )
        raise typer.Exit(code=1)

    results_data = json.loads(results_path.read_text(encoding="utf-8"))
    experiment_entries = results_data.get("experiments", [])

    if not experiment_entries:
        console.print("[bold yellow]No experiments found in this run.[/bold yellow]")
        raise typer.Exit(code=1)

    num_succeeded = sum(1 for e in experiment_entries if e.get("status") == "success")
    num_failed = sum(1 for e in experiment_entries if e.get("status") != "success")
    effective_weights = _parse_weights(weights, preset)

    console.print(Panel(
        f"[bold cyan]Experiment Comparator & Metric Evaluator[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow]         {effective_run_id}\n"
        f"[yellow]Task Type:[/yellow]      {task_type}\n"
        f"[yellow]Primary Metric:[/yellow] {primary_metric}\n"
        f"[yellow]Preset / Weights:[/yellow] [magenta]{preset.upper()}[/magenta] {effective_weights}\n"
        f"[yellow]Experiments:[/yellow]    [green]{num_succeeded} succeeded[/green]  |  "
        f"[red]{num_failed} failed[/red]  |  [dim]{len(experiment_entries)} total[/dim]",
        title="[bold blue]atlas compare[/bold blue]",
        border_style="cyan",
    ))

    try:
        with console.status("[cyan]Computing extended metrics, feature consensus & composite scores...[/cyan]"):
            ranked = rank_experiments(
                effective_run_id,
                task_type=task_type,
                primary_metric=primary_metric,
                experiment_entries=experiment_entries,
                weights=effective_weights,
            )
    except Exception as exc:
        console.print(f"[bold red]Comparator failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not ranked:
        console.print("[bold yellow]No experiments could be ranked.[/bold yellow]")
        raise typer.Exit(code=1)

    console.rule("[bold cyan]Multi-Objective Comparison[/bold cyan]")

    render_comparison_table(ranked, task_type, primary_metric)
    render_feature_consensus_table(experiment_entries)
    render_winner_panel(ranked, primary_metric, task_type)
    render_tradeoff_notes(ranked, primary_metric)

    output_path = save_comparison_json(effective_run_id, ranked, task_type, primary_metric)

    # Generate standalone prediction inference package for winner
    winners = [r for r in ranked if r.is_winner]
    predict_script_path: Optional[Path] = None
    if winners:
        predict_script_path = generate_inference_script(effective_run_id, winners[0])

    artifacts_msg = f"[cyan]→[/cyan] Comparison: [dim]{output_path.resolve()}[/dim]\n"
    if predict_script_path:
        artifacts_msg += f"[cyan]→[/cyan] Inference:  [bold green]{predict_script_path.resolve()}[/bold green]  [dim](python predict.py -i <data.csv>)[/dim]\n"
    artifacts_msg += f"[cyan]→[/cyan] Run dir:    [dim]{run_dir.resolve()}[/dim]"

    console.print(Panel(
        artifacts_msg,
        title="[bold blue]📁 Artifacts & Deployment Package Saved[/bold blue]",
        border_style="blue",
    ))

    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
