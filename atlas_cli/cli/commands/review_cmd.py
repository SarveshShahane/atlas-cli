"""
atlas review command — AI Reviewer & Auto-Critique Loop entry point.

Inspects model metrics, performs rule-based & LLM diagnostic audit, retrains
one refined regularized model iteration, and prints optimization rationale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from atlas_cli.agents.reviewer.renderer import render_critique_report
from atlas_cli.agents.reviewer.runner import review_and_refine
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


def review(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r",
        help="Run ID of experiments to review. Defaults to the most recent run.",
    ),
    seed: int = typer.Option(
        42, "--seed", "-s",
        help="Random seed for refined training reproducibility.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="LLM model string override for reflection critique.",
    ),
    max_iterations: int = typer.Option(
        1, "--max-iterations", "-i",
        help="Number of auto-critique refinement iterations (1 to 3).",
    ),
) -> None:
    """Auto-diagnose model issues (overfitting, underfitting) & execute 1-pass critique-and-refine retry loop."""

    effective_run_id = run_id or _find_latest_run_id()
    if not effective_run_id:
        console.print(Panel(
            "[bold red]No runs found.[/bold red]\n"
            "[dim]Run 'atlas plan', 'atlas experiment', and 'atlas compare' first.[/dim]",
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    run_dir = settings.workspace_dir / "runs" / effective_run_id
    if not run_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Run directory not found: {run_dir}")
        raise typer.Exit(code=1)

    if not (run_dir / "experiment_results.json").exists():
        console.print(
            f"[bold red]Error:[/bold red] No experiment_results.json in run {effective_run_id}. "
            "Run 'atlas experiment' first."
        )
        raise typer.Exit(code=1)

    try:
        with console.status("[cyan]Running AI Reviewer & Auto-Critique Loop...[/cyan]"):
            report = review_and_refine(
                effective_run_id,
                random_seed=seed,
                model_override=model,
            )
    except Exception as exc:
        console.print(f"[bold red]AI Reviewer failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.rule("[bold cyan]AI Reviewer & Refinement Report[/bold cyan]")
    render_critique_report(report)

    critique_path = run_dir / "critique_report.json"
    console.print(Panel(
        f"[cyan]→[/cyan] Critique Report: [dim]{critique_path.resolve()}[/dim]\n"
        f"[cyan]→[/cyan] Models directory: [dim]{(run_dir / 'models').resolve()}[/dim]",
        title="[bold blue]📁 Artifacts Saved[/bold blue]",
        border_style="blue",
    ))

    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
