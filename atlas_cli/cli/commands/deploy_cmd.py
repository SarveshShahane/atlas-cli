"""
atlas deploy command — Deployment Scaffold Generator entry point.

Packages selected or winning model into a complete production FastAPI
inference microservice directory with Dockerfile, requirements.txt, and tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from atlas_cli.agents.deployment.renderer import render_deployment_summary
from atlas_cli.agents.deployment.scaffolder import scaffold_deployment
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


def deploy(
    experiment_id: Optional[str] = typer.Option(
        None, "--exp-id", "-e",
        help="Winning Experiment ID / model name to package. Defaults to winning model.",
    ),
    output_dir: str = typer.Option(
        "./deploy", "--out", "-o",
        help="Target microservice scaffold directory.",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r",
        help="Run ID of the experiment workflow. Defaults to the latest run.",
    ),
) -> None:
    """Package selected model into production FastAPI inference endpoint directory."""

    # Resolve run ID
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

    try:
        with console.status("[cyan]Scaffolding FastAPI microservice...[/cyan]"):
            result = scaffold_deployment(
                effective_run_id,
                output_dir,
                experiment_id=experiment_id,
            )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]Deployment scaffolding failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.rule("[bold cyan]FastAPI Microservice Scaffolded[/bold cyan]")
    render_deployment_summary(result)

    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
