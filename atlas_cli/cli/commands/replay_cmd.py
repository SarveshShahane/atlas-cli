"""
atlas replay command — Reproducibility Engine entry point.

Reloads run snapshot metadata, validates dataset integrity, and re-executes
preprocessing & parallel multi-model experiments with exact random seeds to
verify 100% metric identity.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from atlas_cli.agents.reproducibility.renderer import render_replay_report
from atlas_cli.agents.reproducibility.replayer import replay_run
from atlas_cli.core.config import settings

console = Console()


def replay(
    run_id: str = typer.Argument(
        ...,
        help="Run ID of the experiment workflow to reproduce.",
    ),
) -> None:
    """Reload run metadata, pipeline configuration, and random seeds to execute exact replay."""

    run_dir = settings.workspace_dir / "runs" / run_id
    if not run_dir.exists():
        console.print(Panel(
            f"[bold red]Run directory not found:[/bold red] {run_dir}\n"
            "[dim]Check the run ID or list available runs in .atlas_cli/runs/[/dim]",
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    try:
        with console.status(f"[cyan]Executing reproducibility replay for run {run_id}...[/cyan]"):
            result = replay_run(run_id)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]Replay execution failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.rule("[bold cyan]Reproducibility Replay Verification[/bold cyan]")
    render_replay_report(result)

    console.rule(f"[dim]Run ID: {run_id}[/dim]")
