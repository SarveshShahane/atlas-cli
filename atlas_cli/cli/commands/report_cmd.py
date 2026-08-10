"""
atlas report command — Automated Report Generator entry point.

Collects all run artifacts, renders Jinja2 Markdown and HTML templates,
and outputs publication-ready executive reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from atlas_cli.agents.report_generator.collector import collect_report_data
from atlas_cli.agents.report_generator.generator import generate_reports
from atlas_cli.agents.report_generator.renderer import render_report_summary
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


def report(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r",
        help="Run ID of the experiment workflow. Defaults to the latest run.",
    ),
    output_dir: str = typer.Option(
        "./reports", "--out", "-o",
        help="Output directory for generated reports.",
    ),
) -> None:
    """Compile dataset summaries, model metrics, charts, and explanations into REPORT.md / REPORT.html."""

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

    # Step 1: Collect report data
    try:
        with console.status("[cyan]Collecting run artifacts...[/cyan]"):
            ctx = collect_report_data(effective_run_id)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]Data collection failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # Step 2: Generate reports
    try:
        with console.status("[cyan]Rendering Markdown and HTML reports...[/cyan]"):
            outputs = generate_reports(ctx, output_dir)
    except Exception as exc:
        console.print(f"[bold red]Report generation failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not outputs:
        console.print("[bold red]No reports were generated.[/bold red]")
        raise typer.Exit(code=1)

    # Step 3: Render terminal summary
    console.rule("[bold cyan]Executive Report Generated[/bold cyan]")
    render_report_summary(ctx, outputs)

    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
