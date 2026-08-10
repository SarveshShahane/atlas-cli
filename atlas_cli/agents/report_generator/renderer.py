"""
Rich Terminal Renderer — Phase 9 Report Generator.

Displays a summary of the generated report with section availability
and output file paths.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger("atlas_cli")
console = Console()


def render_report_summary(
    ctx: dict[str, Any],
    outputs: dict[str, Path],
) -> None:
    """Render a terminal summary of the generated report."""

    # Section availability table
    sections = [
        ("Dataset Intelligence", ctx.get("has_dataset", False)),
        ("Data Quality Profile", ctx.get("has_quality", False)),
        ("Risk Assessment", ctx.get("has_risks", False)),
        ("Execution Plan", ctx.get("has_plan", False)),
        ("Experiment Results", ctx.get("has_experiments", False)),
        ("Model Comparison", ctx.get("has_comparison", False)),
        ("AI Reviewer Notes", ctx.get("has_critique", False)),
        ("Model Explainability", ctx.get("has_explainability", False)),
        ("SHAP Visualizations", ctx.get("has_plots", False)),
    ]

    table = Table(
        title="📋 Report Sections",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=False,
        title_style="bold bright_white",
    )
    table.add_column("Section", style="white", min_width=25)
    table.add_column("Status", justify="center", min_width=10)

    included = 0
    for name, available in sections:
        if available:
            table.add_row(name, "[bright_green]✅ Included[/bright_green]")
            included += 1
        else:
            table.add_row(name, "[dim]⬜ Skipped[/dim]")

    console.print()
    console.print(table)
    console.print()

    # Summary stats
    winner_name = "—"
    if ctx.get("has_comparison") and ctx["comparison"].get("winner"):
        winner_name = ctx["comparison"]["winner"]["model_name"]

    console.print(Panel(
        f"[bold cyan]Executive Report Generated[/bold cyan]\n\n"
        f"[yellow]Run ID:[/yellow]       {ctx['run_id']}\n"
        f"[yellow]Sections:[/yellow]     [bright_green]{included}[/bright_green] / {len(sections)} included\n"
        f"[yellow]Winner Model:[/yellow] [bright_green]{winner_name}[/bright_green]\n"
        f"[yellow]Generated:[/yellow]    {ctx['generated_at']}",
        title="[bold blue]atlas report[/bold blue]",
        border_style="cyan",
    ))

    # Output files
    file_lines = []
    for label, path in outputs.items():
        file_lines.append(
            f"[cyan]→[/cyan] {label.upper()}: [dim]{path.resolve()}[/dim]"
        )

    if file_lines:
        console.print(Panel(
            "\n".join(file_lines),
            title="[bold blue]📁 Generated Files[/bold blue]",
            border_style="blue",
        ))
