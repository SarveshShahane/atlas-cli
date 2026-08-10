"""
Rich Terminal Renderer — Phase 10 Reproducibility Engine.

Renders snapshot integrity reports, model metric comparison tables, and
reproducibility verdict panels.
"""
from __future__ import annotations

import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.reproducibility.replayer import ReplayResult

logger = logging.getLogger("atlas_cli")
console = Console()


def render_replay_report(result: ReplayResult) -> None:
    """Render full reproducibility replay report to terminal."""

    # Header Panel
    status_str = "[bold bright_green]VALID[/bold bright_green]" if result.snapshot_valid else "[bold red]INVALID / WARN[/bold red]"
    console.print(Panel(
        f"[bold cyan]Reproducibility Replay Engine[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow]         {result.run_id}\n"
        f"[yellow]Dataset:[/yellow]        {result.dataset_path}\n"
        f"[yellow]Snapshot Status:[/yellow] {status_str}\n"
        f"[yellow]Models Replayed:[/yellow] {result.num_models_tested}\n"
        f"[yellow]Exact Matches:[/yellow]   {result.num_exact_matches} / {len(result.metric_diffs)} metrics",
        title="[bold blue]atlas replay[/bold blue]",
        border_style="cyan",
    ))

    # Warnings if any
    if result.warnings:
        warn_text = "\n".join(f"• [yellow]{w}[/yellow]" for w in result.warnings)
        console.print(Panel(
            warn_text,
            title="[bold yellow]⚠️ Snapshot Warnings[/bold yellow]",
            border_style="yellow",
        ))

    # Metric Comparison Table
    if result.metric_diffs:
        console.rule("[bold cyan]Metric Reproducibility Verification[/bold cyan]")

        table = Table(
            title="🔄 Original vs. Replayed Metrics Comparison",
            border_style="bright_cyan",
            header_style="bold bright_cyan",
            show_lines=True,
            title_style="bold bright_white",
            pad_edge=True,
        )

        table.add_column("Model", style="bold white", min_width=20)
        table.add_column("Metric", style="cyan", min_width=12)
        table.add_column("Original", justify="right", min_width=10)
        table.add_column("Replayed", justify="right", min_width=10)
        table.add_column("Diff", justify="right", min_width=10)
        table.add_column("Status", justify="center", min_width=12)

        for diff in result.metric_diffs:
            if diff.is_exact_match:
                status_cell = Text("✅ Match", style="bold green")
                diff_cell = Text("0.0000", style="dim")
            else:
                status_cell = Text("⚠️ Drift", style="bold yellow")
                diff_cell = Text(f"{diff.diff_magnitude:.6f}", style="bold yellow")

            table.add_row(
                diff.model_name,
                diff.metric_name,
                f"{diff.original_value:.4f}",
                f"{diff.replayed_value:.4f}",
                diff_cell,
                status_cell,
            )

        console.print()
        console.print(table)
        console.print()

    # Verdict Panel
    if result.reproduced_successfully:
        console.print(Panel(
            "[bold bright_green]100% REPRODUCIBILITY VERIFIED[/bold bright_green]\n\n"
            "All model evaluation metrics, random splits, and execution results "
            "matched the original run snapshot exactly.",
            title="[bold bright_green]🏆 Replay Verdict[/bold bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            "[bold yellow]REPRODUCIBILITY DRIFT DETECTED[/bold yellow]\n\n"
            "Some metric values differed from the original run. "
            "Check dataset modifications or dependency versions.",
            title="[bold yellow]⚠️ Replay Verdict[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
