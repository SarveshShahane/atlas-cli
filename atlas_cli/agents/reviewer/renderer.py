"""
Rich Terminal Renderer — Phase 7 AI Reviewer.

Renders stylish Rich terminal panels for AI critique reflection, diagnostic
issue findings, initial vs. refined model comparison tables, and final verdicts.
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.reviewer.schemas import CritiqueReport

console = Console()


def render_critique_report(report: CritiqueReport) -> None:
    """
    Render complete AI Reviewer & Auto-Critique Report to terminal.

    Args:
        report: CritiqueReport object.
    """
    # ── Header Panel ──────────────────────────────────────────────────
    health_style_map = {
        "healthy": "bold green",
        "warning": "bold yellow",
        "critical": "bold red",
    }
    h_style = health_style_map.get(report.diagnosis.overall_health, "bold white")

    console.print(
        Panel(
            f"[bold cyan]AI Reviewer & Auto-Critique Loop[/bold cyan]\n"
            f"[yellow]Run ID:[/yellow]         {report.run_id}\n"
            f"[yellow]Target Model:[/yellow]   {report.initial_winner_name}\n"
            f"[yellow]Model Health:[/yellow]   [{h_style}]{report.diagnosis.overall_health.upper()}[/{h_style}]\n"
            f"[yellow]Primary Metric:[/yellow] {report.primary_metric.upper()}",
            title="[bold blue]atlas review[/bold blue]",
            border_style="cyan",
        )
    )

    # ── Diagnostics Panel ─────────────────────────────────────────────
    if report.diagnosis.issues:
        issue_lines = []
        for i in report.diagnosis.issues:
            sev_color = "red" if i.severity in ("critical", "high") else "yellow"
            issue_lines.append(
                f"  [{sev_color}]• [{i.severity.upper()}] {i.category.upper()}:[/{sev_color}] {i.description}\n"
                f"    [dim]Recommendation: {i.recommendation}[/dim]"
            )
        diag_content = "\n".join(issue_lines)
    else:
        diag_content = "  [bold green]✓ No critical overfitting, underfitting, or imbalance issues detected.[/bold green]"

    console.print(
        Panel(
            diag_content,
            title="[bold yellow]🔍 Automated Model Diagnostics[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # ── Critique & Reflection Panel ───────────────────────────────────
    plan = report.refinement_plan
    param_str = ", ".join(f"{k}={v}" for k, v in plan.refined_hyperparams.items())
    critique_content = (
        f"[bold underline]Critique Summary[/bold underline]\n"
        f"{plan.critique_summary}\n\n"
        f"[bold underline]Root Cause Analysis[/bold underline]\n"
        f"{plan.root_cause_analysis}\n\n"
        f"[bold underline]1-Pass Refinement Strategy[/bold underline]\n"
        f"{plan.proposed_adjustments}\n\n"
        f"[bold yellow]Refined Hyperparameters:[/bold yellow] [dim]{param_str}[/dim]"
    )

    console.print(
        Panel(
            critique_content,
            title="[bold cyan]🧠 AI Reflection & Refinement Strategy[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # ── Initial vs Refined Comparison Table ───────────────────────────
    comp = report.comparison
    table = Table(
        title="📊 Refinement Retry Comparison — Initial vs. Refined Model",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=True,
    )

    pm = report.primary_metric.upper()
    table.add_column("Model Iteration", style="bold white", min_width=22)
    table.add_column(f"Train {pm}", justify="right", min_width=12)
    table.add_column(f"Val {pm}", justify="right", min_width=12)
    table.add_column(f"Test {pm}", justify="right", min_width=12)
    table.add_column("Train-Val Gap", justify="right", min_width=14)

    # Initial row
    init_gap = comp.initial_train_metric - comp.initial_val_metric
    gap_color = "red" if init_gap >= 0.10 else "yellow" if init_gap >= 0.05 else "green"
    table.add_row(
        Text(f"{comp.model_name} (Initial)", style="dim"),
        f"{comp.initial_train_metric:.4f}",
        f"{comp.initial_val_metric:.4f}",
        f"{comp.initial_test_metric:.4f}",
        Text(f"{init_gap:.4f}", style=gap_color),
    )

    # Refined row
    ref_gap = comp.refined_train_metric - comp.refined_val_metric
    ref_gap_color = "green" if ref_gap < init_gap else "yellow"
    table.add_row(
        Text(f"{comp.model_name} (Refined)", style="bold bright_green"),
        f"{comp.refined_train_metric:.4f}",
        f"{comp.refined_val_metric:.4f}",
        Text(f"{comp.refined_test_metric:.4f}", style="bold bright_cyan"),
        Text(f"{ref_gap:.4f}", style=ref_gap_color),
    )

    console.print()
    console.print(table)
    console.print()

    # ── Verdict Panel ─────────────────────────────────────────────────
    verdict_style = "bright_green" if report.refinement_successful else "cyan"
    verdict_icon = "🏆" if report.refinement_successful else "ℹ"

    console.print(
        Panel(
            f"[{verdict_style}]{comp.verdict}[/{verdict_style}]\n\n"
            f"[dim]Optimization Rationale:[/dim] {report.opt_rationale}",
            title=f"[bold {verdict_style}]{verdict_icon} Refinement Verdict[/bold {verdict_style}]",
            border_style=verdict_style,
            padding=(1, 2),
        )
    )
