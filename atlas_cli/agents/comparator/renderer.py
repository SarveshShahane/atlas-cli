"""
Rich Terminal Renderer — Phase 6.

Renders beautifully formatted comparison tables, winner panels, trade-off
analysis notes, and persists results to JSON.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.comparator.scorer import RankedExperiment
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")
console = Console()

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}

# ── Metric display names ────────────────────────────────────────────────
_METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "f1_macro": "F1 (Macro)",
    "f1_weighted": "F1 (Weighted)",
    "roc_auc": "ROC-AUC",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
}

# Metrics where lower values are better
_LOWER_IS_BETTER = {"rmse", "mae", "log_loss"}


def _fmt_metric(value: float, metric_name: str = "") -> str:
    """Format a metric value to 4 decimal places."""
    return f"{value:.4f}"


def _fmt_size(kb: float) -> str:
    """Format file size in human-readable form."""
    if kb >= 1024:
        return f"{kb / 1024:.1f} MB"
    return f"{kb:.1f} KB"


# ── Main comparison table ────────────────────────────────────────────────
def render_comparison_table(
    ranked: list[RankedExperiment],
    task_type: str,
    primary_metric: str,
) -> None:
    """
    Render a full side-by-side metric comparison table.

    Each model is a row with columns for all relevant metrics,
    training time, inference time, model size, and composite score.
    The winner row is highlighted in green.
    """
    if not ranked:
        console.print("[yellow]No experiments to compare.[/yellow]")
        return

    is_classification = task_type in _CLASSIFICATION_TASKS

    # Build the table
    table = Table(
        title="📈 Experiment Comparison — Multi-Objective Evaluation",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=True,
        title_style="bold bright_white",
        pad_edge=True,
    )

    # Fixed columns
    table.add_column("#", justify="center", width=4, style="dim")
    table.add_column("Model", style="bold white", min_width=20)
    table.add_column("Library", style="dim", min_width=12)

    # Primary metric (highlighted)
    pm_label = _METRIC_LABELS.get(primary_metric, primary_metric.upper())
    table.add_column(f"🎯 {pm_label}\n(Val)", justify="right", min_width=10)
    table.add_column(f"🎯 {pm_label}\n(Test)", justify="right", min_width=10)

    # Extra classification/regression metrics
    if is_classification:
        extra_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    else:
        extra_metrics = ["rmse", "mae", "r2"]

    # Only add columns for metrics not already shown as primary
    extra_display = [m for m in extra_metrics if m != primary_metric]
    for m in extra_display:
        label = _METRIC_LABELS.get(m, m.upper())
        table.add_column(f"{label}\n(Test)", justify="right", min_width=9)

    # Operational metrics
    table.add_column("⏱ Train\n(sec)", justify="right", min_width=8)
    table.add_column("⚡ Infer\n(ms)", justify="right", min_width=8)
    table.add_column("📦 Size", justify="right", min_width=8)
    table.add_column("⭐ Score", justify="right", min_width=8, style="bold")

    # Add rows
    for r in ranked:
        if r.error_message:
            # Failed experiment
            row = [
                str(r.rank),
                Text(r.model_name, style="dim red"),
                r.library,
                "—", "—",
            ]
            row.extend(["—"] * len(extra_display))
            row.extend([f"{r.training_time_s:.2f}", "—", "—", "—"])
            table.add_row(*row)
            continue

        # Style for winner
        name_style = "bold bright_green" if r.is_winner else "white"
        score_style = "bold bright_green" if r.is_winner else "bold yellow"

        pm_val = r.val_metrics.get(primary_metric, 0.0)
        pm_test = r.test_metrics.get(primary_metric, 0.0)

        row_values = [
            Text(f"{'🏆' if r.is_winner else ''} {r.rank}", style="bold green" if r.is_winner else "dim"),
            Text(r.model_name, style=name_style),
            r.library,
            Text(_fmt_metric(pm_val), style="cyan"),
            Text(_fmt_metric(pm_test), style="bright_cyan"),
        ]

        # Extra metrics (test set)
        for m in extra_display:
            val = r.test_metrics.get(m)
            if val is not None:
                row_values.append(_fmt_metric(val, m))
            else:
                row_values.append("—")

        # Operational
        row_values.append(f"{r.training_time_s:.2f}")
        row_values.append(f"{r.inference_time_ms:.1f}")
        row_values.append(_fmt_size(r.model_size_kb))
        row_values.append(Text(f"{r.composite_score:.4f}", style=score_style))

        table.add_row(*row_values)

    console.print()
    console.print(table)
    console.print()


# ── Winner panel ─────────────────────────────────────────────────────────
def render_winner_panel(
    ranked: list[RankedExperiment],
    primary_metric: str,
    task_type: str,
) -> None:
    """
    Render a dedicated panel for the winning model with full metric breakdown
    and score component analysis.
    """
    winners = [r for r in ranked if r.is_winner]
    if not winners:
        console.print(
            Panel(
                "[bold yellow]No winning model could be determined.[/bold yellow]\n"
                "[dim]All experiments may have failed.[/dim]",
                title="[bold yellow]⚠ No Winner[/bold yellow]",
                border_style="yellow",
            )
        )
        return

    winner = winners[0]
    pm_label = _METRIC_LABELS.get(primary_metric, primary_metric.upper())

    # Test metrics section
    test_lines = []
    for k, v in sorted(winner.test_metrics.items()):
        label = _METRIC_LABELS.get(k, k)
        test_lines.append(f"  [cyan]{label}:[/cyan] {v:.4f}")
    test_block = "\n".join(test_lines) if test_lines else "  [dim]Not available[/dim]"

    # Validation metrics section
    val_lines = []
    for k, v in sorted(winner.val_metrics.items()):
        label = _METRIC_LABELS.get(k, k)
        val_lines.append(f"  [dim]{label}: {v:.4f}[/dim]")
    val_block = "\n".join(val_lines) if val_lines else "  [dim]Not available[/dim]"

    # Score breakdown
    breakdown_lines = []
    for component, value in sorted(winner.score_breakdown.items()):
        if component == "weighted_total":
            continue
        weight_map = {
            "performance": f"{int(50)}%",
            "latency": f"{int(20)}%",
            "size": f"{int(15)}%",
            "cost": f"{int(15)}%",
        }
        w = weight_map.get(component, "?")
        bar_len = int(value * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        breakdown_lines.append(
            f"  [yellow]{component.capitalize():>12}[/yellow] ({w}) [{bar}] {value:.4f}"
        )
    breakdown_block = "\n".join(breakdown_lines)

    panel_content = (
        f"[bold bright_green]{winner.model_name}[/bold bright_green]  "
        f"([dim]{winner.library}[/dim])\n\n"
        f"[bold]Composite Score:[/bold]  [bright_green]{winner.composite_score:.4f}[/bright_green]\n"
        f"[bold]{pm_label} (Test):[/bold]  [cyan]{winner.test_metrics.get(primary_metric, 0):.4f}[/cyan]\n\n"
        f"[bold underline]Score Breakdown[/bold underline]\n{breakdown_block}\n\n"
        f"[bold underline]Test Set Metrics[/bold underline]\n{test_block}\n\n"
        f"[bold underline]Validation Metrics[/bold underline]\n{val_block}\n\n"
        f"[bold underline]Operational[/bold underline]\n"
        f"  [cyan]Training Time:[/cyan]  {winner.training_time_s:.2f}s\n"
        f"  [cyan]Inference Time:[/cyan] {winner.inference_time_ms:.1f}ms\n"
        f"  [cyan]Model Size:[/cyan]     {_fmt_size(winner.model_size_kb)}"
    )

    console.print(
        Panel(
            panel_content,
            title="[bold bright_green]🏆 Optimal Production Candidate[/bold bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        )
    )


# ── Trade-off analysis ───────────────────────────────────────────────────
def render_tradeoff_notes(
    ranked: list[RankedExperiment],
    primary_metric: str,
) -> None:
    """
    Generate and display bullet-point trade-off analysis between the top
    models, highlighting speed vs. accuracy trade-offs.
    """
    successful = [r for r in ranked if r.composite_score > 0]
    if len(successful) < 2:
        return

    pm_label = _METRIC_LABELS.get(primary_metric, primary_metric.upper())
    winner = successful[0]
    notes: list[str] = []

    for challenger in successful[1:]:
        # Metric comparison
        w_perf = winner.test_metrics.get(primary_metric, 0.0)
        c_perf = challenger.test_metrics.get(primary_metric, 0.0)
        perf_diff = w_perf - c_perf

        # Speed comparison
        w_speed = winner.inference_time_ms
        c_speed = challenger.inference_time_ms

        # Size comparison
        w_size = winner.model_size_kb
        c_size = challenger.model_size_kb

        parts = []

        if abs(perf_diff) > 0.001:
            direction = "higher" if perf_diff > 0 else "lower"
            parts.append(
                f"[bold]{winner.model_name}[/bold] has "
                f"[cyan]{abs(perf_diff):.4f}[/cyan] {direction} {pm_label} "
                f"than [bold]{challenger.model_name}[/bold]"
            )

        if c_speed > 0 and w_speed > 0:
            speed_ratio = c_speed / w_speed
            if speed_ratio > 1.2:
                parts.append(
                    f"[bold]{winner.model_name}[/bold] is "
                    f"[green]{speed_ratio:.1f}x faster[/green] at inference"
                )
            elif speed_ratio < 0.8:
                parts.append(
                    f"[bold]{challenger.model_name}[/bold] is "
                    f"[green]{1 / speed_ratio:.1f}x faster[/green] at inference"
                )

        if c_size > 0 and w_size > 0:
            size_ratio = c_size / w_size
            if size_ratio > 1.5:
                parts.append(
                    f"[bold]{winner.model_name}[/bold] is "
                    f"[green]{size_ratio:.1f}x smaller[/green] on disk"
                )
            elif size_ratio < 0.7:
                parts.append(
                    f"[bold]{challenger.model_name}[/bold] is "
                    f"[green]{1 / size_ratio:.1f}x smaller[/green] on disk"
                )

        if parts:
            notes.append("  • " + "; ".join(parts))

    if notes:
        content = "\n".join(notes)
        console.print(
            Panel(
                content,
                title="[bold yellow]⚖ Trade-off Analysis[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )


# ── JSON persistence ─────────────────────────────────────────────────────
def save_comparison_json(
    run_id: str,
    ranked: list[RankedExperiment],
    task_type: str,
    primary_metric: str,
) -> Path:
    """
    Persist ranked comparison results to comparison_results.json.

    Returns:
        Path to the saved JSON file.
    """
    run_dir = settings.workspace_dir / "runs" / run_id

    data = {
        "run_id": run_id,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "num_compared": len(ranked),
        "winner": None,
        "rankings": [],
    }

    for r in ranked:
        entry = {
            "rank": r.rank,
            "model_name": r.model_name,
            "library": r.library,
            "composite_score": r.composite_score,
            "score_breakdown": r.score_breakdown,
            "is_winner": r.is_winner,
            "val_metrics": r.val_metrics,
            "test_metrics": r.test_metrics,
            "inference_time_ms": r.inference_time_ms,
            "model_size_kb": r.model_size_kb,
            "training_time_s": r.training_time_s,
            "error": r.error_message,
        }
        data["rankings"].append(entry)

        if r.is_winner:
            data["winner"] = {
                "model_name": r.model_name,
                "library": r.library,
                "composite_score": r.composite_score,
                "primary_metric_test": r.test_metrics.get(primary_metric, 0.0),
            }

    output_path = run_dir / "comparison_results.json"
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Comparison results saved: {output_path}")

    return output_path
