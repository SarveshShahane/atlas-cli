"""
Rich Terminal Renderer — Phase 6.

Renders comparison tables, generalization gap badges, feature consensus tables, winner panels,
and trade-off notes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.comparator.scorer import RankedExperiment
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")
console = Console()

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}

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


def _fmt_metric(value: float, metric_name: str = "") -> str:
    return f"{value:.4f}"


def _fmt_size(kb: float) -> str:
    if kb >= 1024:
        return f"{kb / 1024:.1f} MB"
    return f"{kb:.1f} KB"


def render_comparison_table(
    ranked: list[RankedExperiment],
    task_type: str,
    primary_metric: str,
) -> None:
    """Render side-by-side metric comparison table with generalization gap indicators."""
    if not ranked:
        console.print("[yellow]No experiments to compare.[/yellow]")
        return

    table = Table(
        title="📈 Experiment Comparison — Multi-Objective Evaluation",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=True,
        title_style="bold bright_white",
        pad_edge=True,
    )

    table.add_column("#", justify="center", width=4, style="dim")
    table.add_column("Model", style="bold white", min_width=18)
    table.add_column("Library", style="dim", min_width=12)

    pm_label = _METRIC_LABELS.get(primary_metric, primary_metric.upper())
    table.add_column(f"🎯 {pm_label}\n(Test)", justify="right", min_width=10)
    table.add_column("Gen Gap\n(Train-Test)", justify="center", min_width=10)

    table.add_column("⏱ Train\n(sec)", justify="right", min_width=8)
    table.add_column("⚡ Infer\n(ms)", justify="right", min_width=8)
    table.add_column("📦 Size", justify="right", min_width=8)
    table.add_column("⭐ Score", justify="right", min_width=8, style="bold")

    for r in ranked:
        if r.error_message:
            row = [
                str(r.rank),
                Text(r.model_name, style="dim red"),
                r.library,
                "—", "—",
                f"{r.training_time_s:.2f}", "—", "—", "—",
            ]
            table.add_row(*row)
            continue

        name_style = "bold bright_green" if r.is_winner else "white"
        score_style = "bold bright_green" if r.is_winner else "bold yellow"
        pm_test = r.test_metrics.get(primary_metric, r.val_metrics.get(primary_metric, 0.0))

        gap_style = "bold green" if r.overfitting_gap <= 0.05 else ("bold red" if r.overfitting_gap > 0.10 else "yellow")
        gap_text = Text(f"{r.overfitting_gap:.4f} ({r.overfitting_status})", style=gap_style)

        row_values = [
            Text(f"{'🏆' if r.is_winner else ''} {r.rank}", style="bold green" if r.is_winner else "dim"),
            Text(r.model_name, style=name_style),
            r.library,
            Text(_fmt_metric(pm_test), style="bright_cyan"),
            gap_text,
            f"{r.training_time_s:.2f}",
            f"{r.inference_time_ms:.1f}",
            _fmt_size(r.model_size_kb),
            Text(f"{r.composite_score:.4f}", style=score_style),
        ]
        table.add_row(*row_values)

    console.print()
    console.print(table)
    console.print()


def render_feature_consensus_table(experiment_entries: list[dict[str, Any]]) -> None:
    """Render cross-model feature consensus matrix."""
    fi_list = []
    for entry in experiment_entries:
        if entry.get("status") == "success" and entry.get("feature_importances"):
            fi_list.append((entry.get("model_name"), entry.get("feature_importances")))

    if not fi_list:
        return

    table = Table(
        title="🧠 Cross-Model Feature Consensus Matrix",
        border_style="magenta",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Model Candidate", style="bold white")
    table.add_column("Top Driving Feature #1", style="cyan")
    table.add_column("Top Feature #2", style="dim cyan")
    table.add_column("Top Feature #3", style="dim cyan")

    for model_name, fi_dict in fi_list:
        top_items = list(fi_dict.items())[:3]
        f1 = f"{top_items[0][0]} ({top_items[0][1]:.2f})" if len(top_items) > 0 else "—"
        f2 = f"{top_items[1][0]} ({top_items[1][1]:.2f})" if len(top_items) > 1 else "—"
        f3 = f"{top_items[2][0]} ({top_items[2][1]:.2f})" if len(top_items) > 2 else "—"
        table.add_row(model_name, f1, f2, f3)

    console.print(table)
    console.print()


def render_winner_panel(
    ranked: list[RankedExperiment],
    primary_metric: str,
    task_type: str,
) -> None:
    """Render a dedicated panel for the winning model."""
    winners = [r for r in ranked if r.is_winner]
    if not winners:
        console.print(
            Panel(
                "[bold yellow]No winning model could be determined.[/bold yellow]",
                title="[bold yellow]⚠ No Winner[/bold yellow]",
                border_style="yellow",
            )
        )
        return

    winner = winners[0]
    pm_label = _METRIC_LABELS.get(primary_metric, primary_metric.upper())

    test_lines = [f"  [cyan]{_METRIC_LABELS.get(k, k)}:[/cyan] {v:.4f}" for k, v in sorted(winner.test_metrics.items())]
    test_block = "\n".join(test_lines) if test_lines else "  [dim]Not available[/dim]"

    breakdown_lines = []
    for component, value in sorted(winner.score_breakdown.items()):
        if component == "weighted_total":
            continue
        bar_len = int(value * 15)
        bar = "█" * bar_len + "░" * (15 - bar_len)
        breakdown_lines.append(f"  [yellow]{component.capitalize():>18}[/yellow] [{bar}] {value:.4f}")
    breakdown_block = "\n".join(breakdown_lines)

    panel_content = (
        f"[bold bright_green]{winner.model_name}[/bold bright_green]  ([dim]{winner.library}[/dim])\n\n"
        f"[bold]Composite Score:[/bold]  [bright_green]{winner.composite_score:.4f}[/bright_green]\n"
        f"[bold]{pm_label} (Test):[/bold]  [cyan]{winner.test_metrics.get(primary_metric, 0):.4f}[/cyan]\n"
        f"[bold]Gen Gap (Train-Test):[/bold] [green]{winner.overfitting_gap:.4f} ({winner.overfitting_status})[/green]\n\n"
        f"[bold underline]Score Breakdown[/bold underline]\n{breakdown_block}\n\n"
        f"[bold underline]Test Set Metrics[/bold underline]\n{test_block}\n\n"
        f"[bold underline]Operational Stats[/bold underline]\n"
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


def render_tradeoff_notes(
    ranked: list[RankedExperiment],
    primary_metric: str,
) -> None:
    """Generate and display trade-off analysis between top models."""
    successful = [r for r in ranked if r.composite_score > 0]
    if len(successful) < 2:
        return

    winner = successful[0]
    notes: list[str] = []

    for challenger in successful[1:]:
        w_perf = winner.test_metrics.get(primary_metric, 0.0)
        c_perf = challenger.test_metrics.get(primary_metric, 0.0)
        perf_diff = w_perf - c_perf

        w_speed = winner.inference_time_ms
        c_speed = challenger.inference_time_ms

        parts = []
        if abs(perf_diff) > 0.001:
            direction = "higher" if perf_diff > 0 else "lower"
            parts.append(f"[bold]{winner.model_name}[/bold] has [cyan]{abs(perf_diff):.4f}[/cyan] {direction} score than [bold]{challenger.model_name}[/bold]")

        if c_speed > 0 and w_speed > 0:
            ratio = c_speed / w_speed
            if ratio > 1.2:
                parts.append(f"[bold]{winner.model_name}[/bold] is [green]{ratio:.1f}x faster[/green] at inference")

        if parts:
            notes.append("  • " + "; ".join(parts))

    if notes:
        console.print(
            Panel(
                "\n".join(notes),
                title="[bold yellow]⚖ Trade-off Analysis[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )


def save_comparison_json(
    run_id: str,
    ranked: list[RankedExperiment],
    task_type: str,
    primary_metric: str,
) -> Path:
    """Persist ranked comparison results to comparison_results.json."""
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
            "overfitting_gap": r.overfitting_gap,
            "overfitting_status": r.overfitting_status,
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
