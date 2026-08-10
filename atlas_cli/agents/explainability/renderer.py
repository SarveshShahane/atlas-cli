"""
Rich Terminal Renderer — Phase 8 Explainability.

Renders feature importance tables, local explanation panels, LLM narrative,
and plot references in a beautifully formatted terminal view.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.explainability.schemas import ExplainabilityResult
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")
console = Console()


def render_explainability_report(result: ExplainabilityResult) -> None:
    """Render the full explainability report to the terminal."""
    _render_header(result)
    _render_feature_importance_table(result)
    _render_local_explanations(result)
    _render_narrative(result)
    _render_plot_references(result)


def _render_header(result: ExplainabilityResult) -> None:
    """Render the explainability header panel."""
    console.print(Panel(
        f"[bold cyan]Model Explainability Engine[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow]         {result.run_id}\n"
        f"[yellow]Model:[/yellow]          [bold]{result.model_name}[/bold]\n"
        f"[yellow]Library:[/yellow]        {result.library}\n"
        f"[yellow]Task Type:[/yellow]      {result.task_type}\n"
        f"[yellow]Explainer:[/yellow]      {result.explainer_type}\n"
        f"[yellow]{result.primary_metric}:[/yellow]  "
        f"[bright_green]{result.primary_metric_value:.4f}[/bright_green]\n"
        f"[yellow]Features:[/yellow]       {result.num_features}\n"
        f"[yellow]Samples:[/yellow]        {result.num_samples_explained}",
        title="[bold blue]atlas explain[/bold blue]",
        border_style="cyan",
    ))


def _render_feature_importance_table(result: ExplainabilityResult) -> None:
    """Render global feature importance as a ranked table with bar visualization."""
    if not result.global_importances:
        console.print("[yellow]No feature importances available.[/yellow]")
        return

    console.rule("[bold cyan]Global Feature Importance (SHAP)[/bold cyan]")

    table = Table(
        title="🔍 Feature Importance — Mean |SHAP Value|",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=True,
        title_style="bold bright_white",
        pad_edge=True,
    )

    table.add_column("#", justify="center", width=4, style="dim")
    table.add_column("Feature", style="bold white", min_width=18)
    table.add_column("Mean |SHAP|", justify="right", min_width=12)
    table.add_column("Impact", min_width=30)

    # Get max SHAP for bar scaling
    max_shap = result.global_importances[0].mean_abs_shap if result.global_importances else 1.0

    top_n = min(20, len(result.global_importances))
    for feat in result.global_importances[:top_n]:
        # Visual bar
        bar_width = 25
        filled = int((feat.mean_abs_shap / max_shap) * bar_width) if max_shap > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color based on rank
        if feat.rank <= 3:
            name_style = "bold bright_green"
            bar_style = "bright_green"
        elif feat.rank <= 7:
            name_style = "bold cyan"
            bar_style = "cyan"
        else:
            name_style = "white"
            bar_style = "dim"

        table.add_row(
            str(feat.rank),
            Text(feat.feature_name, style=name_style),
            f"{feat.mean_abs_shap:.6f}",
            Text(bar, style=bar_style),
        )

    console.print()
    console.print(table)
    console.print()


def _render_local_explanations(result: ExplainabilityResult) -> None:
    """Render local SHAP explanations for sample instances."""
    if not result.local_explanations:
        return

    console.rule("[bold cyan]Local Instance Explanations[/bold cyan]")

    # Split into most confident and least confident
    n_half = len(result.local_explanations) // 2
    most_confident = result.local_explanations[:n_half] if n_half > 0 else result.local_explanations[:3]
    least_confident = result.local_explanations[n_half:] if n_half > 0 else []

    if most_confident:
        _render_local_group(most_confident, "Most Confident Predictions", "bright_green")

    if least_confident:
        _render_local_group(least_confident, "Least Confident Predictions", "yellow")


def _render_local_group(
    explanations: list,
    title: str,
    color: str,
) -> None:
    """Render a group of local explanations."""
    panels = []
    for exp in explanations:
        lines = [
            f"[yellow]Instance:[/yellow] #{exp.instance_index}",
            f"[yellow]Prediction:[/yellow] {exp.predicted_class}",
            f"[yellow]Confidence:[/yellow] {exp.confidence:.4f}",
            "",
            "[bold]Top Feature Contributions:[/bold]",
        ]
        for contrib in exp.top_contributions:
            direction = "↑" if contrib["shap_value"] > 0 else "↓"
            shap_style = "green" if contrib["shap_value"] > 0 else "red"
            lines.append(
                f"  {direction} [{shap_style}]{contrib['shap_value']:+.4f}[/{shap_style}]  "
                f"[cyan]{contrib['feature']}[/cyan] = {contrib['feature_value']:.2f}"
            )

        panel_content = "\n".join(lines)
        panels.append(
            Panel(
                panel_content,
                border_style=color,
                width=45,
                padding=(0, 1),
            )
        )

    console.print(
        Panel(
            Columns(panels, equal=True, expand=True),
            title=f"[bold {color}]{title}[/bold {color}]",
            border_style=color,
            padding=(1, 1),
        )
    )


def _render_narrative(result: ExplainabilityResult) -> None:
    """Render LLM-generated narrative explanation."""
    if not result.why_chosen_narrative and not result.feature_impact_narrative:
        return

    console.rule("[bold cyan]Model Explanation Narrative[/bold cyan]")

    if result.why_chosen_narrative:
        console.print(Panel(
            f"[bright_white]{result.why_chosen_narrative}[/bright_white]",
            title="[bold bright_green]💡 Why This Model Was Chosen[/bold bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        ))

    if result.feature_impact_narrative:
        console.print(Panel(
            f"[bright_white]{result.feature_impact_narrative}[/bright_white]",
            title="[bold magenta]📊 Feature Impact Insights[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        ))


def _render_plot_references(result: ExplainabilityResult) -> None:
    """Render plot file references."""
    lines = []
    if result.summary_plot_path:
        lines.append(f"[cyan]→[/cyan] SHAP Summary Plot:      [dim]{result.summary_plot_path}[/dim]")
    if result.importance_plot_path:
        lines.append(f"[cyan]→[/cyan] Feature Importance Plot: [dim]{result.importance_plot_path}[/dim]")

    if lines:
        console.print(Panel(
            "\n".join(lines),
            title="[bold blue]📈 Generated Plots[/bold blue]",
            border_style="blue",
        ))


def save_explainability_json(result: ExplainabilityResult) -> Path:
    """
    Persist explainability results to JSON.

    Returns:
        Path to the saved JSON file.
    """
    run_dir = settings.workspace_dir / "runs" / result.run_id

    data = {
        "run_id": result.run_id,
        "model_name": result.model_name,
        "library": result.library,
        "task_type": result.task_type,
        "primary_metric": result.primary_metric,
        "primary_metric_value": result.primary_metric_value,
        "explainer_type": result.explainer_type,
        "num_features": result.num_features,
        "num_samples_explained": result.num_samples_explained,
        "global_importances": [
            {
                "rank": f.rank,
                "feature_name": f.feature_name,
                "mean_abs_shap": f.mean_abs_shap,
            }
            for f in result.global_importances
        ],
        "local_explanations": [
            {
                "instance_index": le.instance_index,
                "predicted_class": le.predicted_class,
                "confidence": le.confidence,
                "top_contributions": le.top_contributions,
            }
            for le in result.local_explanations
        ],
        "plots": {
            "summary_plot": result.summary_plot_path or None,
            "importance_plot": result.importance_plot_path or None,
        },
        "narrative": {
            "why_chosen": result.why_chosen_narrative,
            "feature_impact": result.feature_impact_narrative,
        },
    }

    output_path = run_dir / "explainability_results.json"
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Explainability results saved: {output_path}")

    return output_path
