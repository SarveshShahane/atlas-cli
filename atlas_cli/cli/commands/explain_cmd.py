"""
atlas explain command — Model Explainability Engine entry point.

Runs SHAP analysis on the winning (or specified) model, generates feature
importance visualizations, and produces plain-English model explanations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from atlas_cli.agents.explainability.llm_narrator import generate_narrative
from atlas_cli.agents.explainability.plot_generator import generate_plots
from atlas_cli.agents.explainability.renderer import (
    render_explainability_report,
    save_explainability_json,
)
from atlas_cli.agents.explainability.shap_engine import compute_shap_explanations
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


def explain(
    experiment_id: Optional[str] = typer.Option(
        None, "--exp-id", "-e",
        help="Experiment ID (model name) to explain. Defaults to the winning model.",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r",
        help="Run ID of the experiment workflow. Defaults to the most recent run.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="LLM model string override for narrative generation.",
    ),
) -> None:
    """Run SHAP analysis and generate feature importance explanations."""

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

    # Step 1: SHAP Computation
    try:
        with console.status("[cyan]Computing SHAP explanations...[/cyan]"):
            result = compute_shap_explanations(
                effective_run_id,
                experiment_id=experiment_id,
            )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]SHAP computation failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # Step 2: Generate Plots
    try:
        with console.status("[cyan]Generating SHAP visualizations...[/cyan]"):
            summary_path, importance_path = generate_plots(result, effective_run_id)
            result.summary_plot_path = summary_path
            result.importance_plot_path = importance_path
    except Exception as exc:
        console.print(f"[bold yellow]Warning:[/bold yellow] Plot generation failed: {exc}")

    # Step 3: LLM Narrative
    try:
        with console.status("[cyan]Generating model explanation narrative...[/cyan]"):
            why_chosen, feature_insights = generate_narrative(
                result,
                model_override=model,
            )
            result.why_chosen_narrative = why_chosen
            result.feature_impact_narrative = feature_insights
    except Exception as exc:
        console.print(f"[bold yellow]Warning:[/bold yellow] Narrative generation failed: {exc}")

    # Step 4: Render to terminal
    console.rule("[bold cyan]Model Explainability Report[/bold cyan]")
    render_explainability_report(result)

    # Step 5: Save results
    output_path = save_explainability_json(result)

    console.print(Panel(
        f"[cyan]→[/cyan] Explainability: [dim]{output_path.resolve()}[/dim]\n"
        f"[cyan]→[/cyan] Run directory:  [dim]{run_dir.resolve()}[/dim]",
        title="[bold blue]📁 Artifacts Saved[/bold blue]",
        border_style="blue",
    ))

    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
