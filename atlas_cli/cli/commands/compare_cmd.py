import typer
from typing import Optional
from rich.panel import Panel
from atlas_cli.core.logger import console


def compare(
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run ID of experiments to evaluate"),
) -> None:
    """Render comparison metrics and declare optimal production candidate."""
    console.print(Panel(
        f"[bold cyan]Experiment Comparator & Evaluator (Stub)[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow] {run_id or 'Latest Run'}\n"
        f"[dim]Module 5 Comparator will rank models across accuracy, precision, recall, training time, latency, and memory footprint.[/dim]",
        title="[bold blue]atlas compare[/bold blue]",
        border_style="green"
    ))
