import typer
from typing import Optional
from rich.panel import Panel
from atlas_cli.core.logger import console


def report(
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run ID of the experiment workflow"),
    output_dir: str = typer.Option("./reports", "--out", "-o", help="Output directory for generated reports"),
) -> None:
    """Compile dataset summaries, model metrics, charts, and explanations into REPORT.md / REPORT.html."""
    console.print(Panel(
        f"[bold cyan]Automated Executive Report Generator (Stub)[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow] {run_id or 'Latest Run'}\n"
        f"[yellow]Output Directory:[/yellow] {output_dir}\n"
        f"[dim]Module 8 Report Generator will export comprehensive Markdown and Jinja2-rendered HTML reports.[/dim]",
        title="[bold blue]atlas report[/bold blue]",
        border_style="yellow"
    ))
