import typer
from pathlib import Path
from typing import Optional
from rich.panel import Panel
from atlas_cli.core.logger import console

app = typer.Typer(help="Generate an autonomous execution plan for preprocessing, feature engineering, and model candidates.")

@app.callback(invoke_without_command=True)
def plan(
    file_path: Path = typer.Argument(..., help="Path to the target dataset file"),
    goal: str = typer.Option("Predict target variable", "--goal", "-g", help="Objective or task description"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model identifier")
):
    """Generate pipeline strategy and candidate model plan via LLM planner."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File '{file_path}' does not exist.")
        raise typer.Exit(code=1)

    console.print(Panel(
        f"[bold cyan]Pipeline Planner Strategy (Stub)[/bold cyan]\n"
        f"[yellow]Dataset:[/yellow] {file_path.resolve()}\n"
        f"[yellow]Goal:[/yellow] {goal}\n"
        f"[yellow]LLM Model:[/yellow] {model or 'default (gpt-4o-mini)'}\n"
        f"[dim]Module 2 Planner will evaluate task type, data risks, transformations, and candidate model graph.[/dim]",
        title="[bold blue]atlas plan[/bold blue]",
        border_style="magenta"
    ))
