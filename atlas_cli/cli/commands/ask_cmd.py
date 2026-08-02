import typer
from rich.panel import Panel
from atlas_cli.core.logger import console

app = typer.Typer(help="Ask natural language questions about runs, experiments, data, and models.")

@app.callback(invoke_without_command=True)
def ask(
    query: str = typer.Argument(..., help="Natural language query or question about the workspace")
):
    """Query workspace metadata, dataset risks, and experiment results using AI reasoning."""
    console.print(Panel(
        f"[bold cyan]Autonomous Data Science Assistant (Stub)[/bold cyan]\n"
        f"[yellow]User Query:[/yellow] {query}\n"
        f"[dim]Atlas CLI Assistant will query local SQLite database and analyze workspace context to answer questions.[/dim]",
        title="[bold blue]atlas ask[/bold blue]",
        border_style="cyan"
    ))
