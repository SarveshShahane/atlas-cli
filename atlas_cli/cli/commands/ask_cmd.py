import typer
from rich.panel import Panel
from atlas_cli.core.logger import console


def ask(
    query: str = typer.Argument(..., help="Natural language query about the workspace"),
) -> None:
    """Query workspace metadata, dataset risks, and experiment results using AI reasoning."""
    console.print(Panel(
        f"[bold cyan]Autonomous Data Science Assistant (Stub)[/bold cyan]\n"
        f"[yellow]User Query:[/yellow] {query}\n"
        f"[dim]Atlas CLI Assistant will query local SQLite database and analyze workspace context to answer questions.[/dim]",
        title="[bold blue]atlas ask[/bold blue]",
        border_style="cyan"
    ))
