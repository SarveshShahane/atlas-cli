"""
atlas ask command — Natural Language Assistant entry point.

Queries workspace metadata, dataset risks, model candidates, and experiment
results using LLM reasoning and workspace inspection.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from atlas_cli.agents.assistant.reasoning import answer_query

console = Console()


def ask(
    query: str = typer.Argument(..., help="Natural language query about the workspace"),
) -> None:
    """Query workspace metadata, dataset risks, and experiment results using AI reasoning."""

    try:
        with console.status("[cyan]Analyzing workspace context and generating answer...[/cyan]"):
            answer = answer_query(query)
    except Exception as exc:
        console.print(Panel(
            f"[bold red]Assistant query failed:[/bold red] {exc}",
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    console.print(Panel(
        f"[yellow]User Query:[/yellow] [bold]{query}[/bold]\n\n"
        f"{answer}",
        title="[bold blue]atlas ask — AI Assistant[/bold blue]",
        border_style="cyan",
        padding=(1, 2),
    ))
