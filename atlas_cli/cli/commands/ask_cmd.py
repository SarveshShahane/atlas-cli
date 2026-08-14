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


from typing import Optional


def ask(
    query: Optional[str] = typer.Argument(None, help="Natural language query about the workspace"),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Filter context to specific run ID"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Launch interactive REPL chat mode"),
    generate_code: bool = typer.Option(False, "--generate-code", "-c", help="Generate ready-to-run Python code snippet"),
) -> None:
    """Query workspace metadata, dataset risks, and experiment results using AI reasoning."""
    from rich.prompt import Prompt

    if interactive or not query:
        console.rule("[bold cyan]atlas ask — Interactive AI Assistant[/bold cyan]")
        console.print("[dim]Type your question about workspace runs, models, or risks. Type 'exit' or 'quit' to exit.[/dim]\n")
        
        while True:
            try:
                user_q = Prompt.ask("[bold cyan]atlas > [/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Exiting interactive mode.[/dim]")
                break

            if not user_q or user_q.lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting interactive mode.[/dim]")
                break

            try:
                with console.status("[cyan]Thinking...[/cyan]"):
                    answer = answer_query(user_q, run_id=run_id, generate_code=generate_code)
                console.print(Panel(
                    answer,
                    title=f"[bold blue]Question: {user_q}[/bold blue]",
                    border_style="cyan",
                    padding=(1, 2),
                ))
            except Exception as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
        return

    try:
        with console.status("[cyan]Analyzing workspace context and generating answer...[/cyan]"):
            answer = answer_query(query, run_id=run_id, generate_code=generate_code)
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
