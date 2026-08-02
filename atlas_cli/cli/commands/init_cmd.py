import typer
from rich.panel import Panel
from atlas_cli.core.logger import console
from atlas_cli.core.config import settings
from atlas_cli.db.session import create_db_and_tables

app = typer.Typer(help="Initialize Atlas CLI local workspace and database schema.")

@app.callback(invoke_without_command=True)
def init():
    """Initialize local Atlas CLI SQLite database and artifact directory."""
    try:
        create_db_and_tables()
        console.print(Panel.fit(
            f"[bold green]Atlas CLI Workspace Initialized Successfully![/bold green]\n"
            f"[cyan]Database Location:[/cyan] {settings.db_path.resolve()}\n"
            f"[cyan]Status:[/cyan] Ready for execution.",
            title="[bold blue]Initialization[/bold blue]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Failed to initialize Atlas CLI workspace:[/bold red] {e}")
        raise typer.Exit(code=1)
