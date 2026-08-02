import typer
from typing import Optional
from rich.panel import Panel
from atlas_cli.core.logger import console


def deploy(
    experiment_id: Optional[str] = typer.Option(None, "--exp-id", "-e", help="Winning Experiment ID to package"),
    output_dir: str = typer.Option("./deploy", "--out", "-o", help="Target microservice scaffold directory"),
) -> None:
    """Package selected model into production FastAPI inference endpoint directory."""
    console.print(Panel(
        f"[bold cyan]Deployment Scaffold Generator (Stub)[/bold cyan]\n"
        f"[yellow]Experiment ID:[/yellow] {experiment_id or 'Winning Model'}\n"
        f"[yellow]Scaffold Target:[/yellow] {output_dir}\n"
        f"[dim]Module 10 Deployment Generator will emit main.py, schemas.py, Dockerfile, requirements.txt, and test_api.py.[/dim]",
        title="[bold blue]atlas deploy[/bold blue]",
        border_style="green"
    ))
