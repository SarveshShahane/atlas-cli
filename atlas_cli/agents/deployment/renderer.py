"""
Rich Terminal Renderer — Phase 10 Deployment Scaffold Generator.

Renders scaffolded file tree structures, API endpoint documentation,
and microservice quickstart instructions.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from atlas_cli.agents.deployment.scaffolder import ScaffoldResult

logger = logging.getLogger("atlas_cli")
console = Console()


def render_deployment_summary(result: ScaffoldResult) -> None:
    """Render full deployment scaffolding summary to terminal."""

    # Header Panel
    console.print(Panel(
        f"[bold cyan]FastAPI Deployment Scaffold Generator[/bold cyan]\n"
        f"[yellow]Run ID:[/yellow]         {result.run_id}\n"
        f"[yellow]Target Model:[/yellow]   [bold bright_green]{result.model_name}[/bold bright_green]\n"
        f"[yellow]Library:[/yellow]        {result.library}\n"
        f"[yellow]Task Type:[/yellow]      {result.task_type}\n"
        f"[yellow]Target Dir:[/yellow]     {result.output_dir.resolve()}",
        title="[bold blue]atlas deploy[/bold blue]",
        border_style="cyan",
    ))

    # File Tree Table
    table = Table(
        title="📁 Scaffolded Microservice Directory",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=False,
        title_style="bold bright_white",
    )
    table.add_column("File / Directory", style="bold white", min_width=25)
    table.add_column("Description", style="dim", min_width=45)

    descriptions = {
        "main.py": "FastAPI application with /health, /info, /predict, /predict_batch",
        "schemas.py": "Pydantic request & response models matching dataset features",
        "requirements.txt": "Production microservice dependency requirements",
        "Dockerfile": "Production multi-stage slim container build configuration",
        "docker-compose.yml": "Container service orchestration manifest",
        "test_api.py": "Automated verification test script",
        "model/model.joblib": "Serialized model weights artifact",
        "model/pipeline.joblib": "Serialized feature preprocessing pipeline",
    }

    for f in result.scaffolded_files:
        desc = descriptions.get(f, "Microservice component")
        icon = "📄" if "." in Path(f).name else "📁"
        table.add_row(f"{icon} {f}", desc)

    console.print()
    console.print(table)
    console.print()

    # Endpoints Panel
    endpoints_text = "\n".join(
        f"  [bright_green]{url.split()[0]:<6}[/bright_green] [cyan]{url.split()[1]}[/cyan]"
        for url in result.endpoint_urls
    )
    console.print(Panel(
        f"[bold]Exposed REST Endpoints:[/bold]\n\n{endpoints_text}\n\n"
        f"[dim]Interactive Swagger UI will be available at: http://localhost:8000/docs[/dim]",
        title="[bold yellow]🔌 API Interface[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))

    # Quickstart Panel
    rel_out = result.output_dir
    console.print(Panel(
        f"[bold]Option 1: Run with Uvicorn (Local)[/bold]\n"
        f"  [cyan]cd {rel_out} && uvicorn main:app --reload --port 8000[/cyan]\n\n"
        f"[bold]Option 2: Run with Docker Compose[/bold]\n"
        f"  [cyan]cd {rel_out} && docker compose up --build -d[/cyan]\n\n"
        f"[bold]Option 3: Test Microservice[/bold]\n"
        f"  [cyan]cd {rel_out} && python test_api.py[/cyan]",
        title="[bold bright_green]🚀 Quickstart Deployment Instructions[/bold bright_green]",
        border_style="bright_green",
        padding=(1, 2),
    ))
