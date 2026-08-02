import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )
    return logging.getLogger("atlas_cli")

logger = setup_logger()
