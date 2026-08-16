from rich.panel import Panel
from rich.text import Text
from atlas_cli.core.logger import console

BANNER_TEXT = r"""
    ___  _____ _        ___   _____   _____ _     _____ 
   / _ \|_   _| |      / _ \ /  ___| /  __ \ |   |_   _|
  / /_\ \ | | | |     / /_\ \\ `--.  | /  \/ |     | |  
  |  _  | | | | |___  |  _  | `--. \ | |   | |___  | |  
  |_| |_| \_/ \_____/ |_| |_|\____/  \_/\_/\_____/ \_/  
"""

def print_banner() -> None:
    """Print stylish ASCII banner for Atlas CLI."""
    panel = Panel(
        Text(BANNER_TEXT, style="bold cyan") + Text("\n  Autonomous Data Science Operating System v0.1.0\n", style="italic white"),
        border_style="cyan",
        subtitle="-- Atlas --",
        subtitle_align="center"
    )
    console.print(panel)
