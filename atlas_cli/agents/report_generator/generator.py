"""
Report Generator — Phase 9.

Loads Jinja2 templates and renders Markdown and HTML reports from the
collected report context. Copies plot images alongside the reports.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

# Template directory
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


def _get_jinja_env() -> Environment:
    """Create a Jinja2 environment pointing at the templates directory."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html.j2",), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def generate_reports(
    ctx: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Render Markdown and HTML reports from the report context.

    Args:
        ctx: Report context dictionary from collector.collect_report_data().
        output_dir: Target directory for generated reports.

    Returns:
        Dictionary mapping report type to output file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = _get_jinja_env()
    outputs: dict[str, Path] = {}

    # ── Render Markdown ──────────────────────────────────────────────────
    try:
        md_template = env.get_template("report.md.j2")
        md_content = md_template.render(**ctx)
        md_path = out / "REPORT.md"
        md_path.write_text(md_content, encoding="utf-8")
        outputs["markdown"] = md_path
        logger.info(f"Markdown report saved: {md_path}")
    except Exception as exc:
        logger.error(f"Failed to render Markdown report: {exc}")

    # ── Render HTML ──────────────────────────────────────────────────────
    try:
        html_template = env.get_template("report.html.j2")
        html_content = html_template.render(**ctx)
        html_path = out / "REPORT.html"
        html_path.write_text(html_content, encoding="utf-8")
        outputs["html"] = html_path
        logger.info(f"HTML report saved: {html_path}")
    except Exception as exc:
        logger.error(f"Failed to render HTML report: {exc}")

    # ── Copy plot images alongside markdown report ───────────────────────
    if ctx.get("has_plots"):
        plots = ctx.get("plots", {})
        for key in ("summary_path", "importance_path"):
            src = plots.get(key)
            if src and Path(src).exists():
                dst = out / Path(src).name
                try:
                    shutil.copy2(src, dst)
                    logger.info(f"Plot copied: {dst}")
                except Exception as exc:
                    logger.warning(f"Failed to copy plot {src}: {exc}")

    return outputs
