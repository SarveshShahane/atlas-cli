"""
Reporter — serialises intelligence outputs to JSON artifacts and
renders Rich terminal tables for the atlas analyze command.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_cli.agents.dataset_intelligence.loader import FileMeta
from atlas_cli.agents.dataset_intelligence.profiler import ProfileReport
from atlas_cli.agents.dataset_intelligence.risk import RiskAssessment, Severity
from atlas_cli.agents.dataset_intelligence.schema import SchemaReport

console = Console()

SEVERITY_STYLE: dict[Severity, str] = {
    "CRITICAL": "bold red",
    "WARNING": "bold yellow",
    "INFO": "dim cyan",
}

TYPE_STYLE: dict[str, str] = {
    "numeric": "cyan",
    "categorical": "green",
    "datetime": "blue",
    "text": "magenta",
    "boolean": "yellow",
    "high_cardinality": "orange3",
    "email": "bold magenta",
    "ip_address": "bold blue",
    "url": "underline cyan",
    "uuid": "dim yellow",
    "phone_number": "bold green",
    "spatial_coords": "bold red",
    "unknown": "dim",
}


def export_json_artifacts(
    run_dir: Path,
    meta: FileMeta,
    schema: SchemaReport,
    profile: ProfileReport,
    risk: RiskAssessment,
) -> dict[str, Path]:
    """
    Write dataset_summary.json, quality_report.json, and risk_assessment.json
    to the given run directory.

    Returns:
        Mapping of artifact name -> file path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = {
        "file_name": meta.file_name,
        "file_format": meta.file_format,
        "file_size_mb": meta.file_size_mb,
        "num_rows": meta.num_rows,
        "num_cols": meta.num_cols,
        "dataset_hash": meta.dataset_hash,
        "schema": schema.to_dict(),
    }

    paths = {
        "dataset_summary": run_dir / "dataset_summary.json",
        "quality_report": run_dir / "quality_report.json",
        "risk_assessment": run_dir / "risk_assessment.json",
    }

    for key, path in paths.items():
        subpath = analysis_dir / path.name
        if key == "dataset_summary":
            content = json.dumps(dataset_summary, indent=2, default=str)
        elif key == "quality_report":
            content = json.dumps(profile.to_dict(), indent=2, default=str)
        else:
            content = json.dumps(risk.to_dict(), indent=2, default=str)

        path.write_text(content, encoding="utf-8")
        subpath.write_text(content, encoding="utf-8")

    return paths


def render_overview(meta: FileMeta, profile: ProfileReport) -> None:
    """Print a compact dataset overview panel."""
    duplicate_style = "bold yellow" if profile.duplicate_pct > 5 else "green"
    anomaly_str = ""
    if profile.multivariate_anomalies:
        an = profile.multivariate_anomalies
        an_style = "bold yellow" if an.isolation_forest_pct > 5 else "dim"
        anomaly_str = (
            f"  [bold]Anomalies:[/bold] Isolation Forest: [{an_style}]{an.isolation_forest_count} ({an.isolation_forest_pct:.1f}%)[/{an_style}]  "
            f"LOF: [{an_style}]{an.lof_count} ({an.lof_pct:.1f}%)[/{an_style}]\n"
        )

    console.print(
        Panel(
            f"[bold]File:[/bold] {meta.file_name}  "
            f"[bold]Format:[/bold] {meta.file_format}  "
            f"[bold]Size:[/bold] {meta.file_size_mb} MB\n"
            f"[bold]Rows:[/bold] {meta.num_rows:,}  "
            f"[bold]Columns:[/bold] {meta.num_cols}  "
            f"[bold]Duplicates:[/bold] [{duplicate_style}]{profile.duplicate_rows} ({profile.duplicate_pct:.1f}%)[/{duplicate_style}]\n"
            f"{anomaly_str}"
            f"[bold]Hash (MD5):[/bold] [dim]{meta.dataset_hash}[/dim]",
            title="[bold blue]📂 Dataset Overview[/bold blue]",
            border_style="blue",
        )
    )


def render_schema_table(schema: SchemaReport) -> None:
    """Render column schema classification table."""
    table = Table(
        title="🧬 Schema Classification",
        border_style="cyan",
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("Column", style="white", no_wrap=True)
    table.add_column("Inferred Type", justify="center")
    table.add_column("Dtype", style="dim", justify="center")
    table.add_column("Nulls", justify="right")
    table.add_column("Null %", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Sample Values", style="dim", overflow="fold")

    for col in schema.columns:
        style = TYPE_STYLE.get(col.inferred_type, "white")
        null_style = "bold red" if col.null_pct >= 50 else ("yellow" if col.null_pct >= 20 else "green")
        table.add_row(
            col.name,
            Text(col.inferred_type, style=style),
            col.dtype,
            str(col.null_count),
            Text(f"{col.null_pct:.1f}%", style=null_style),
            str(col.unique_count),
            ", ".join(col.sample_values[:3]),
        )
    console.print(table)


def render_profile_table(profile: ProfileReport) -> None:
    """Render per-column statistical profile table (numeric columns only)."""
    numeric_cols = [c for c in profile.columns if c.skewness is not None]
    if not numeric_cols:
        return

    table = Table(
        title="📊 Statistical Profile (Numeric Columns)",
        border_style="magenta",
        show_lines=False,
        header_style="bold magenta",
    )
    table.add_column("Column", style="white", no_wrap=True)
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("MAD", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Skewness", justify="right")
    table.add_column("Kurtosis", justify="right")
    table.add_column("Zero Var", justify="center")

    for col in numeric_cols:
        skew_style = "bold yellow" if col.skewness is not None and abs(col.skewness) > 2 else "white"
        kurt_style = "bold yellow" if col.kurtosis is not None and abs(col.kurtosis) > 5 else "white"
        zv_text = Text("YES", style="bold red") if col.zero_variance else Text("NO", style="dim green")

        table.add_row(
            col.name,
            f"{col.mean:.4f}" if col.mean is not None else "—",
            f"{col.std:.4f}" if col.std is not None else "—",
            f"{col.mad:.4f}" if col.mad is not None else "—",
            f"{col.min_val:.4f}" if col.min_val is not None else "—",
            f"{col.max_val:.4f}" if col.max_val is not None else "—",
            Text(f"{col.skewness:.3f}", style=skew_style) if col.skewness is not None else Text("—"),
            Text(f"{col.kurtosis:.3f}", style=kurt_style) if col.kurtosis is not None else Text("—"),
            zv_text,
        )
    console.print(table)


def render_outliers_table(profile: ProfileReport) -> None:
    """Render multi-method Outlier & Anomaly Detection table."""
    numeric_cols = [c for c in profile.columns if c.skewness is not None]
    if not numeric_cols:
        return

    table = Table(
        title="🔍 Outlier & Anomaly Detection Analysis",
        border_style="red",
        show_lines=False,
        header_style="bold red",
    )
    table.add_column("Column", style="white", no_wrap=True)
    table.add_column("IQR Outliers (Rule)", justify="right")
    table.add_column("Z-Score (>3.0)", justify="right")
    table.add_column("Mod Z-Score (MAD>3.5)", justify="right")
    table.add_column("IQR Outlier Bounds [Lower, Upper]", style="dim", justify="center")

    for col in numeric_cols:
        iqr_style = "bold red" if col.iqr_outliers_pct > 10.0 else ("yellow" if col.iqr_outliers_pct > 0 else "dim")
        z_style = "bold red" if col.zscore_outliers_pct > 5.0 else ("yellow" if col.zscore_outliers_pct > 0 else "dim")
        modz_style = "bold red" if col.modified_zscore_outliers_pct > 5.0 else ("yellow" if col.modified_zscore_outliers_pct > 0 else "dim")

        bounds_str = f"[{col.iqr_lower_bound:.2f}, {col.iqr_upper_bound:.2f}]" if col.iqr_lower_bound is not None else "—"

        table.add_row(
            col.name,
            Text(f"{col.iqr_outliers_count} ({col.iqr_outliers_pct:.1f}%)", style=iqr_style),
            Text(f"{col.zscore_outliers_count} ({col.zscore_outliers_pct:.1f}%)", style=z_style),
            Text(f"{col.modified_zscore_outliers_count} ({col.modified_zscore_outliers_pct:.1f}%)", style=modz_style),
            bounds_str,
        )

    console.print(table)


def render_mutual_information(profile: ProfileReport) -> None:
    """Render Mutual Information and Spearman rank correlation table."""
    if not profile.mutual_information:
        return

    table = Table(
        title="🧠 Mutual Information & Relationship with Target",
        border_style="green",
        show_lines=False,
        header_style="bold green",
    )
    table.add_column("Feature Column", style="white", no_wrap=True)
    table.add_column("Mutual Info Score", justify="right")
    table.add_column("Spearman |ρ|", justify="right")

    for mi in profile.mutual_information[:10]:
        mi_style = "bold yellow" if mi.mi_score >= 0.85 else "white"
        table.add_row(
            mi.column,
            Text(f"{mi.mi_score:.4f}", style=mi_style),
            f"{mi.spearman_corr:.4f}" if mi.spearman_corr is not None else "—",
        )
    console.print(table)


def render_vif_table(profile: ProfileReport) -> None:
    """Render Variance Inflation Factor (VIF) multicollinearity table."""
    if not profile.vif_metrics:
        return

    table = Table(
        title="⛓️ Multicollinearity Analysis (VIF)",
        border_style="yellow",
        show_lines=False,
        header_style="bold yellow",
    )
    table.add_column("Column", style="white", no_wrap=True)
    table.add_column("VIF Score", justify="right")
    table.add_column("Status", justify="center")

    for vif_item in profile.vif_metrics[:8]:
        vif_style = "bold red" if vif_item.vif >= 10.0 else "yellow"
        status = Text("HIGH REDUNDANCY", style="bold red") if vif_item.vif >= 10.0 else Text("MODERATE", style="dim yellow")
        table.add_row(
            vif_item.column,
            Text(f"{vif_item.vif:.2f}", style=vif_style),
            status,
        )
    console.print(table)


def render_risk_table(risk: RiskAssessment) -> None:
    """Render color-coded risk flags table."""
    if not risk.risks:
        console.print(Panel(
            "[bold green]✅ No data quality risks detected![/bold green]",
            title="[bold green]Risk Assessment[/bold green]",
            border_style="green",
        ))
        return

    table = Table(
        title=f"⚠️ Risk Assessment — Overall: {risk.overall_severity}",
        border_style="red" if risk.overall_severity == "CRITICAL" else "yellow",
        show_lines=True,
        header_style="bold",
    )
    table.add_column("Severity", justify="center", width=10)
    table.add_column("Category", style="bold white", width=22)
    table.add_column("Column", style="dim", width=22)
    table.add_column("Description", overflow="fold")
    table.add_column("Recommendation", overflow="fold", style="dim")

    for r in sorted(risk.risks, key=lambda x: {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[x.severity]):
        style = SEVERITY_STYLE[r.severity]
        table.add_row(
            Text(r.severity, style=style),
            r.category,
            r.column or "—",
            r.description,
            r.recommendation,
        )
    console.print(table)


def render_target_imbalance(profile: ProfileReport) -> None:
    """Print target class distribution if imbalance data exists."""
    if not profile.target_imbalance:
        return

    ti = profile.target_imbalance
    style = "bold red" if ti.is_imbalanced else "bold green"
    status = "IMBALANCED ⚠️" if ti.is_imbalanced else "BALANCED ✅"
    dist_str = "  ".join(f"{cls}: {pct}%" for cls, pct in ti.class_distribution.items())

    console.print(Panel(
        f"[bold]Target Column:[/bold] {ti.column}\n"
        f"[bold]Status:[/bold] [{style}]{status}[/{style}]\n"
        f"[bold]Imbalance Ratio:[/bold] {ti.imbalance_ratio:.2%}\n"
        f"[bold]Distribution:[/bold] {dist_str}",
        title="[bold yellow]🎯 Target Distribution[/bold yellow]",
        border_style="yellow",
    ))


def render_artifacts_summary(paths: dict[str, Path]) -> None:
    """Print artifact save locations."""
    lines = "\n".join(f"  [cyan]→[/cyan] [dim]{p.resolve()}[/dim]" for p in paths.values())
    console.print(Panel(
        lines,
        title="[bold green]✅ Artifacts Saved[/bold green]",
        border_style="green",
    ))
