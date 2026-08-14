"""
atlas export command — Dataset Export Engine entry point.
Exports cleaned, feature-engineered, and train/test split datasets into CSV or Parquet format.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from atlas_cli.agents.dataset_intelligence.cleaner import clean_dataset
from atlas_cli.agents.experimentation.splitter import split_data
from atlas_cli.agents.feature_engineering.runner import process_feature_engineering
from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan
from atlas_cli.core.config import settings

console = Console()


def _find_latest_run_id() -> Optional[str]:
    """Find the most recently created run directory."""
    runs_dir = settings.workspace_dir / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0].name if run_dirs else None


def _format_file_size(path: Path) -> str:
    """Format byte size into human readable string."""
    if not path.exists():
        return "0 B"
    size_bytes = path.stat().st_size
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def _write_df_to_format(df: pd.DataFrame, dest_path: Path, output_format: str, run_dir: Optional[Path] = None) -> Path:
    """Write DataFrame in specified output format (CSV or Parquet) and mirror to run_dir/exports/."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if output_format.lower() == "parquet" else ".csv"
    
    try:
        out_file = dest_path.with_suffix(ext)
        if ext == ".parquet":
            df.to_parquet(out_file, index=False)
        else:
            df.to_csv(out_file, index=False)
    except ImportError:
        console.print("[dim yellow]Note: 'pyarrow' is not installed for Parquet export. Falling back to CSV.[/dim yellow]")
        ext = ".csv"
        out_file = dest_path.with_suffix(ext)
        df.to_csv(out_file, index=False)

    if run_dir:
        run_exports = run_dir / "exports"
        run_exports.mkdir(parents=True, exist_ok=True)
        mirror_file = (run_exports / dest_path.name).with_suffix(ext)
        if ext == ".parquet":
            try:
                df.to_parquet(mirror_file, index=False)
            except Exception:
                df.to_csv(mirror_file.with_suffix(".csv"), index=False)
        else:
            df.to_csv(mirror_file, index=False)

    return out_file


def export(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r",
        help="Run ID to export datasets from. Defaults to the most recent run.",
    ),
    dataset_type: str = typer.Option(
        "all", "--type", "-t",
        help="Dataset type to export: 'cleaned', 'fe' (feature engineered), 'splits' (train/val/test), or 'all'.",
    ),
    output_dir: Path = typer.Option(
        Path("./exports"), "--output-dir", "-o",
        help="Target directory where exported dataset files will be saved.",
    ),
    output_format: str = typer.Option(
        "csv", "--format", "-f",
        help="Output file format: 'csv' or 'parquet'.",
    ),
) -> None:
    """
    Export cleaned, feature engineered, and train/test splitted datasets for a run.

    Examples:
        atlas export
        atlas export --run-id 0c80fc43 --type fe --format parquet
        atlas export --type splits -o ./my_splits
    """
    effective_run_id = run_id or _find_latest_run_id()
    if not effective_run_id:
        console.print(Panel(
            "[bold red]No runs found.[/bold red]\n"
            "[dim]Run 'atlas analyze', 'atlas plan', or 'atlas clean' first.[/dim]",
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    run_dir = settings.workspace_dir / "runs" / effective_run_id
    if not run_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Run directory not found: {run_dir}")
        raise typer.Exit(code=1)

    norm_type = dataset_type.lower().replace("_", "").replace("-", "")
    if norm_type not in {"all", "cleaned", "fe", "featureengineered", "splits", "traintest"}:
        console.print(f"[bold red]Error:[/bold red] Invalid dataset type '{dataset_type}'. Choose from: cleaned, fe, splits, all.")
        raise typer.Exit(code=1)

    output_format_norm = output_format.lower()
    if output_format_norm not in {"csv", "parquet"}:
        console.print(f"[bold red]Error:[/bold red] Invalid format '{output_format}'. Choose 'csv' or 'parquet'.")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    exported_items = []

    # 1. Cleaned Dataset Export
    if norm_type in {"all", "cleaned"}:
        cleaned_csv = run_dir / "cleaned_data.csv"
        if not cleaned_csv.exists():
            summary_path = run_dir / "dataset_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                dataset_path = Path(summary.get("file_name", ""))
                if not dataset_path.exists():
                    dataset_path = Path.cwd() / dataset_path.name
                if dataset_path.exists():
                    with console.status("[cyan]Generating cleaned dataset export...[/cyan]"):
                        clean_dataset(dataset_path, output_dir=run_dir)

        if cleaned_csv.exists():
            df_cleaned = pd.read_csv(cleaned_csv)
            dest_file = output_dir / "cleaned_dataset"
            final_path = _write_df_to_format(df_cleaned, dest_file, output_format_norm, run_dir=run_dir)
            exported_items.append({
                "type": "Cleaned Dataset",
                "filename": final_path.name,
                "shape": f"{len(df_cleaned):,} rows × {len(df_cleaned.columns)} cols",
                "size": _format_file_size(final_path),
                "path": str(final_path.resolve()),
            })

    # 2. Feature Engineered Dataset Export
    if norm_type in {"all", "fe", "featureengineered"}:
        fe_csv = run_dir / "feature_engineered_data.csv"
        if not fe_csv.exists() and (run_dir / "execution_plan.json").exists():
            with console.status("[cyan]Generating feature engineered dataset export...[/cyan]"):
                try:
                    process_feature_engineering(effective_run_id)
                except Exception as exc:
                    console.print(f"[dim yellow]Warning: Could not compute feature engineering: {exc}[/dim yellow]")

        if fe_csv.exists():
            df_fe = pd.read_csv(fe_csv)
            dest_file = output_dir / "feature_engineered_dataset"
            final_path = _write_df_to_format(df_fe, dest_file, output_format_norm, run_dir=run_dir)
            exported_items.append({
                "type": "Feature Engineered Dataset",
                "filename": final_path.name,
                "shape": f"{len(df_fe):,} rows × {len(df_fe.columns)} cols",
                "size": _format_file_size(final_path),
                "path": str(final_path.resolve()),
            })

    # 3. Train / Validation / Test Split Dataset Export
    if norm_type in {"all", "splits", "traintest"}:
        splits_dir = run_dir / "splits"
        splits_found = False

        if splits_dir.exists():
            for split_name in ["train", "val", "test"]:
                split_csv = splits_dir / f"{split_name}.csv"
                if split_csv.exists():
                    splits_found = True
                    df_split = pd.read_csv(split_csv)
                    dest_file = output_dir / f"{split_name}_split"
                    final_path = _write_df_to_format(df_split, dest_file, output_format_norm, run_dir=run_dir)
                    exported_items.append({
                        "type": f"{split_name.capitalize()} Split",
                        "filename": final_path.name,
                        "shape": f"{len(df_split):,} rows × {len(df_split.columns)} cols",
                        "size": _format_file_size(final_path),
                        "path": str(final_path.resolve()),
                    })

        if not splits_found and (run_dir / "execution_plan.json").exists():
            with console.status("[cyan]Generating train/test split dataset exports...[/cyan]"):
                try:
                    X_train, X_val, X_test, y_train, y_val, y_test, _, feature_names, *_ = process_feature_engineering(effective_run_id)
                    plan = ExecutionPlan.model_validate_json((run_dir / "execution_plan.json").read_text(encoding="utf-8"))
                    target_name = plan.target_column or "target"
                    for split_name, X_split, y_split in [
                        ("train", X_train, y_train),
                        ("val", X_val, y_val),
                        ("test", X_test, y_test),
                    ]:
                        X_split_dense = X_split.toarray() if hasattr(X_split, "toarray") else X_split
                        df_split = pd.DataFrame(data=X_split_dense, columns=feature_names)
                        df_split[target_name] = y_split

                        dest_file = output_dir / f"{split_name}_split"
                        final_path = _write_df_to_format(df_split, dest_file, output_format_norm, run_dir=run_dir)
                        exported_items.append({
                            "type": f"{split_name.capitalize()} Split",
                            "filename": final_path.name,
                            "shape": f"{len(df_split):,} rows × {len(df_split.columns)} cols",
                            "size": _format_file_size(final_path),
                            "path": str(final_path.resolve()),
                        })
                except Exception as exc:
                    console.print(f"[dim yellow]Warning: Could not compute dataset splits: {exc}[/dim yellow]")

    if not exported_items:
        console.print(Panel(
            f"[bold yellow]No datasets found to export for Run ID '{effective_run_id}'.[/bold yellow]\n"
            f"[dim]Run 'atlas clean' or 'atlas experiment' to generate dataset outputs.[/dim]",
            title="[bold yellow]⚠️ No Datasets Exported[/bold yellow]",
            border_style="yellow",
        ))
        return

    console.rule("[bold cyan]Dataset Export Summary[/bold cyan]")
    table = Table(title="📦 Exported Datasets", border_style="cyan", header_style="bold cyan", show_lines=True)
    table.add_column("Dataset Type", style="bold white")
    table.add_column("File Name", style="green")
    table.add_column("Dimensions", justify="right", style="cyan")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Destination Path", style="dim")

    for item in exported_items:
        table.add_row(item["type"], item["filename"], item["shape"], item["size"], item["path"])

    console.print(table)
    console.print(Panel(
        f"[bold green]Successfully exported {len(exported_items)} dataset(s)[/bold green]\n"
        f"[dim]Target Directory:[/dim] [white]{output_dir.resolve()}[/white]",
        title="[bold green]✅ Export Complete[/bold green]",
        border_style="green",
    ))
    console.rule(f"[dim]Run ID: {effective_run_id}[/dim]")
