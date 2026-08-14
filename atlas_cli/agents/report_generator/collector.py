"""
Report Data Collector — Phase 9.

Aggregates all JSON artifacts from a run directory into a unified context
dictionary for Jinja2 template rendering. Handles missing artifacts gracefully.
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")


def _load_json(path: Path) -> Optional[dict]:
    """Safely load a JSON file, returning None if missing or malformed."""
    if not path.exists():
        logger.debug(f"Artifact not found (skipping): {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to parse {path.name}: {exc}")
        return None


def _encode_image_base64(path: Path) -> Optional[str]:
    """Read a PNG file and return a base64-encoded data URI."""
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as exc:
        logger.warning(f"Failed to encode image {path.name}: {exc}")
        return None


def collect_report_data(run_id: str) -> dict[str, Any]:
    """
    Collect all available run artifacts into a unified report context.

    Args:
        run_id: Run identifier.

    Returns:
        Dictionary with all report sections populated from JSON artifacts.

    Raises:
        FileNotFoundError: If the run directory does not exist.
    """
    run_dir = settings.workspace_dir / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ctx: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "has_dataset": False,
        "has_quality": False,
        "has_risks": False,
        "has_plan": False,
        "has_experiments": False,
        "has_comparison": False,
        "has_critique": False,
        "has_explainability": False,
        "has_plots": False,
    }

    # ── Dataset Summary ──────────────────────────────────────────────────
    dataset = _load_json(run_dir / "dataset_summary.json")
    if dataset:
        ctx["has_dataset"] = True
        ctx["dataset"] = {
            "file_name": dataset.get("file_name", "Unknown"),
            "file_format": dataset.get("file_format", "Unknown"),
            "file_size_mb": dataset.get("file_size_mb", 0),
            "num_rows": dataset.get("num_rows", 0),
            "num_cols": dataset.get("num_cols", 0),
            "dataset_hash": dataset.get("dataset_hash", ""),
            "columns": dataset.get("schema", {}).get("columns", []),
        }

    # ── Quality Report ───────────────────────────────────────────────────
    quality = _load_json(run_dir / "quality_report.json")
    if quality:
        sanitized_cols = []
        for col in quality.get("columns", []):
            c = dict(col)
            c["missing_pct"] = c.get("missing_pct") or 0.0
            c["skewness"] = c.get("skewness") if c.get("skewness") is not None else 0.0
            c["mean"] = c.get("mean") if c.get("mean") is not None else 0.0
            c["std"] = c.get("std") if c.get("std") is not None else 0.0
            sanitized_cols.append(c)

        ctx["has_quality"] = True
        ctx["quality"] = {
            "num_rows": quality.get("num_rows", 0),
            "num_cols": quality.get("num_cols", 0),
            "duplicate_rows": quality.get("duplicate_rows", 0),
            "duplicate_pct": quality.get("duplicate_pct", 0),
            "columns": sanitized_cols,
            "high_correlations": quality.get("high_correlations", []),
            "target_imbalance": quality.get("target_imbalance"),
        }

    # ── Risk Assessment ──────────────────────────────────────────────────
    risks = _load_json(run_dir / "risk_assessment.json")
    if risks:
        ctx["has_risks"] = True
        ctx["risks"] = {
            "overall_severity": risks.get("overall_severity", "UNKNOWN"),
            "total_risks": risks.get("total_risks", 0),
            "items_list": risks.get("risks", []),
        }

    # ── Execution Plan ───────────────────────────────────────────────────
    plan = _load_json(run_dir / "execution_plan.json")
    if plan:
        ctx["has_plan"] = True
        ctx["plan"] = {
            "task_type": plan.get("task_type", "Unknown"),
            "target_column": plan.get("target_column", "Unknown"),
            "reasoning": plan.get("reasoning", ""),
            "preprocessing": plan.get("preprocessing", {}),
            "feature_engineering": plan.get("feature_engineering", {}),
            "model_candidates": plan.get("model_candidates", []),
            "evaluation": plan.get("evaluation", {}),
        }

    # ── Experiment Results ───────────────────────────────────────────────
    experiments = _load_json(run_dir / "experiment_results.json")
    if experiments:
        ctx["has_experiments"] = True
        ctx["experiments"] = {
            "task_type": experiments.get("task_type", "Unknown"),
            "total_duration": experiments.get("total_duration_seconds", 0),
            "num_candidates": experiments.get("num_candidates", 0),
            "num_succeeded": experiments.get("num_succeeded", 0),
            "num_failed": experiments.get("num_failed", 0),
            "primary_metric": experiments.get("primary_metric", "accuracy"),
            "items_list": experiments.get("experiments", []),
        }

    # ── Comparison Results ───────────────────────────────────────────────
    comparison = _load_json(run_dir / "comparison_results.json")
    if comparison:
        ctx["has_comparison"] = True
        ctx["comparison"] = {
            "task_type": comparison.get("task_type", "Unknown"),
            "primary_metric": comparison.get("primary_metric", "accuracy"),
            "num_compared": comparison.get("num_compared", 0),
            "winner": comparison.get("winner"),
            "rankings": comparison.get("rankings", []),
        }

    # ── AI Reviewer Critique ─────────────────────────────────────────────
    critique = _load_json(run_dir / "critique_report.json")
    if critique:
        ctx["has_critique"] = True
        ctx["critique"] = {
            "task_type": critique.get("task_type", "Unknown"),
            "primary_metric": critique.get("primary_metric", "accuracy"),
            "initial_winner": critique.get("initial_winner_name", "Unknown"),
            "diagnosis": critique.get("diagnosis", {}),
            "refinement_plan": critique.get("refinement_plan", {}),
            "comparison": critique.get("comparison", {}),
            "refinement_successful": critique.get("refinement_successful", False),
            "opt_rationale": critique.get("opt_rationale", ""),
        }

    # ── Explainability Results ───────────────────────────────────────────
    explain = _load_json(run_dir / "explainability_results.json")
    if explain:
        ctx["has_explainability"] = True
        ctx["explainability"] = {
            "model_name": explain.get("model_name", "Unknown"),
            "library": explain.get("library", ""),
            "explainer_type": explain.get("explainer_type", ""),
            "num_features": explain.get("num_features", 0),
            "num_samples": explain.get("num_samples_explained", 0),
            "global_importances": explain.get("global_importances", []),
            "local_explanations": explain.get("local_explanations", []),
            "narrative": explain.get("narrative", {}),
        }

    # ── Plot Images ──────────────────────────────────────────────────────
    plots_dir = run_dir / "plots"
    summary_plot = plots_dir / "shap_summary.png"
    importance_plot = plots_dir / "shap_feature_importance.png"

    if summary_plot.exists() or importance_plot.exists():
        ctx["has_plots"] = True
        ctx["plots"] = {
            "summary_path": str(summary_plot) if summary_plot.exists() else None,
            "importance_path": str(importance_plot) if importance_plot.exists() else None,
            "summary_b64": _encode_image_base64(summary_plot),
            "importance_b64": _encode_image_base64(importance_plot),
        }

    return ctx
