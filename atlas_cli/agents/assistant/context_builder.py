"""
Assistant Context Builder — Natural Language Assistant.

Scans local workspace directory (.atlas_cli/runs/) to construct a concise,
information-dense context dictionary for LLM question-answering.
Optimized to keep token footprint low (~1000 tokens) while preserving critical
dataset profiling, model rankings, SHAP feature importances, and AI reviewer critique.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")


def _load_json(path: Path) -> Optional[dict]:
    """Safely load JSON artifact."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_run_ids_from_query(query: str) -> list[str]:
    """Extract candidate run IDs mentioned in the user query."""
    runs_dir = settings.workspace_dir / "runs"
    if not runs_dir.exists():
        return []

    existing_ids = [d.name for d in runs_dir.iterdir() if d.is_dir()]
    found = []
    for rid in existing_ids:
        if rid.lower() in query.lower():
            found.append(rid)
    return found


def _get_run_artifact(d: Path, subpath: str, root_filename: str) -> Optional[dict]:
    """Helper to load artifact JSON from subfolder or root fallback."""
    p_sub = d / subpath
    if p_sub.exists():
        return _load_json(p_sub)
    p_root = d / root_filename
    if p_root.exists():
        return _load_json(p_root)
    return None


def _normalize_feature_item(item: dict) -> dict[str, Any]:
    """Normalize feature importance dictionary from various schema formats."""
    name = (
        item.get("feature_name")
        or item.get("feature")
        or item.get("name")
        or item.get("col")
        or "unknown"
    )
    val = (
        item.get("mean_abs_shap")
        if item.get("mean_abs_shap") is not None
        else (item.get("importance") if item.get("importance") is not None else item.get("score", 0.0))
    )
    try:
        val = float(val)
    except (TypeError, ValueError):
        val = 0.0
    return {"feature": str(name), "importance": val}


def build_workspace_context(query: str, run_id: Optional[str] = None) -> dict[str, Any]:
    """
    Build a concise, high-signal structured workspace context for the LLM.

    Args:
        query: User natural language query.
        run_id: Optional run ID to restrict context to a single project/run.

    Returns:
        Dict containing total workspace runs and focused details of the target run.
    """
    runs_dir = settings.workspace_dir / "runs"
    if not runs_dir.exists():
        return {"runs": [], "total_runs": 0}

    all_run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime if d.exists() else 0,
        reverse=True,
    )
    if not all_run_dirs:
        return {"runs": [], "total_runs": 0}

    if run_id:
        target_run_ids = [run_id]
    else:
        target_run_ids = extract_run_ids_from_query(query)

    if target_run_ids:
        selected_dirs = [d for d in all_run_dirs if d.name in target_run_ids]
        if not selected_dirs:
            selected_dirs = [all_run_dirs[0]]
    else:
        # Focus primarily on the latest active run to keep token count well within LLM limits
        selected_dirs = [all_run_dirs[0]]

    runs_data = []
    for d in selected_dirs:
        rid = d.name
        run_info: dict[str, Any] = {"run_id": rid}

        # 1. Dataset Summary & Schema
        ds_summary = _get_run_artifact(d, "analysis/dataset_summary.json", "dataset_summary.json")
        if ds_summary:
            schema_cols = ds_summary.get("schema", {}).get("columns", [])
            col_list = [
                {"name": c.get("name"), "type": c.get("inferred_type") or c.get("dtype")}
                for c in schema_cols if isinstance(c, dict)
            ]
            run_info["dataset"] = {
                "file_name": ds_summary.get("file_name") or ds_summary.get("dataset_name"),
                "num_rows": ds_summary.get("num_rows"),
                "num_cols": ds_summary.get("num_cols"),
                "target_column": ds_summary.get("target") or ds_summary.get("target_column"),
                "columns": col_list,
            }

        # 2. Quality & Risk Assessment & Profiling
        quality = _get_run_artifact(d, "analysis/quality_report.json", "quality_report.json") or {}
        risks_data = _get_run_artifact(d, "analysis/risk_assessment.json", "risk_assessment.json") or {}
        
        risks_list = risks_data.get("risks", quality.get("risks", []))
        compact_risks = [
            {
                "category": r.get("category"),
                "severity": r.get("severity"),
                "column": r.get("column"),
                "description": r.get("description"),
            }
            for r in risks_list if isinstance(r, dict)
        ]
        
        raw_mi = quality.get("mutual_information", [])
        norm_mi = [_normalize_feature_item(m) for m in raw_mi if isinstance(m, dict)]
        norm_mi.sort(key=lambda x: x["importance"], reverse=True)

        run_info["data_quality"] = {
            "overall_severity": risks_data.get("overall_severity") or quality.get("overall_severity", "LOW"),
            "duplicate_rows": quality.get("duplicate_rows", 0),
            "risks": compact_risks,
            "mutual_information": norm_mi,
            "vif_metrics": quality.get("vif_metrics", []),
            "high_correlations": quality.get("high_correlations", []),
        }

        # 3. Execution Plan
        plan = _get_run_artifact(d, "plan/execution_plan.json", "execution_plan.json")
        if plan:
            candidates = [
                {"name": m.get("name"), "library": m.get("library"), "priority": m.get("priority")}
                for m in plan.get("model_candidates", []) if isinstance(m, dict)
            ]
            run_info["plan"] = {
                "task_type": plan.get("task_type"),
                "target_column": plan.get("target_column"),
                "reasoning": plan.get("reasoning"),
                "model_candidates": candidates,
            }

        # 4. Comparison & Model Rankings
        comp = _load_json(d / "comparison_results.json") or _load_json(d / "models" / "comparison_results.json")
        exp_results = _get_run_artifact(d, "models/experiment_results.json", "experiment_results.json") or {}
        
        if comp:
            winner = comp.get("winner") or exp_results.get("winner") or {}
            winner_name = winner.get("model_name") if isinstance(winner, dict) else str(winner)
            rankings = [
                {
                    "rank": rk.get("rank"),
                    "model_name": rk.get("model_name"),
                    "test_metric": rk.get("primary_metric_test") or rk.get("test_score"),
                    "composite_score": rk.get("composite_score"),
                }
                for rk in comp.get("rankings", []) if isinstance(rk, dict)
            ]
            run_info["model_results"] = {
                "winner_model": winner_name,
                "primary_metric": comp.get("primary_metric") or exp_results.get("primary_metric"),
                "winner_score": winner.get("mean_cv_score") if isinstance(winner, dict) else None,
                "rankings": rankings,
                "feature_consensus": comp.get("feature_consensus", []),
            }

        # 5. Explainability & SHAP Feature Importances
        explain = _load_json(d / "explainability_results.json")
        if explain:
            raw_importances = explain.get("global_importances", [])
            norm_importances = [_normalize_feature_item(f) for f in raw_importances if isinstance(f, dict)]
            norm_importances.sort(key=lambda x: x["importance"], reverse=True)

            run_info["explainability"] = {
                "model_name": explain.get("model_name"),
                "global_feature_importances": norm_importances,
                "feature_impact_narrative": explain.get("feature_impact_narrative") or explain.get("narrative", {}).get("impact") or explain.get("narrative", {}).get("feature_impact"),
                "why_chosen_narrative": explain.get("why_chosen_narrative") or explain.get("narrative", {}).get("why_chosen"),
            }

        # 6. AI Reviewer Critique
        reviewer = _load_json(d / "critique_report.json") or _load_json(d / "reviewer_report.json")
        if reviewer:
            run_info["ai_reviewer"] = {
                "diagnosis": reviewer.get("diagnosis") or reviewer.get("critique_summary"),
                "actions_applied": reviewer.get("actions", []),
                "refined_model": reviewer.get("refined_model_name") or reviewer.get("refined_model"),
                "before_metrics": reviewer.get("before_metrics", {}),
                "after_metrics": reviewer.get("after_metrics", {}),
            }

        runs_data.append(run_info)

    return {
        "query": query,
        "total_workspace_runs": len(all_run_dirs),
        "selected_runs": runs_data,
    }
