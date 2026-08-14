"""
Assistant Context Builder — Natural Language Assistant.

Scans local workspace directory (.atlas_cli/runs/) and SQLite database to
construct comprehensive context for LLM question-answering.
"""
from __future__ import annotations

import json
import logging, re
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


def build_workspace_context(query: str, run_id: Optional[str] = None) -> dict[str, Any]:
    """
    Build comprehensive structured workspace context dictionary for the LLM.

    Args:
        query: User natural language query.
        run_id: Optional run ID to restrict context to a single project/run.

    Returns:
        Dict containing total workspace runs and complete details of selected runs.
    """
    runs_dir = settings.workspace_dir / "runs"
    if not runs_dir.exists():
        return {"runs": [], "total_runs": 0}

    all_run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if run_id:
        target_run_ids = [run_id]
    else:
        target_run_ids = extract_run_ids_from_query(query)

    # If specific run requested, prioritize it; else include latest 5 runs
    if target_run_ids:
        selected_dirs = [d for d in all_run_dirs if d.name in target_run_ids]
        if not selected_dirs and all_run_dirs:
            selected_dirs = [all_run_dirs[0]]
    else:
        selected_dirs = all_run_dirs[:5]

    runs_data = []
    for d in selected_dirs:
        rid = d.name
        run_info: dict[str, Any] = {"run_id": rid, "run_dir_path": str(d.resolve())}

        # 1. Dataset Intelligence & Schema
        ds_summary = _get_run_artifact(d, "analysis/dataset_summary.json", "dataset_summary.json")
        if ds_summary:
            run_info["dataset"] = {
                "file_name": ds_summary.get("file_name"),
                "file_format": ds_summary.get("file_format"),
                "file_size_mb": ds_summary.get("file_size_mb"),
                "num_rows": ds_summary.get("num_rows"),
                "num_cols": ds_summary.get("num_cols"),
                "schema_columns": ds_summary.get("schema", {}).get("columns", []),
            }

        # 2. Quality & Risk Assessment
        risks_data = _get_run_artifact(d, "analysis/risk_assessment.json", "risk_assessment.json")
        if risks_data:
            run_info["risks_summary"] = {
                "overall_severity": risks_data.get("overall_severity"),
                "risks": risks_data.get("risks", []),
            }

        # 3. Cleaning Report
        clean_data = _get_run_artifact(d, "cleaned/clean_report.json", "clean_report.json")
        cleaned_csv = d / "cleaned" / "cleaned_data.csv"
        if not cleaned_csv.exists():
            cleaned_csv = d / "cleaned_data.csv"

        if clean_data or cleaned_csv.exists():
            run_info["cleaning"] = {
                "cleaned_csv_exists": cleaned_csv.exists(),
                "cleaned_csv_path": str(cleaned_csv.resolve()) if cleaned_csv.exists() else None,
                "clean_report": clean_data,
            }

        # 4. Execution Plan
        plan = _get_run_artifact(d, "plan/execution_plan.json", "execution_plan.json")
        if plan:
            run_info["plan"] = {
                "goal_reasoning": plan.get("reasoning"),
                "task_type": plan.get("task_type"),
                "target_column": plan.get("target_column"),
                "preprocessing": plan.get("preprocessing", {}),
                "feature_engineering": plan.get("feature_engineering", {}),
                "model_candidates": plan.get("model_candidates", []),
                "evaluation": plan.get("evaluation", {}),
            }

        # 5. Feature Engineering Metadata
        feats_meta = _get_run_artifact(d, "features/features_meta.json", "features_meta.json")
        if feats_meta:
            run_info["feature_engineering_meta"] = feats_meta

        # 6. Experiments & Model Candidate Results
        exp_results = _get_run_artifact(d, "models/experiment_results.json", "experiment_results.json")
        if exp_results:
            results_list = exp_results.get("results", exp_results if isinstance(exp_results, list) else [])
            run_info["experiment_models"] = [
                {
                    "model_name": r.get("model_name"),
                    "library": r.get("library"),
                    "status": r.get("status"),
                    "metrics": r.get("metrics", {}),
                    "train_metrics": r.get("train_metrics", {}),
                    "feature_importances": r.get("feature_importances", {}),
                    "duration_seconds": r.get("duration_seconds"),
                    "hyperparams": r.get("hyperparams", {}),
                    "artifact_path": str((d / "models" / f"{r.get('model_name', '').lower().replace(' ', '_')}.joblib").resolve()),
                }
                for r in results_list if isinstance(r, dict)
            ]

        # 7. Comparison & Winner
        comp = _load_json(d / "comparison_results.json")
        if comp:
            run_info["winner"] = comp.get("winner")
            run_info["rankings"] = comp.get("rankings", [])

        # 8. AI Reviewer Critique
        reviewer = _load_json(d / "reviewer_report.json")
        if reviewer:
            run_info["reviewer_critique"] = reviewer

        # 9. Explainability & SHAP
        explain = _load_json(d / "explainability_results.json")
        if explain:
            run_info["shap_global_importances"] = explain.get("global_importances", [])
            run_info["why_winner_chosen"] = explain.get("narrative", {}).get("why_chosen")

        # 10. Executive Reports & Exports
        report_md = d / "reports" / "REPORT.md"
        if not report_md.exists():
            report_md = d / "REPORT.md"
        if report_md.exists():
            run_info["executive_report_path"] = str(report_md.resolve())

        exports_dir = d / "exports"
        if exports_dir.exists():
            run_info["exports_available"] = [f.name for f in exports_dir.iterdir() if f.is_file()]

        runs_data.append(run_info)

    return {
        "query": query,
        "total_runs": len(all_run_dirs),
        "selected_runs": runs_data,
    }
