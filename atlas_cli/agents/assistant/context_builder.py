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


def build_workspace_context(query: str) -> dict[str, Any]:
    """
    Build structured workspace context dictionary for the LLM.

    Args:
        query: User natural language query.

    Returns:
        Dict of workspace runs, dataset summaries, experiment results, and risks.
    """
    runs_dir = settings.workspace_dir / "runs"
    if not runs_dir.exists():
        return {"runs": [], "total_runs": 0}

    target_run_ids = extract_run_ids_from_query(query)
    all_run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    # If specific run requested, prioritize it; else include latest 5 runs
    if target_run_ids:
        selected_dirs = [d for d in all_run_dirs if d.name in target_run_ids]
    else:
        selected_dirs = all_run_dirs[:5]

    runs_data = []
    for d in selected_dirs:
        rid = d.name
        run_info: dict[str, Any] = {"run_id": rid}

        ds_summary = _load_json(d / "dataset_summary.json")
        if ds_summary:
            run_info["dataset"] = {
                "file_name": ds_summary.get("file_name"),
                "file_format": ds_summary.get("file_format"),
                "num_rows": ds_summary.get("num_rows"),
                "num_cols": ds_summary.get("num_cols"),
            }

        risks = _load_json(d / "risk_assessment.json")
        if risks:
            run_info["risks"] = risks.get("risks", [])

        plan = _load_json(d / "execution_plan.json")
        if plan:
            run_info["plan"] = {
                "goal": plan.get("reasoning"),
                "task_type": plan.get("task_type"),
                "target_column": plan.get("target_column"),
                "primary_metric": plan.get("evaluation", {}).get("primary_metric"),
            }

        comp = _load_json(d / "comparison_results.json")
        if comp:
            run_info["winner"] = comp.get("winner")
            run_info["rankings"] = comp.get("rankings", [])

        explain = _load_json(d / "explainability_results.json")
        if explain:
            run_info["top_features"] = [
                f["feature_name"] for f in explain.get("global_importances", [])[:5]
            ]
            run_info["why_chosen"] = explain.get("narrative", {}).get("why_chosen")

        runs_data.append(run_info)

    return {
        "query": query,
        "total_runs": len(all_run_dirs),
        "selected_runs": runs_data,
    }
