"""
Assistant Reasoning Engine — Natural Language Assistant.

Combines workspace context with LLM reasoning to answer natural language
queries about runs, dataset quality risks, model candidates, and performance metrics.
Includes a local heuristic QA fallback engine when LLM is offline.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from atlas_cli.agents.assistant.context_builder import build_workspace_context
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

_ASSISTANT_PROMPT = """You are Atlas, an expert Autonomous Data Science Assistant.
Answer the user's natural language question accurately and concisely using ONLY the provided workspace context for the specified dataset/run. Do NOT mention or infer details about other datasets.

## User Question
{query}

## Workspace Context
{context_json}

## Instructions
- Provide a direct, clear, professional answer strictly relevant to the given dataset in context.
- Highlight key model names, metrics, dataset names, column names, or risks when applicable.
- Keep the response concise and well-structured (2-4 paragraphs maximum).
"""


_CODE_GEN_PROMPT = """You are Atlas, an expert Autonomous Data Science Assistant.
The user wants executable Python code to accomplish a task based on the workspace context.

## User Question
{query}

## Workspace Context
{context_json}

## Instructions
- Provide ready-to-run Python code inside ```python ... ``` code fences.
- Include necessary imports (pandas, joblib, sklearn, etc.) and reference model paths like `.atlas_cli/runs/<run_id>/models/<winner>.joblib`.
- Include concise markdown explanation above the code block.
"""


def answer_query(
    query: str,
    run_id: Optional[str] = None,
    generate_code: bool = False,
) -> str:
    """
    Answer a user query about workspace metadata, dataset risks, or experiments.

    Args:
        query: Natural language question.
        run_id: Optional run ID to restrict context to a single project/run.
        generate_code: Whether to instruct LLM to emit Python code snippet.

    Returns:
        String answer.
    """
    context = build_workspace_context(query, run_id=run_id)
    context_json = json.dumps(context, indent=2)

    # Check if an API key is configured before calling LLM
    from atlas_cli.agents.pipeline_planner.llm_client import _active_provider
    has_provider = False
    try:
        has_provider = _active_provider() is not None
    except Exception:
        has_provider = False

    if has_provider:
        try:
            from atlas_cli.agents.pipeline_planner.llm_client import call as llm_call

            logger.info(f"Calling LLM ({settings.llm_model}) for natural language assistant...")
            template = _CODE_GEN_PROMPT if generate_code else _ASSISTANT_PROMPT
            prompt = template.format(
                query=query,
                context_json=context_json,
            )
            response = llm_call(
                messages=[{"role": "user", "content": prompt}],
                model=settings.llm_model,
                temperature=0.2,
                max_tokens=1200,
            )
            if response and response.strip():
                return response.strip()
        except Exception as exc:
            logger.debug(f"LLM call failed for assistant ({exc}); running local heuristic QA engine...")

    # Fallback to local heuristic QA engine
    return _heuristic_qa_fallback(query, context)


def _heuristic_qa_fallback(query: str, context: dict[str, Any]) -> str:
    """Provide accurate, deterministic responses for common workspace queries without LLM."""
    q_lower = query.lower()
    runs = context.get("selected_runs", [])

    if not runs:
        return "No runs or dataset analyses found in the workspace. Run 'atlas analyze <file>' or 'atlas plan <file> --goal <goal>' first."

    # Best model or metrics query
    if "metric" in q_lower or "algorithm" in q_lower or "eval" in q_lower or "score" in q_lower:
        for r in runs:
            plan = r.get("plan", {})
            winner = r.get("winner")
            rankings = r.get("rankings", [])
            ds_name = r.get("dataset", {}).get("file_name", "dataset")
            pm = plan.get("primary_metric", "Primary Metric")
            if rankings:
                lines = [f"📊 **Evaluation Metrics for {ds_name} (Run {r['run_id']}):**\n", f"• **Primary Metric:** `{pm}`\n"]
                for rk in rankings:
                    lines.append(f"• **{rk.get('model_name')}:** Test {pm} = {rk.get('primary_metric_test', 0.0):.4f} (Composite Score: {rk.get('composite_score', 0.0):.4f})")
                return "\n".join(lines)
            elif winner:
                return (
                    f"📊 **Evaluation Metrics for {ds_name} (Run {r['run_id']}):**\n\n"
                    f"• **Primary Metric:** `{pm}`\n"
                    f"• **Top Model ({winner.get('model_name')}):** Test {pm} = {winner.get('primary_metric_test', 0.0):.4f}"
                )

    if "best" in q_lower or "winner" in q_lower or "top" in q_lower or "performing" in q_lower:
        for r in runs:
            winner = r.get("winner")
            if winner:
                pm = winner.get("primary_metric_test", 0.0)
                return (
                    f"🏆 **Best Performing Model for Run {r['run_id']}:**\n\n"
                    f"• **Model:** {winner.get('model_name')}\n"
                    f"• **Library:** `{winner.get('library')}`\n"
                    f"• **Composite Score:** {winner.get('composite_score'):.4f}\n"
                    f"• **Test {r.get('plan', {}).get('primary_metric', 'Metric')}:** {pm:.4f}\n\n"
                    f"*(Ranked #1 across all evaluated candidates in multi-objective evaluation)*"
                )

    # Missing values / data leakage / risks query
    if "risk" in q_lower or "missing" in q_lower or "leakage" in q_lower or "quality" in q_lower or "outlier" in q_lower:
        for r in runs:
            risks = r.get("risks", [])
            ds = r.get("dataset", {})
            ds_name = ds.get("file_name", "dataset")

            if risks:
                lines = [f"⚠️ **Data Quality & Risk Summary for {ds_name} (Run {r['run_id']}):**\n"]
                for rsk in risks:
                    col_str = f" on column '{rsk['column']}'" if rsk.get("column") else ""
                    lines.append(
                        f"• **[{rsk.get('severity', 'WARN')}] {rsk.get('category')}**{col_str}: "
                        f"{rsk.get('description')} → *{rsk.get('recommendation')}*"
                    )
                return "\n".join(lines)
            else:
                return f"✅ No critical missing values, data leakage risks, or quality issues detected in **{ds_name}**."

    # General workspace summary
    r0 = runs[0]
    winner_str = r0.get("winner", {}).get("model_name", "N/A")
    return (
        f"📊 **Workspace Context Summary (Latest Run: {r0['run_id']}):**\n\n"
        f"• **Dataset:** {r0.get('dataset', {}).get('file_name', 'Unknown')}\n"
        f"• **Task:** {r0.get('plan', {}).get('task_type', 'Predictive Modeling')}\n"
        f"• **Winner Model:** {winner_str}\n"
        f"• **Total Workspace Runs:** {context.get('total_runs', 0)}\n\n"
        f"For specific metrics, run `atlas compare --run-id {r0['run_id']}` or `atlas explain --run-id {r0['run_id']}`."
    )
