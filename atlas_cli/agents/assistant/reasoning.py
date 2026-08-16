"""
Assistant Reasoning Engine — Natural Language Assistant.

Combines workspace context with LLM reasoning to answer natural language
queries about runs, dataset quality risks, model candidates, and performance metrics.
Includes a comprehensive deterministic QA fallback engine when LLM is offline.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from atlas_cli.agents.assistant.context_builder import build_workspace_context
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

_ASSISTANT_PROMPT = """You are Atlas, an expert Autonomous Data Science Assistant.
Answer the user's natural language question directly, accurately, and thoroughly using the provided workspace context for the specified dataset/run.

## User Question
{query}

## Workspace Context
{context_json}

## Instructions
- Provide a direct, professional, and insightful answer grounded strictly in the workspace data.
- If asked about feature contributions or importances, reference the specific feature names and their importance/MI values from the context.
- If asked about models or performance, reference the candidate models, winner, primary metric, and composite scores.
- If asked about data health or risks, reference detected anomalies, outliers, high correlations, or VIF scores.
- Format with clear Markdown bolding, bullet points, and clean typography.
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
                max_tokens=1000,
            )
            if response and response.strip():
                return response.strip()
        except Exception as exc:
            logger.warning(f"LLM call failed for assistant ({exc}); running local deterministic QA engine...")

    # Fallback to local heuristic QA engine
    return _heuristic_qa_fallback(query, context)


def _heuristic_qa_fallback(query: str, context: dict[str, Any]) -> str:
    """Provide accurate, deterministic responses for common workspace queries without LLM."""
    q_lower = query.lower()
    runs = context.get("selected_runs", [])

    if not runs:
        return "No runs or dataset analyses found in the workspace. Run 'atlas analyze <file>' or 'atlas run <file>' first."

    r0 = runs[0]
    ds_name = r0.get("dataset", {}).get("file_name", "dataset")
    target_col = r0.get("dataset", {}).get("target_column") or r0.get("plan", {}).get("target_column", "target")
    explain = r0.get("explainability", {})
    
    from atlas_cli.agents.assistant.context_builder import _normalize_feature_item
    raw_feat_imps = explain.get("global_feature_importances", [])
    feat_imps = [_normalize_feature_item(f) for f in raw_feat_imps if isinstance(f, dict)]
    feat_imps.sort(key=lambda x: x["importance"], reverse=True)

    raw_mi = r0.get("data_quality", {}).get("mutual_information", [])
    mi_list = [_normalize_feature_item(m) for m in raw_mi if isinstance(m, dict)]
    mi_list.sort(key=lambda x: x["importance"], reverse=True)
    consensus = r0.get("model_results", {}).get("feature_consensus", [])

    # ── 1. Least / Minimum Contributing Feature ──────────────────────────────
    if any(k in q_lower for k in ("least", "less", "lowest", "bottom", "weakest", "minimum", "worst feature", "unimportant")):
        if feat_imps:
            # Sorted descending, so least is at the end
            least = feat_imps[-1]
            feat_name = least.get("feature", "unknown")
            imp_val = least.get("importance", 0.0)
            all_feats = ", ".join([f"`{f.get('feature')}` ({f.get('importance', 0):.3f})" for f in feat_imps])
            return (
                f"📉 **Least Contributing Feature in {ds_name}:**\n\n"
                f"Based on SHAP global feature attribution for the trained model:\n"
                f"• **Feature:** `{feat_name}`\n"
                f"• **Mean |SHAP Value| / Importance:** `{imp_val:.4f}`\n\n"
                f"**All Feature Importances (Ranked High to Low):**\n"
                f"{all_feats}\n\n"
                f"`{feat_name}` has the lowest predictive signal for `{target_col}` and could potentially be dropped without significantly degrading model accuracy."
            )
        elif mi_list:
            least_mi = mi_list[-1]
            return (
                f"📉 **Least Contributing Feature in {ds_name} (Mutual Information):**\n\n"
                f"• **Feature:** `{least_mi.get('feature')}`\n"
                f"• **Mutual Information Score:** `{least_mi.get('importance', 0):.4f}`\n\n"
                f"Out of all evaluated columns, `{least_mi.get('feature')}` shares the lowest mutual dependency with target `{target_col}`."
            )

    # ── 2. Most / Top Contributing Features ──────────────────────────────────
    if any(k in q_lower for k in ("most", "top feature", "highest", "important feature", "importance", "contributing", "contribution")):
        if feat_imps:
            lines = [f"📊 **Global Feature Importances for {ds_name} (Model: {explain.get('model_name', 'Winner')}):**\n"]
            for idx, f in enumerate(feat_imps, 1):
                name = f.get("feature", "feature")
                imp = f.get("importance", 0.0)
                lines.append(f"{idx}. **`{name}`**: SHAP value = `{imp:.4f}`")
            if explain.get("feature_impact_narrative"):
                lines.append(f"\n💡 **Impact Analysis:** {explain['feature_impact_narrative']}")
            return "\n".join(lines)
        elif mi_list:
            lines = [f"📊 **Top Features by Mutual Information for {ds_name}:**\n"]
            for idx, m in enumerate(mi_list, 1):
                lines.append(f"{idx}. **`{m.get('feature')}`**: MI Score = `{m.get('mi_score', 0):.4f}`")
            return "\n".join(lines)

    # ── 3. Winner / Best Model & Rankings ────────────────────────────────────
    if any(k in q_lower for k in ("best", "winner", "top model", "performing", "chosen", "why chosen", "rank", "score", "metric")):
        results = r0.get("model_results", {})
        winner = results.get("winner_model", "N/A")
        pm = results.get("primary_metric", "Metric")
        score = results.get("winner_score")
        rankings = results.get("rankings", [])
        
        lines = [
            f"🏆 **Model Comparison & Winner for {ds_name} (Run {r0['run_id']}):**\n",
            f"• **Winner Model:** **{winner}**",
            f"• **Primary Metric (`{pm}`):** `{score:.4f}`" if score is not None else f"• **Primary Metric:** `{pm}`",
        ]
        if explain.get("why_chosen_narrative"):
            lines.append(f"• **Selection Rationale:** {explain['why_chosen_narrative']}")
        if rankings:
            lines.append("\n**Candidate Rankings:**")
            for rk in rankings:
                test_s = rk.get('test_metric')
                test_str = f"Test {pm} = {test_s:.4f}" if test_s is not None else "Evaluated"
                comp_s = rk.get('composite_score')
                comp_str = f" | Composite: {comp_s:.4f}" if comp_s is not None else ""
                lines.append(f"{rk.get('rank', '•')}. **{rk.get('model_name')}**: {test_str}{comp_str}")
        return "\n".join(lines)

    # ── 4. Dataset Quality, Health, & Risks ───────────────────────────────────
    if any(k in q_lower for k in ("risk", "missing", "leakage", "quality", "outlier", "vif", "collinear", "correlation", "imbalance")):
        dq = r0.get("data_quality", {})
        risks = dq.get("risks", [])
        lines = [f"🛡️ **Data Health & Risk Assessment for {ds_name} (Run {r0['run_id']}):**\n"]
        lines.append(f"• **Overall Severity:** `{dq.get('overall_severity', 'LOW')}`")
        lines.append(f"• **Duplicate Rows:** `{dq.get('duplicate_rows', 0)}`")
        
        if risks:
            lines.append("\n**Identified Risks & Recommendations:**")
            for rsk in risks:
                col_str = f" (`{rsk['column']}`)" if rsk.get("column") else ""
                lines.append(f"• **[{rsk.get('severity', 'WARN')}] {rsk.get('category')}**{col_str}: {rsk.get('description')}")
        else:
            lines.append("• **Risks:** ✅ No critical data quality or leakage risks detected.")
        
        corrs = dq.get("high_correlations", [])
        if corrs:
            corr_strs = [
                f"`{c.get('pair', ['', ''])[0]}` ↔ `{c.get('pair', ['', ''])[1]}` ({c.get('correlation', 0):.2f})"
                for c in corrs[:3]
                if isinstance(c, dict)
            ]
            lines.append(f"\n**High Correlations:** {', '.join(corr_strs)}")
        return "\n".join(lines)

    # ── 5. AI Reviewer / Auto-Critique Loop ──────────────────────────────────
    if any(k in q_lower for k in ("review", "critique", "improve", "refine", "diagnosis")):
        rev = r0.get("ai_reviewer", {})
        if rev:
            diag = rev.get("diagnosis", "Model performance verified.")
            actions = rev.get("actions_applied", [])
            lines = [
                f"🔍 **AI Reviewer Diagnosis & Refinement:**\n",
                f"• **Diagnosis:** {diag}",
                f"• **Refined Model:** `{rev.get('refined_model', 'Optimized')}`",
            ]
            if actions:
                lines.append(f"• **Actions Applied:** {', '.join(actions)}")
            return "\n".join(lines)

    # ── 6. Default Comprehensive Overview ────────────────────────────────────
    ds = r0.get("dataset", {})
    results = r0.get("model_results", {})
    return (
        f"📊 **Workspace Context Summary (Run: `{r0['run_id']}`):**\n\n"
        f"• **Dataset:** `{ds.get('file_name', 'Unknown')}` ({ds.get('num_rows', '?')} rows, {ds.get('num_cols', '?')} columns)\n"
        f"• **Target Column:** `{ds.get('target_column') or r0.get('plan', {}).get('target_column', 'N/A')}`\n"
        f"• **Task:** `{r0.get('plan', {}).get('task_type', 'Predictive Modeling')}`\n"
        f"• **Winner Model:** **{results.get('winner_model', 'Evaluated')}**\n"
        f"• **Total Runs in Workspace:** {context.get('total_workspace_runs', 1)}\n\n"
        f"You can ask specific questions such as:\n"
        f"  - *\"What is the most contributing feature?\"*\n"
        f"  - *\"What is the least contributing feature?\"*\n"
        f"  - *\"Why was {results.get('winner_model', 'the winning model')} chosen?\"*\n"
        f"  - *\"Were there any data quality risks or outliers?\"*"
    )
