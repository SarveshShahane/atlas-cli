"""
LLM Narrative Generator — Phase 8.

Uses the LLM to produce plain-English justification for model selection
and business-level feature impact insights. Falls back to template-based
narratives if the LLM call fails.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from atlas_cli.agents.explainability.schemas import ExplainabilityResult
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

_NARRATIVE_PROMPT = """You are an expert data scientist writing a model explainability report for a non-technical business audience.

## Model Details
- Model: {model_name}
- Library: {library}
- Task: {task_type}
- Primary Metric ({primary_metric}): {primary_metric_value:.4f}
- SHAP Explainer Used: {explainer_type}

## Top Feature Importances (by mean |SHAP value|)
{feature_table}

## Instructions
Write two sections in plain English:

1. **Why This Model Was Chosen**: Explain in 2-3 sentences why this model is the recommended production candidate, referencing its performance and characteristics. Write for a business stakeholder, not a data scientist.

2. **Feature Impact Insights**: For each of the top 5 features, write 1-2 sentences explaining what the feature represents, why it matters for predictions, and any business implications. Use clear, jargon-free language.

Respond ONLY with a valid JSON object:
{{
  "why_chosen": "<2-3 sentence justification>",
  "feature_insights": "<Feature impact narrative with all top 5 features>"
}}
"""


def generate_narrative(
    result: ExplainabilityResult,
    *,
    model_override: Optional[str] = None,
) -> tuple[str, str]:
    """
    Generate LLM-powered narrative explanations.

    Args:
        result: ExplainabilityResult with global importances.
        model_override: Optional LLM model string override.

    Returns:
        Tuple of (why_chosen_narrative, feature_impact_narrative).
    """
    top_features = result.global_importances[:10]
    feature_table = "\n".join(
        f"  {f.rank}. {f.feature_name}: {f.mean_abs_shap:.4f}"
        for f in top_features
    )

    prompt = _NARRATIVE_PROMPT.format(
        model_name=result.model_name,
        library=result.library,
        task_type=result.task_type,
        primary_metric=result.primary_metric,
        primary_metric_value=result.primary_metric_value,
        explainer_type=result.explainer_type,
        feature_table=feature_table,
    )

    llm_model = model_override or settings.llm_model

    try:
        from atlas_cli.agents.pipeline_planner.llm_client import call as llm_call

        logger.info(f"Calling LLM ({llm_model}) for explainability narrative...")
        raw_response = llm_call(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            temperature=0.3,
            max_tokens=1500,
        )

        # Parse JSON from response
        content = raw_response.strip()
        # Handle markdown code fence wrapping
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        data = json.loads(content)
        why_chosen = data.get("why_chosen", "")
        feature_insights = data.get("feature_insights", "")

        if why_chosen and feature_insights:
            return why_chosen, feature_insights
        logger.warning("LLM response missing required fields; using fallback.")

    except Exception as exc:
        logger.warning(f"LLM narrative generation failed ({exc}); using template fallback.")

    return _generate_fallback_narrative(result)


def _generate_fallback_narrative(result: ExplainabilityResult) -> tuple[str, str]:
    """Generate template-based narratives when LLM is unavailable."""

    top_features = result.global_importances[:5]
    top_name = top_features[0].feature_name if top_features else "unknown"
    top_shap = top_features[0].mean_abs_shap if top_features else 0.0

    why_chosen = (
        f"The {result.model_name} was selected as the optimal production candidate "
        f"based on its {result.primary_metric} score of {result.primary_metric_value:.4f}. "
        f"This model demonstrated the best balance between predictive accuracy, inference speed, "
        f"model size, and training efficiency across all evaluated candidates."
    )

    feature_lines = []
    for f in top_features:
        feature_lines.append(
            f"• {f.feature_name} (importance: {f.mean_abs_shap:.4f}): "
            f"This feature ranked #{f.rank} in overall impact on model predictions. "
            f"Higher or lower values of this feature significantly influence the model's output."
        )

    feature_insights = (
        f"The most influential feature is '{top_name}' with a mean absolute SHAP value of "
        f"{top_shap:.4f}, indicating it has the strongest impact on predictions.\n\n"
        + "\n".join(feature_lines)
    )

    return why_chosen, feature_insights
