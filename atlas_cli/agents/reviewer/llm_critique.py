"""
LLM Reflection & Critique Client — Phase 7.

Translates model diagnostic findings, training vs validation metric gaps,
and dataset characteristics into a structured RefinementPlan. Uses LiteLLM
with automatic heuristic fallback.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from atlas_cli.agents.reviewer.diagnostics import DiagnosisResult
from atlas_cli.agents.reviewer.schemas import RefinementPlan
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

_CRITIQUE_PROMPT_TEMPLATE = """You are an expert Lead Machine Learning Engineer reviewing model training results.

## Task & Dataset Info
- Task Type: {task_type}
- Primary Metric: {primary_metric}

## Target Model
- Model Name: {model_name}
- Library: {library}

## Performance Breakdown
- Training {primary_metric}: {train_metric:.4f}
- Validation {primary_metric}: {val_metric:.4f}
- Metric Gap (Train - Val): {metric_gap:.4f}

## Rule-Based Diagnostic Findings
Overall Health: {overall_health}
Detected Issues:
{issues_text}

## Goal
Critique this model run and propose an optimized, regularized, or tuned set of hyperparameters for a 1-pass retry experiment to reduce overfitting or improve generalization.

Respond ONLY with a valid JSON object matching this schema:
{{
  "target_model_name": "{model_name}",
  "target_library": "{library}",
  "critique_summary": "<2-sentence critique of model performance>",
  "root_cause_analysis": "<1-2 sentence root cause explanation>",
  "proposed_adjustments": "<summary of hyperparameter adjustments>",
  "refined_hyperparams": {suggested_json}
}}
"""


def generate_critique(
    model_name: str,
    library: str,
    *,
    task_type: str,
    primary_metric: str,
    val_metrics: dict[str, float],
    train_metrics: dict[str, float],
    diagnosis: DiagnosisResult,
    model_override: str | None = None,
) -> RefinementPlan:
    """
    Generate an AI reflection critique and hyperparameter refinement plan.

    Args:
        model_name: Name of target candidate model.
        library: Model library identifier.
        task_type: ML task type string.
        primary_metric: Primary evaluation metric.
        val_metrics: Validation metrics dict.
        train_metrics: Training metrics dict.
        diagnosis: DiagnosisResult from rule-based engine.
        model_override: Optional LLM model override.

    Returns:
        RefinementPlan containing critique notes and refined hyperparams.
    """
    train_metric = train_metrics.get(primary_metric, val_metrics.get(primary_metric, 0.0))
    val_metric = val_metrics.get(primary_metric, 0.0)
    metric_gap = train_metric - val_metric

    # Fallback plan generation (used when LLM key is absent or call fails)
    def _make_fallback_plan(reason: str = "Rule-based heuristic fallback") -> RefinementPlan:
        issues_desc = "; ".join(i.description for i in diagnosis.issues) or "Model performance evaluated."
        has_overfit = any(i.category == "overfitting" for i in diagnosis.issues)

        if has_overfit:
            critique = f"Model {model_name} exhibits overfitting with a {metric_gap:.4f} gap between train and validation {primary_metric}."
            root_cause = "Model capacity is too unconstrained, allowing it to fit training sample noise."
            proposed = f"Applied regularization hyperparameters ({diagnosis.suggested_params}) to constrain model complexity."
        else:
            critique = f"Model {model_name} performed with validation {primary_metric} = {val_metric:.4f}."
            root_cause = "Model hyperparameters were baseline defaults."
            proposed = "Adjusted hyperparameter configuration for refined trial."

        return RefinementPlan(
            target_model_name=model_name,
            target_library=library,
            critique_summary=f"{critique} [{reason}]",
            root_cause_analysis=root_cause,
            proposed_adjustments=proposed,
            refined_hyperparams=diagnosis.suggested_params,
        )

    # Prepare prompt
    issues_text = "\n".join(
        f"- [{i.severity.upper()}] {i.category}: {i.description} -> {i.recommendation}"
        for i in diagnosis.issues
    ) or "- No critical issues detected."

    suggested_json = json.dumps(diagnosis.suggested_params, indent=2)

    prompt = _CRITIQUE_PROMPT_TEMPLATE.format(
        task_type=task_type,
        primary_metric=primary_metric,
        model_name=model_name,
        library=library,
        train_metric=train_metric,
        val_metric=val_metric,
        metric_gap=metric_gap,
        overall_health=diagnosis.overall_health,
        issues_text=issues_text,
        suggested_json=suggested_json,
    )

    llm_model = model_override or settings.llm_model

    try:
        from atlas_cli.agents.pipeline_planner.llm_client import call as llm_call

        logger.info(f"Calling LLM ({llm_model}) for AI Reviewer reflection...")
        content = llm_call(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            temperature=0.2,
            max_tokens=1000,
        )
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            data = json.loads(content)
        return RefinementPlan.model_validate(data)

    except Exception as exc:
        logger.warning(f"LLM critique call failed ({exc}); using rule-based diagnostic plan.")
        return _make_fallback_plan(reason=f"Diagnostic fallback: {type(exc).__name__}")
