"""
Explainability Schemas — Phase 8.

Structured result types for SHAP analysis, feature importances,
local explanations, and LLM narrative outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FeatureImportance:
    """Global feature importance entry."""

    rank: int
    feature_name: str
    mean_abs_shap: float
    direction: str = "+"
    signed_mean_shap: float = 0.0


@dataclass
class LocalExplanation:
    """Local SHAP explanation for a single instance."""

    instance_index: int
    predicted_class: str
    confidence: float
    top_contributions: list[dict[str, float]] = field(default_factory=list)
    """Each dict has keys: feature, shap_value, feature_value."""


@dataclass
class ExplainabilityResult:
    """Full output of the explainability engine."""

    run_id: str
    model_name: str
    library: str
    task_type: str
    primary_metric: str
    primary_metric_value: float

    # Global SHAP
    global_importances: list[FeatureImportance] = field(default_factory=list)

    # Local explanations
    local_explanations: list[LocalExplanation] = field(default_factory=list)
    target_row_explanation: Optional[LocalExplanation] = None

    # Interactions
    top_interaction_pairs: list[dict[str, Any]] = field(default_factory=list)

    # Plot paths
    summary_plot_path: Optional[str] = None
    importance_plot_path: Optional[str] = None

    # LLM narrative
    why_chosen_narrative: str = ""
    feature_impact_narrative: str = ""

    # Feature names
    feature_names: list[str] = field(default_factory=list)

    # Raw SHAP metadata
    num_features: int = 0
    num_samples_explained: int = 0
    explainer_type: str = ""
