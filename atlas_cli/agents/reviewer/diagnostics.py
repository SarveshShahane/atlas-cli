"""
Rule-Based Model Diagnostic Engine — Phase 7.

Inspects train vs. validation metrics, class imbalance metrics, training time,
and model complexity to detect overfitting, underfitting, or imbalance bias.
Generates candidate regularized hyperparameters for the refinement retry run.
"""
from __future__ import annotations

import logging
from typing import Any

from atlas_cli.agents.reviewer.schemas import (
    DiagnosticIssue,
    DiagnosisResult,
    IssueCategory,
    Severity,
)

logger = logging.getLogger("atlas_cli")

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}


def diagnose_model(
    model_name: str,
    library: str,
    *,
    task_type: str,
    primary_metric: str,
    val_metrics: dict[str, float],
    train_metrics: dict[str, float],
    current_hyperparams: dict[str, Any] | None = None,
) -> DiagnosisResult:
    """
    Perform automated diagnostic audit on a trained model.

    Args:
        model_name: Human-readable model candidate name.
        library: Model library identifier (e.g. 'sklearn.ensemble.RandomForestClassifier').
        task_type: ML task type string.
        primary_metric: Primary evaluation metric name.
        val_metrics: Validation set metric values.
        train_metrics: Training set metric values.
        current_hyperparams: Current model hyperparameters.

    Returns:
        DiagnosisResult containing detected issues and suggested parameters.
    """
    issues: list[DiagnosticIssue] = []
    suggested_params: dict[str, Any] = dict(current_hyperparams or {})
    is_classification = task_type in _CLASSIFICATION_TASKS

    train_primary = train_metrics.get(primary_metric, val_metrics.get(primary_metric, 0.0))
    val_primary = val_metrics.get(primary_metric, 0.0)

    # 1. Overfitting Detection
    if is_classification:
        gap = train_primary - val_primary
        if gap >= 0.15:
            issues.append(
                DiagnosticIssue(
                    category="overfitting",
                    severity="critical" if gap >= 0.25 else "high",
                    description=(
                        f"Severe overfitting detected: Training {primary_metric.upper()} ({train_primary:.4f}) "
                        f"is {gap:.4f} higher than Validation ({val_primary:.4f})."
                    ),
                    metric_gap=round(gap, 4),
                    recommendation="Increase regularization parameters, restrict tree depth, or require more samples per leaf.",
                )
            )
        elif gap >= 0.08:
            issues.append(
                DiagnosticIssue(
                    category="overfitting",
                    severity="medium",
                    description=(
                        f"Moderate overfitting detected: Training {primary_metric.upper()} ({train_primary:.4f}) "
                        f"exceeds Validation ({val_primary:.4f}) by {gap:.4f}."
                    ),
                    metric_gap=round(gap, 4),
                    recommendation="Apply mild regularization and feature subsampling.",
                )
            )
    else:
        # Regression (for RMSE/MAE, lower is better; for R2, higher is better)
        if primary_metric == "r2":
            gap = train_primary - val_primary
            if gap >= 0.15:
                issues.append(
                    DiagnosticIssue(
                        category="overfitting",
                        severity="high",
                        description=(
                            f"Regression overfitting: Train R² ({train_primary:.4f}) "
                            f"exceeds Val R² ({val_primary:.4f}) by {gap:.4f}."
                        ),
                        metric_gap=round(gap, 4),
                        recommendation="Add L2/L1 penalty or decrease model complexity.",
                    )
                )
        elif primary_metric in ("rmse", "mae"):
            train_val_ratio = val_primary / (train_primary + 1e-6)
            if train_val_ratio >= 1.4:
                issues.append(
                    DiagnosticIssue(
                        category="overfitting",
                        severity="high",
                        description=(
                            f"Regression overfitting: Val {primary_metric.upper()} ({val_primary:.4f}) "
                            f"is {train_val_ratio:.2f}x higher than Train ({train_primary:.4f})."
                        ),
                        metric_gap=round(val_primary - train_primary, 4),
                        recommendation="Increase regularization to prevent memorizing noise.",
                    )
                )

    # 2. Underfitting Detection
    if is_classification and val_primary < 0.65 and train_primary < 0.70:
        issues.append(
            DiagnosticIssue(
                category="underfitting",
                severity="high",
                description=(
                    f"Underfitting detected: Both Training ({train_primary:.4f}) and "
                    f"Validation ({val_primary:.4f}) {primary_metric.upper()} are low."
                ),
                recommendation="Increase model capacity, relax tree depth limits, or engineer richer features.",
            )
        )
    elif not is_classification and primary_metric == "r2" and val_primary < 0.30:
        issues.append(
            DiagnosticIssue(
                category="underfitting",
                severity="high",
                description=f"Low predictive capacity: Val R² is only {val_primary:.4f}.",
                recommendation="Use non-linear ensemble models with higher capacity.",
            )
        )

    # 3. Class Imbalance Disparity Detection
    if is_classification:
        acc = val_metrics.get("accuracy", 0.0)
        f1_m = val_metrics.get("f1_macro", val_metrics.get("f1", 0.0))
        if acc > 0.80 and (acc - f1_m) >= 0.15:
            issues.append(
                DiagnosticIssue(
                    category="class_imbalance",
                    severity="medium",
                    description=(
                        f"Class imbalance impact: Accuracy ({acc:.4f}) is much higher than "
                        f"F1 Macro ({f1_m:.4f}), indicating minority class neglect."
                    ),
                    recommendation="Enable class weight balancing or adjust classification decision threshold.",
                )
            )

    # 4. Generate Suggested Hyperparameter Regularizations
    has_overfitting = any(i.category == "overfitting" for i in issues)
    has_underfitting = any(i.category == "underfitting" for i in issues)

    lib_lower = library.lower()

    if "randomforest" in lib_lower or "extratrees" in lib_lower:
        if has_overfitting:
            suggested_params["max_depth"] = 8
            suggested_params["min_samples_split"] = 6
            suggested_params["min_samples_leaf"] = 3
            suggested_params["max_features"] = "sqrt"
        elif has_underfitting:
            suggested_params["n_estimators"] = 200
            suggested_params["max_depth"] = None
            suggested_params["min_samples_split"] = 2

    elif "xgboost" in lib_lower:
        if has_overfitting:
            suggested_params["max_depth"] = 4
            suggested_params["min_child_weight"] = 5
            suggested_params["subsample"] = 0.8
            suggested_params["colsample_bytree"] = 0.8
            suggested_params["reg_alpha"] = 0.5
            suggested_params["reg_lambda"] = 1.0
        elif has_underfitting:
            suggested_params["max_depth"] = 7
            suggested_params["learning_rate"] = 0.1

    elif "lightgbm" in lib_lower:
        if has_overfitting:
            suggested_params["num_leaves"] = 20
            suggested_params["min_child_samples"] = 20
            suggested_params["subsample"] = 0.8
            suggested_params["colsample_bytree"] = 0.8
            suggested_params["reg_alpha"] = 0.5
        elif has_underfitting:
            suggested_params["num_leaves"] = 63
            suggested_params["learning_rate"] = 0.1

    elif "catboost" in lib_lower:
        if has_overfitting:
            suggested_params["depth"] = 4
            suggested_params["l2_leaf_reg"] = 5.0
        elif has_underfitting:
            suggested_params["depth"] = 8
            suggested_params["iterations"] = 500

    elif "logisticregression" in lib_lower:
        if has_overfitting:
            suggested_params["C"] = 0.1
        elif has_underfitting:
            suggested_params["C"] = 10.0

    elif "ridge" in lib_lower:
        if has_overfitting:
            suggested_params["alpha"] = 10.0
        elif has_underfitting:
            suggested_params["alpha"] = 0.1

    # Determine overall health
    if any(i.severity == "critical" for i in issues):
        overall_health = "critical"
    elif issues:
        overall_health = "warning"
    else:
        overall_health = "healthy"

    return DiagnosisResult(
        model_name=model_name,
        has_issues=bool(issues),
        issues=issues,
        overall_health=overall_health,
        suggested_params=suggested_params,
    )
