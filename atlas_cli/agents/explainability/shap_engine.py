"""
SHAP Computation Engine — Phase 8.

Loads a trained model artifact and test split data, auto-selects the
appropriate SHAP explainer (TreeExplainer vs KernelExplainer), and computes
global feature importances and local instance explanations.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from atlas_cli.agents.explainability.schemas import (
    ExplainabilityResult,
    FeatureImportance,
    LocalExplanation,
)
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

# Model families that support TreeExplainer
_TREE_FAMILIES = {
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
}

# KernelExplainer limits to keep runtime reasonable
_KERNEL_BG_SAMPLES = 100
_KERNEL_EXPLAIN_SAMPLES = 50

# Number of local explanations (most/least confident)
_NUM_LOCAL_EXAMPLES = 3


def _is_tree_model(estimator: Any) -> bool:
    """Check if an estimator supports SHAP TreeExplainer."""
    class_name = type(estimator).__name__
    return class_name in _TREE_FAMILIES


def _resolve_feature_names(
    run_dir: Path,
    num_features: int,
) -> list[str]:
    """
    Recover feature names from run artifacts.

    Priority:
      1. dataset_summary.json schema columns (excluding target)
      2. features_meta.json feature_names
      3. Generic Feature_0, Feature_1, ...
    """
    # Try dataset_summary.json — get original column names (exclude target)
    summary_path = run_dir / "dataset_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            columns = summary.get("schema", {}).get("columns", [])
            # Execution plan has the target column
            plan_path = run_dir / "execution_plan.json"
            target_col = None
            if plan_path.exists():
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                target_col = plan.get("target_column")

            col_names = [c["name"] for c in columns if c["name"] != target_col]

            # After feature engineering, count may differ (one-hot etc.)
            # If counts match, use original names
            if len(col_names) == num_features:
                return col_names
        except Exception as exc:
            logger.debug(f"Could not parse dataset_summary.json for feature names: {exc}")

    # Try features_meta.json
    meta_path = run_dir / "features_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            names = meta.get("feature_names", [])
            if len(names) == num_features:
                return names
        except Exception as exc:
            logger.debug(f"Could not parse features_meta.json: {exc}")

    # Fallback to generic names
    return [f"Feature_{i}" for i in range(num_features)]


def _resolve_winner_model(
    run_dir: Path,
    experiment_id: Optional[str] = None,
) -> tuple[str, str, str, dict]:
    """
    Resolve which model to explain.

    Returns:
        (model_name, library, safe_model_filename, metrics_dict)
    """
    if experiment_id:
        # Search experiment_results.json for matching model
        results_path = run_dir / "experiment_results.json"
        if results_path.exists():
            data = json.loads(results_path.read_text(encoding="utf-8"))
            for exp in data.get("experiments", []):
                exp_name_safe = exp["model_name"].lower().replace(" ", "_").replace("/", "_")
                if experiment_id.lower() in (exp["model_name"].lower(), exp_name_safe):
                    return (
                        exp["model_name"],
                        exp.get("library", "unknown"),
                        exp_name_safe,
                        exp.get("metrics", {}),
                    )
        raise ValueError(f"Experiment '{experiment_id}' not found in run results.")

    # Default: load winner from comparison_results.json
    comparison_path = run_dir / "comparison_results.json"
    if not comparison_path.exists():
        raise FileNotFoundError(
            "No comparison_results.json found. Run 'atlas compare' first, "
            "or specify --exp-id."
        )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    winner = comparison.get("winner")
    if not winner:
        raise ValueError("No winning model declared in comparison results.")

    model_name = winner["model_name"]
    library = winner["library"]
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")

    # Get full metrics from rankings
    metrics = {}
    for r in comparison.get("rankings", []):
        if r.get("is_winner"):
            metrics = r.get("test_metrics", r.get("val_metrics", {}))
            break

    return model_name, library, safe_name, metrics


def compute_shap_explanations(
    run_id: str,
    *,
    experiment_id: Optional[str] = None,
) -> ExplainabilityResult:
    """
    Compute SHAP explanations for the specified or winning model.

    Args:
        run_id: Run identifier.
        experiment_id: Optional specific model to explain.

    Returns:
        ExplainabilityResult with global and local SHAP explanations.
    """
    import shap

    run_dir = settings.workspace_dir / "runs" / run_id

    # Load execution plan metadata
    plan_path = run_dir / "execution_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"No execution_plan.json in run {run_id}.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    task_type = plan.get("task_type", "binary_classification")
    primary_metric = plan.get("evaluation", {}).get("primary_metric", "accuracy")

    # Resolve model
    model_name, library, safe_name, metrics = _resolve_winner_model(run_dir, experiment_id)

    # Load model artifact
    model_path = run_dir / "models" / f"{safe_name}.joblib"
    if not model_path.exists():
        # Try refined variant
        model_path = run_dir / "models" / f"{safe_name}_refined.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found for '{model_name}' at {model_path}")

    estimator = joblib.load(model_path)
    logger.info(f"Loaded model: {model_name} ({type(estimator).__name__})")

    # Load test data
    splits_dir = run_dir / "splits"
    X_test = np.load(splits_dir / "X_test.npy")
    y_test = np.load(splits_dir / "y_test.npy")

    num_features = X_test.shape[1]
    feature_names = _resolve_feature_names(run_dir, num_features)

    # Primary metric value
    primary_metric_value = metrics.get(primary_metric, 0.0)

    # Initialize result
    result = ExplainabilityResult(
        run_id=run_id,
        model_name=model_name,
        library=library,
        task_type=task_type,
        primary_metric=primary_metric,
        primary_metric_value=primary_metric_value,
        feature_names=feature_names,
        num_features=num_features,
        num_samples_explained=len(X_test),
    )

    # Select SHAP explainer
    if _is_tree_model(estimator):
        logger.info("Using TreeExplainer for tree-based model.")
        explainer = shap.TreeExplainer(estimator)
        result.explainer_type = "TreeExplainer"
        shap_values = explainer.shap_values(X_test)
    else:
        logger.info("Using KernelExplainer for non-tree model.")
        result.explainer_type = "KernelExplainer"

        # Subsample background data for KernelExplainer
        bg_size = min(_KERNEL_BG_SAMPLES, len(X_test))
        bg_idx = np.random.choice(len(X_test), size=bg_size, replace=False)
        background = X_test[bg_idx]

        # Subsample explanation set
        explain_size = min(_KERNEL_EXPLAIN_SAMPLES, len(X_test))
        explain_idx = np.random.choice(len(X_test), size=explain_size, replace=False)
        X_explain = X_test[explain_idx]

        # Use predict for regression, predict_proba for classification
        is_classification = task_type in {"binary_classification", "multiclass_classification"}
        if is_classification and hasattr(estimator, "predict_proba"):
            predict_fn = estimator.predict_proba
        else:
            predict_fn = estimator.predict

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_explain)
        result.num_samples_explained = explain_size

        # Remap to work with same downstream code
        X_test = X_explain
        y_test = y_test[explain_idx] if len(explain_idx) <= len(y_test) else y_test[:explain_size]

    # Handle multi-output SHAP values (classification returns list per class)
    if isinstance(shap_values, list):
        # For binary classification, use SHAP values for the positive class (index 1)
        if len(shap_values) == 2:
            shap_matrix = shap_values[1]
        else:
            # Multiclass: average absolute across classes
            shap_matrix = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_matrix = shap_values

    # Global feature importances (mean absolute SHAP)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    for rank, idx in enumerate(sorted_indices, 1):
        result.global_importances.append(
            FeatureImportance(
                rank=rank,
                feature_name=feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}",
                mean_abs_shap=float(mean_abs_shap[idx]),
            )
        )

    # Local explanations — most and least confident predictions
    is_classification = task_type in {"binary_classification", "multiclass_classification"}
    if is_classification and hasattr(estimator, "predict_proba"):
        probas = estimator.predict_proba(X_test)
        if probas.ndim == 2 and probas.shape[1] == 2:
            confidences = np.max(probas, axis=1)
        else:
            confidences = np.max(probas, axis=1)
    else:
        # For regression, use prediction magnitude as proxy
        preds = estimator.predict(X_test)
        confidences = np.abs(preds)

    # Top-N most confident
    most_confident_idx = np.argsort(confidences)[-_NUM_LOCAL_EXAMPLES:][::-1]
    # Top-N least confident
    least_confident_idx = np.argsort(confidences)[:_NUM_LOCAL_EXAMPLES]

    for idx in list(most_confident_idx) + list(least_confident_idx):
        pred = estimator.predict(X_test[idx:idx + 1])[0]
        conf = float(confidences[idx])

        # Top feature contributions for this instance
        instance_shap = shap_matrix[idx]
        top_feat_idx = np.argsort(np.abs(instance_shap))[-5:][::-1]
        contributions = []
        for fi in top_feat_idx:
            contributions.append({
                "feature": feature_names[fi] if fi < len(feature_names) else f"Feature_{fi}",
                "shap_value": float(instance_shap[fi]),
                "feature_value": float(X_test[idx, fi]),
            })

        result.local_explanations.append(
            LocalExplanation(
                instance_index=int(idx),
                predicted_class=str(pred),
                confidence=conf,
                top_contributions=contributions,
            )
        )

    # Store raw SHAP values and X_test for plot generation (transient, not serialized)
    result._shap_matrix = shap_matrix  # type: ignore[attr-defined]
    result._X_test = X_test  # type: ignore[attr-defined]

    return result
