"""
Validation Metrics Calculator — Phase 5.

Computes comprehensive evaluation metrics based on task type:
  - Classification: accuracy, precision, recall, f1, roc_auc
  - Regression: rmse, mae, r2
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("atlas_cli")

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_type: str,
    estimator: Any | None = None,
    X_val: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute evaluation metrics for a single model.

    Args:
        y_true: Ground truth labels / values.
        y_pred: Model predictions.
        task_type: The ML task type string.
        estimator: Optional fitted estimator (used for predict_proba → roc_auc).
        X_val: Optional validation features (needed for predict_proba).

    Returns:
        Dictionary mapping metric names to float values.
    """
    metrics: dict[str, float] = {}

    if task_type in _CLASSIFICATION_TASKS:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        if task_type == "binary_classification":
            metrics["precision"] = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
            metrics["f1_macro"] = metrics["f1"]
            metrics["f1_weighted"] = metrics["f1"]
        else:
            for avg in ("weighted", "macro"):
                suffix = f"_{avg}"
                metrics[f"precision{suffix}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
                metrics[f"recall{suffix}"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
                metrics[f"f1{suffix}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
            metrics["precision"] = metrics["precision_weighted"]
            metrics["recall"] = metrics["recall_weighted"]
            metrics["f1"] = metrics["f1_weighted"]

        if estimator is not None and X_val is not None and hasattr(estimator, "predict_proba"):
            try:
                y_proba = estimator.predict_proba(X_val)
                unique_classes = np.unique(y_true)
                if len(unique_classes) == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    )
            except Exception as exc:
                logger.warning(f"Could not compute ROC-AUC: {exc}")
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics
