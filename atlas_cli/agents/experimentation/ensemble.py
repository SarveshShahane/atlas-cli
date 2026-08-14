"""
Automated Ensemble Synthesizer — Phase 5.

Combines top-performing distinct candidate estimators into a Voting Ensemble
(soft-voting classifier or averaging regressor) and evaluates its performance against single models.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor

from atlas_cli.agents.experimentation.metrics import compute_metrics
from atlas_cli.agents.experimentation.worker import ExperimentResult
from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan

logger = logging.getLogger("atlas_cli")

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}


def build_and_evaluate_ensemble(
    results: list[ExperimentResult],
    plan: ExecutionPlan,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Optional[ExperimentResult]:
    """
    Build a Voting Ensemble from top successful candidate estimators.

    Args:
        results: List of completed ExperimentResult objects.
        plan: ExecutionPlan containing task type and primary metric.
        X_train: Preprocessed training feature matrix.
        y_train: Training target labels.
        X_val: Preprocessed validation feature matrix.
        y_val: Validation target labels.

    Returns:
        New ExperimentResult for the Weighted Ensemble, or None if insufficient models.
    """
    successful_results = [r for r in results if r.status == "success" and r.estimator is not None]
    if len(successful_results) < 2:
        logger.info("Fewer than 2 successful estimators; skipping ensemble synthesis.")
        return None

    primary_metric = plan.evaluation.primary_metric
    is_classification = plan.task_type in _CLASSIFICATION_TASKS

    # Sort successful models by primary metric
    reverse_sort = primary_metric not in {"rmse", "mae", "logloss"}
    sorted_candidates = sorted(
        successful_results,
        key=lambda r: r.metrics.get(primary_metric, 0.0),
        reverse=reverse_sort,
    )

    # Pick top 3 models
    top_models = sorted_candidates[:3]
    estimators = [(r.model_name.lower().replace(" ", "_"), r.estimator) for r in top_models]

    t0 = time.perf_counter()

    try:
        logger.info(f"Synthesizing Ensemble from top {len(estimators)} models: {[r.model_name for r in top_models]}")
        if is_classification:
            # Soft voting if all estimators support predict_proba, else hard voting
            has_proba = all(hasattr(e, "predict_proba") for _, e in estimators)
            voting_mode = "soft" if has_proba else "hard"
            ensemble_model = VotingClassifier(estimators=estimators, voting=voting_mode)
        else:
            ensemble_model = VotingRegressor(estimators=estimators)

        ensemble_model.fit(X_train, y_train)

        # Validation metrics
        y_pred = ensemble_model.predict(X_val)
        val_metrics = compute_metrics(
            y_val,
            y_pred,
            task_type=plan.task_type,
            estimator=ensemble_model,
            X_val=X_val,
        )

        # Train metrics
        y_train_pred = ensemble_model.predict(X_train)
        train_metrics = compute_metrics(
            y_train,
            y_train_pred,
            task_type=plan.task_type,
            estimator=ensemble_model,
            X_val=X_train,
        )

        duration = time.perf_counter() - t0

        ensemble_result = ExperimentResult(
            model_name="Weighted Ensemble",
            library="sklearn.ensemble.Voting",
            status="success",
            metrics=val_metrics,
            train_metrics=train_metrics,
            duration_seconds=duration,
            estimator=ensemble_model,
            hyperparams={"blended_models": [r.model_name for r in top_models]},
        )
        logger.info(f"Ensemble training complete — score ({primary_metric}): {val_metrics.get(primary_metric, 'N/A')}")
        return ensemble_result

    except Exception as exc:
        logger.warning(f"Failed to build Voting Ensemble: {exc}")
        return None
