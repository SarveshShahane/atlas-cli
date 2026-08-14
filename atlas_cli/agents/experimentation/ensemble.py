"""
Automated Ensemble Synthesizer — Phase 5.

Combines top-performing distinct candidate estimators into a Weighted Voting Ensemble
(soft-voting classifier or weighted averaging regressor) and evaluates its performance
using the exact same leakage-free cross-validation and evaluation framework.
Weights are derived strictly from CV / training scores and sum to 1.0.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor

from atlas_cli.agents.experimentation.metrics import compute_metrics
from atlas_cli.agents.experimentation.worker import (
    CVResults,
    ExperimentResult,
    _run_cross_validation,
)
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
    *,
    n_folds: int = 5,
    random_seed: int = 42,
) -> Optional[ExperimentResult]:
    """
    Build a Weighted Voting Ensemble from top successful candidate estimators and evaluate
    it with Stratified (Group) K-Fold Cross-Validation, validation, and training metrics.
    Weights are derived exclusively from CV performance and strictly sum to 1.0.

    Args:
        results: List of completed ExperimentResult objects.
        plan: ExecutionPlan containing task type and primary metric.
        X_train: Preprocessed training feature matrix.
        y_train: Training target labels.
        X_val: Preprocessed validation feature matrix.
        y_val: Validation target labels.
        n_folds: Number of folds for cross-validation.
        random_seed: Random seed for CV splitting.

    Returns:
        New ExperimentResult for the Weighted Ensemble with complete CVResults,
        or None if insufficient models.
    """
    successful_results = [r for r in results if r.status == "success" and r.estimator is not None]
    if len(successful_results) < 2:
        logger.info("Fewer than 2 successful estimators; skipping ensemble synthesis.")
        return None

    primary_metric = plan.evaluation.primary_metric
    is_classification = plan.task_type in _CLASSIFICATION_TASKS

    # Sort successful models by CV macro-F1 (or primary metric fallback)
    def _rank_score(r: ExperimentResult) -> float:
        if r.cv_results:
            return r.cv_results.f1_macro_mean if is_classification else -r.cv_results.f1_macro_mean
        return r.metrics.get(primary_metric, 0.0)

    sorted_candidates = sorted(successful_results, key=_rank_score, reverse=True)

    # Pick top 3 distinct models
    top_models = sorted_candidates[:3]
    estimators = [(r.model_name.lower().replace(" ", "_"), r.estimator) for r in top_models]

    t0 = time.perf_counter()

    try:
        # Compute normalized weights derived exclusively from CV scores (sum strictly to 1.0)
        scores = []
        for r in top_models:
            if r.cv_results and is_classification:
                scores.append(max(0.01, r.cv_results.f1_macro_mean))
            elif r.cv_results and not is_classification:
                rmse = max(0.001, r.cv_results.f1_macro_mean)
                scores.append(1.0 / rmse)
            else:
                scores.append(max(0.01, r.metrics.get(primary_metric, 1.0)))

        scores_arr = np.array(scores, dtype=float)
        weights_arr = scores_arr / np.sum(scores_arr)
        weights_arr = np.round(weights_arr, 4)
        weights_arr[-1] = round(1.0 - float(np.sum(weights_arr[:-1])), 4)
        weights_list = [float(w) for w in weights_arr]
        weights_dict = {r.model_name: float(w) for r, w in zip(top_models, weights_list)}

        logger.info(f"Synthesizing Weighted Ensemble from top {len(estimators)} models with CV-derived weights: {weights_dict}")

        if is_classification:
            # Soft voting if all estimators support predict_proba, else hard voting
            has_proba = all(hasattr(e, "predict_proba") for _, e in estimators)
            voting_mode = "soft" if has_proba else "hard"
            ensemble_model = VotingClassifier(estimators=estimators, voting=voting_mode, weights=weights_list)
        else:
            ensemble_model = VotingRegressor(estimators=estimators, weights=weights_list)

        # ── 1. Stratified (Group) K-Fold Cross-Validation on combined train+val ───
        X_cv = np.vstack([X_train, X_val])
        y_cv = np.concatenate([y_train, y_val])

        cv_results = _run_cross_validation(
            ensemble_model, X_cv, y_cv,
            task_type=plan.task_type,
            n_folds=n_folds,
            random_seed=random_seed,
        )

        logger.info(
            f"[Weighted Ensemble] CV results: "
            f"accuracy={cv_results.accuracy_mean:.4f}±{cv_results.accuracy_std:.4f}, "
            f"f1_macro={cv_results.f1_macro_mean:.4f}±{cv_results.f1_macro_std:.4f}"
        )

        # ── 2. Train on full training set ────────────────────────────────────
        ensemble_model.fit(X_train, y_train)

        # ── 3. Validation metrics ────────────────────────────────────────────
        y_pred = ensemble_model.predict(X_val)
        val_metrics = compute_metrics(
            y_val,
            y_pred,
            task_type=plan.task_type,
            estimator=ensemble_model,
            X_val=X_val,
        )

        # ── 4. Training metrics ──────────────────────────────────────────────
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
            cv_results=cv_results,
            duration_seconds=duration,
            estimator=ensemble_model,
            hyperparams={
                "blended_models": [r.model_name for r in top_models],
                "weights": weights_dict,
            },
            feature_importances=weights_dict,
        )

        return ensemble_result

    except Exception as exc:
        logger.warning(f"Ensemble synthesis failed: {exc}")
        return None
