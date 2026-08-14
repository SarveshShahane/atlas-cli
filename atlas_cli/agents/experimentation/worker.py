"""
Single Model Training Worker — Phase 5.

Self-contained training function designed to run inside a thread/process executor.
Catches all exceptions and returns structured results instead of propagating errors.
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from atlas_cli.agents.experimentation.metrics import compute_metrics
from atlas_cli.agents.pipeline_planner.schemas import ModelCandidate
from atlas_cli.models.wrappers import resolve_estimator

logger = logging.getLogger("atlas_cli")


@dataclass
class ExperimentResult:
    """Structured output of a single model training run."""

    model_name: str
    library: str
    status: str = "pending"
    metrics: dict[str, float] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    estimator: Optional[Any] = None
    error_message: Optional[str] = None
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerArgs:
    """Arguments bundle for the training worker."""

    candidate: ModelCandidate
    task_type: str
    handle_imbalance: bool
    random_seed: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    X_train_raw: Optional[Any] = None
    unfitted_pipeline: Optional[Any] = None


def _extract_feature_importances(estimator: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance or coefficient weights from fitted estimator."""
    importances: np.ndarray | None = None

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None or len(importances) == 0:
        return {}

    n_features = len(importances)
    col_names = (
        feature_names[:n_features]
        if len(feature_names) >= n_features
        else [f"feature_{i}" for i in range(n_features)]
    )

    fi_dict = {col: float(val) for col, val in zip(col_names, importances)}
    sorted_fi = dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_fi


def train_single_model(args: WorkerArgs) -> ExperimentResult:
    """
    Train a single model candidate and compute validation metrics.

    This function is designed to be submitted to a ``ThreadPoolExecutor`` or
    ``ProcessPoolExecutor``. It never raises; all errors are captured in the
    returned :class:`ExperimentResult`.

    Args:
        args: WorkerArgs containing the model candidate and data splits.

    Returns:
        ExperimentResult with metrics, trained estimator, and status.
    """
    candidate = args.candidate
    result = ExperimentResult(
        model_name=candidate.name,
        library=candidate.library,
    )

    t0 = time.perf_counter()

    try:
        estimator = resolve_estimator(
            candidate,
            task_type=args.task_type,
            handle_imbalance=args.handle_imbalance,
            random_seed=args.random_seed,
        )

        if hasattr(estimator, "get_params"):
            result.hyperparams = estimator.get_params()

        logger.info(f"[{candidate.name}] Training started...")

        # Attempt fit with early stopping for gradient boosted trees
        family = candidate.library.lower()
        fit_kwargs = {}

        if "xgboost" in family or "xgb" in family:
            fit_kwargs["eval_set"] = [(args.X_val, args.y_val)]
            fit_kwargs["verbose"] = False
        elif "catboost" in family:
            fit_kwargs["eval_set"] = (args.X_val, args.y_val)
            fit_kwargs["early_stopping_rounds"] = 15
            fit_kwargs["verbose"] = False
        elif "lightgbm" in family or "lgbm" in family:
            fit_kwargs["eval_set"] = [(args.X_val, args.y_val)]

        try:
            estimator.fit(args.X_train, args.y_train, **fit_kwargs)
        except Exception:
            # Fallback to standard fit without kwargs if library signature differs
            estimator.fit(args.X_train, args.y_train)

        # Extract feature importances
        result.feature_importances = _extract_feature_importances(estimator, args.feature_names)

        # Validation metrics
        y_pred = estimator.predict(args.X_val)
        result.metrics = compute_metrics(
            args.y_val,
            y_pred,
            task_type=args.task_type,
            estimator=estimator,
            X_val=args.X_val,
        )

        # Training metrics (for overfitting detection in Reviewer agent)
        y_train_pred = estimator.predict(args.X_train)
        result.train_metrics = compute_metrics(
            args.y_train,
            y_train_pred,
            task_type=args.task_type,
            estimator=estimator,
            X_val=args.X_train,
        )

        result.estimator = estimator
        result.status = "success"
        result.duration_seconds = time.perf_counter() - t0

        logger.info(
            f"[{candidate.name}] Training complete in {result.duration_seconds:.2f}s — "
            f"metrics: {result.metrics}"
        )

    except Exception as exc:
        result.status = "failed"
        result.duration_seconds = time.perf_counter() - t0
        result.error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error(f"[{candidate.name}] Training FAILED: {exc}")

    return result
