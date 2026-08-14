"""
Single Model Training Worker — Phase 5.

Self-contained training function designed to run inside a thread/process executor.
Catches all exceptions and returns structured results instead of propagating errors.

Implements proper Stratified K-Fold cross-validation for model comparison,
with separate validation and test-set evaluation.
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from atlas_cli.agents.experimentation.metrics import compute_metrics
from atlas_cli.agents.experimentation.splitter import _get_feature_group_ids
from atlas_cli.agents.pipeline_planner.schemas import ModelCandidate
from atlas_cli.models.wrappers import resolve_estimator

logger = logging.getLogger("atlas_cli")

# Model families that benefit from feature scaling
_LINEAR_FAMILIES = {
    "LogisticRegression", "Ridge", "Lasso", "ElasticNet",
    "SVC", "SVR", "SGDClassifier", "SGDRegressor",
    "KNeighborsClassifier", "KNeighborsRegressor",
}

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}


@dataclass
class CVResults:
    """Structured cross-validation results."""
    n_folds: int = 5
    strategy: str = "StratifiedKFold"
    n_groups: int = 0
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    f1_macro_mean: float = 0.0
    f1_macro_std: float = 0.0
    per_fold_accuracy: list[float] = field(default_factory=list)
    per_fold_f1_macro: list[float] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Structured output of a single model training run."""

    model_name: str
    library: str
    status: str = "pending"
    metrics: dict[str, float] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    cv_results: Optional[CVResults] = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    estimator: Optional[Any] = None
    error_message: Optional[str] = None
    hyperparams: dict[str, Any] = field(default_factory=dict)
    uses_scaling: bool = False


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
    n_folds: int = 5


def _needs_scaling(class_name: str) -> bool:
    """Check if a model class requires feature scaling."""
    return class_name in _LINEAR_FAMILIES


def _extract_feature_importances(estimator: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance or coefficient weights from fitted estimator."""
    # If the estimator is a Pipeline, extract the inner model
    actual_estimator = estimator
    if isinstance(estimator, Pipeline):
        actual_estimator = estimator.steps[-1][1]

    importances: np.ndarray | None = None

    if hasattr(actual_estimator, "feature_importances_"):
        importances = np.asarray(actual_estimator.feature_importances_, dtype=np.float64)
    elif hasattr(actual_estimator, "coef_"):
        coef = np.asarray(actual_estimator.coef_, dtype=np.float64)
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


def _run_cross_validation(
    estimator: Any,
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    task_type: str,
    n_folds: int,
    random_seed: int,
) -> CVResults:
    """
    Run k-fold cross-validation and return structured results.
    Uses StratifiedGroupKFold / GroupKFold when duplicate feature groups exist,
    otherwise uses StratifiedKFold for classification and KFold for regression.
    """
    is_classification = task_type in _CLASSIFICATION_TASKS

    # Detect duplicate feature groups in the CV set
    groups = _get_feature_group_ids(X_cv)
    n_unique_groups = len(np.unique(groups))
    has_duplicate_groups = n_unique_groups < len(X_cv)

    if has_duplicate_groups:
        if is_classification:
            cv_splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            strategy_name = "StratifiedGroupKFold"
        else:
            cv_splitter = GroupKFold(n_splits=n_folds)
            strategy_name = "GroupKFold"
        logger.info(
            f"Using {strategy_name} (groups: {n_unique_groups} unique feature groups across {len(X_cv)} rows)"
        )
    else:
        if is_classification:
            cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            strategy_name = "StratifiedKFold"
        else:
            cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            strategy_name = "KFold"

    # Define scoring metrics
    if is_classification:
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
        }
    else:
        scoring = {
            "neg_rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }

    try:
        cv_kwargs = {"groups": groups} if has_duplicate_groups else {}
        cv_output = cross_validate(
            estimator, X_cv, y_cv,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,  # Thread-safe
            **cv_kwargs,
        )

        cv_res = CVResults(
            n_folds=n_folds,
            strategy=strategy_name,
            n_groups=n_unique_groups,
        )

        if is_classification:
            acc_scores = cv_output["test_accuracy"]
            f1_scores = cv_output["test_f1_macro"]
            cv_res.accuracy_mean = float(np.mean(acc_scores))
            cv_res.accuracy_std = float(np.std(acc_scores))
            cv_res.f1_macro_mean = float(np.mean(f1_scores))
            cv_res.f1_macro_std = float(np.std(f1_scores))
            cv_res.per_fold_accuracy = [float(s) for s in acc_scores]
            cv_res.per_fold_f1_macro = [float(s) for s in f1_scores]
        else:
            rmse_scores = -cv_output["test_neg_rmse"]
            r2_scores = cv_output["test_r2"]
            cv_res.accuracy_mean = float(np.mean(r2_scores))  # Use R2 as "accuracy" for regression
            cv_res.accuracy_std = float(np.std(r2_scores))
            cv_res.f1_macro_mean = float(np.mean(rmse_scores))
            cv_res.f1_macro_std = float(np.std(rmse_scores))
            cv_res.per_fold_accuracy = [float(s) for s in r2_scores]
            cv_res.per_fold_f1_macro = [float(s) for s in rmse_scores]

        return cv_res

    except Exception as exc:
        logger.warning(f"Cross-validation failed: {exc}")
        return CVResults(n_folds=n_folds, strategy=strategy_name)


def train_single_model(args: WorkerArgs) -> ExperimentResult:
    """
    Train a single model candidate with proper cross-validation and validation metrics.

    Pipeline:
      1. Resolve estimator (optionally wrap in sklearn.Pipeline with StandardScaler).
      2. Run Stratified K-Fold CV on train+val combined data.
      3. Retrain on full training set.
      4. Evaluate on validation set.
      5. Return structured results (CV scores + val metrics).

    This function is designed to be submitted to a ``ThreadPoolExecutor``.
    It never raises; all errors are captured in the returned ExperimentResult.
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

        # Determine the class name for scaling decision
        class_name = type(estimator).__name__
        needs_scale = _needs_scaling(class_name)
        result.uses_scaling = needs_scale

        # Wrap linear models in Pipeline with StandardScaler
        if needs_scale:
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("model", estimator),
            ])
            logger.info(f"[{candidate.name}] Wrapped in StandardScaler pipeline (linear model)")

        if hasattr(estimator, "get_params"):
            result.hyperparams = estimator.get_params()

        logger.info(f"[{candidate.name}] Training started (with {args.n_folds}-fold CV)...")

        # ── 1. Cross-Validation on train+val combined ────────────────────
        X_cv = np.vstack([args.X_train, args.X_val])
        y_cv = np.concatenate([args.y_train, args.y_val])

        cv_results = _run_cross_validation(
            estimator, X_cv, y_cv,
            task_type=args.task_type,
            n_folds=args.n_folds,
            random_seed=args.random_seed,
        )
        result.cv_results = cv_results

        logger.info(
            f"[{candidate.name}] CV results: "
            f"accuracy={cv_results.accuracy_mean:.4f}±{cv_results.accuracy_std:.4f}, "
            f"f1_macro={cv_results.f1_macro_mean:.4f}±{cv_results.f1_macro_std:.4f}"
        )

        # ── 2. Train on full training set ────────────────────────────────
        family = candidate.library.lower()
        fit_kwargs: dict[str, Any] = {}

        if "xgboost" in family or "xgb" in family:
            fit_kwargs["eval_set"] = [(args.X_val, args.y_val)]
            fit_kwargs["verbose"] = False
        elif "catboost" in family:
            fit_kwargs["eval_set"] = (args.X_val, args.y_val)
            fit_kwargs["early_stopping_rounds"] = 15
            fit_kwargs["verbose"] = False
        elif "lightgbm" in family or "lgbm" in family:
            fit_kwargs["eval_set"] = [(args.X_val, args.y_val)]

        # For Pipeline-wrapped models, fit_kwargs with eval_set won't work
        # Fall back to standard fit
        if needs_scale and fit_kwargs:
            fit_kwargs = {}

        try:
            estimator.fit(args.X_train, args.y_train, **fit_kwargs)
        except Exception:
            # Fallback to standard fit without kwargs if library signature differs
            estimator.fit(args.X_train, args.y_train)

        # ── 3. Extract feature importances ───────────────────────────────
        result.feature_importances = _extract_feature_importances(estimator, args.feature_names)

        # ── 4. Validation metrics ────────────────────────────────────────
        y_pred = estimator.predict(args.X_val)
        result.metrics = compute_metrics(
            args.y_val,
            y_pred,
            task_type=args.task_type,
            estimator=estimator,
            X_val=args.X_val,
        )

        # ── 5. Training metrics (for overfitting detection) ──────────────
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
            f"val_metrics: {result.metrics}"
        )

    except Exception as exc:
        result.status = "failed"
        result.duration_seconds = time.perf_counter() - t0
        result.error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error(f"[{candidate.name}] Training FAILED: {exc}")

    return result
