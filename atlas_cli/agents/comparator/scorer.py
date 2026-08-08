"""
Experiment Scorer & Ranker — Phase 6.

Computes extended evaluation metrics on the held-out test set, calculates a
weighted multi-objective composite score, and ranks all successful experiments.

Composite score formula (higher is better):
    score = w_perf * norm(primary_metric)
          + w_lat  * norm(1 / inference_time)
          + w_size * norm(1 / model_size)
          + w_cost * norm(1 / training_time)

All dimensions are min-max normalised to [0, 1] across candidates.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.metrics import log_loss

from atlas_cli.agents.experimentation.metrics import compute_metrics
from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

#  Composite score weights 
W_PERFORMANCE = 0.50
W_LATENCY = 0.20
W_SIZE = 0.15
W_COST = 0.15

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}


# Data classes 
@dataclass
class ExtendedMetrics:
    """Extended evaluation metrics for a single experiment."""

    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    log_loss_val: Optional[float] = None
    inference_time_ms: float = 0.0
    model_size_kb: float = 0.0
    training_time_s: float = 0.0


@dataclass
class RankedExperiment:
    """A fully ranked experiment entry."""

    rank: int
    model_name: str
    library: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    inference_time_ms: float
    model_size_kb: float
    training_time_s: float
    composite_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    is_winner: bool = False
    error_message: Optional[str] = None


# Extended metric computation 
def compute_extended_metrics(
    run_id: str,
    experiment_entry: dict[str, Any],
    *,
    task_type: str,
) -> Optional[ExtendedMetrics]:
    """
    Compute extended metrics for a single experiment by loading the saved
    model artifact and test splits.

    Args:
        run_id: The run identifier.
        experiment_entry: Dict from experiment_results.json with model info.
        task_type: The ML task type string.

    Returns:
        ExtendedMetrics or None if the experiment failed / artifacts missing.
    """
    if experiment_entry.get("status") != "success":
        return None

    run_dir = settings.workspace_dir / "runs" / run_id
    model_name = experiment_entry["model_name"]
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    model_path = run_dir / "models" / f"{safe_name}.joblib"

    if not model_path.exists():
        logger.warning(f"[{model_name}] Model artifact not found: {model_path}")
        return None

    splits_dir = run_dir / "splits"
    try:
        X_test = np.load(splits_dir / "X_test.npy")
        y_test = np.load(splits_dir / "y_test.npy")
        X_val = np.load(splits_dir / "X_val.npy")
        y_val = np.load(splits_dir / "y_val.npy")
    except FileNotFoundError as exc:
        logger.warning(f"[{model_name}] Split data not found: {exc}")
        return None

    try:
        estimator = joblib.load(model_path)
    except Exception as exc:
        logger.warning(f"[{model_name}] Failed to load model: {exc}")
        return None

    em = ExtendedMetrics()

    # Validation metrics (already computed, carry forward)
    em.val_metrics = dict(experiment_entry.get("metrics", {}))

    # Test-set metrics
    try:
        y_pred_test = estimator.predict(X_test)
        em.test_metrics = compute_metrics(
            y_test,
            y_pred_test,
            task_type=task_type,
            estimator=estimator,
            X_val=X_test,
        )
    except Exception as exc:
        logger.warning(f"[{model_name}] Test metric computation failed: {exc}")
        em.test_metrics = {}

    # Log-loss (classification only)
    if task_type in _CLASSIFICATION_TASKS and hasattr(estimator, "predict_proba"):
        try:
            y_proba_val = estimator.predict_proba(X_val)
            em.log_loss_val = float(log_loss(y_val, y_proba_val))
        except Exception as exc:
            logger.warning(f"[{model_name}] Log-loss computation failed: {exc}")

    # Inference time (average over test set)
    try:
        n_runs = 3
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            estimator.predict(X_test)
            times.append(time.perf_counter() - t0)
        avg_time_s = sum(times) / len(times)
        em.inference_time_ms = round(avg_time_s * 1000, 3)
    except Exception as exc:
        logger.warning(f"[{model_name}] Inference timing failed: {exc}")

    # Model file size
    try:
        em.model_size_kb = round(os.path.getsize(model_path) / 1024, 2)
    except OSError:
        pass

    # Training time (from experiment results)
    em.training_time_s = experiment_entry.get("duration_seconds", 0.0)

    return em


# Normalisation helper 
def _min_max_normalise(values: list[float]) -> list[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    span = v_max - v_min
    if span == 0:
        return [1.0] * len(values)
    return [(v - v_min) / span for v in values]


#  Composite scoring
def compute_composite_scores(
    extended_list: list[tuple[dict[str, Any], ExtendedMetrics]],
    primary_metric: str,
) -> list[tuple[float, dict[str, float]]]:
    """
    Compute multi-objective composite scores for all candidates.

    Args:
        extended_list: List of (experiment_entry, ExtendedMetrics) tuples.
        primary_metric: Name of the primary evaluation metric.

    Returns:
        List of (composite_score, breakdown_dict) in the same order as input.
    """
    n = len(extended_list)
    if n == 0:
        return []

    # Extract raw values
    perf_raw = []
    latency_raw = []
    size_raw = []
    cost_raw = []

    for _, em in extended_list:
        # Performance: use test metrics if available, else val metrics
        perf_val = em.test_metrics.get(primary_metric, em.val_metrics.get(primary_metric, 0.0))
        perf_raw.append(perf_val)

        # For latency/size/cost, lower is better → we will invert after normalisation
        latency_raw.append(em.inference_time_ms if em.inference_time_ms > 0 else 0.001)
        size_raw.append(em.model_size_kb if em.model_size_kb > 0 else 0.001)
        cost_raw.append(em.training_time_s if em.training_time_s > 0 else 0.001)

    # Normalise
    perf_norm = _min_max_normalise(perf_raw)
    # For lower-is-better metrics, invert the normalised score
    lat_norm = [1.0 - v for v in _min_max_normalise(latency_raw)]
    size_norm = [1.0 - v for v in _min_max_normalise(size_raw)]
    cost_norm = [1.0 - v for v in _min_max_normalise(cost_raw)]

    results = []
    for i in range(n):
        score = (
            W_PERFORMANCE * perf_norm[i]
            + W_LATENCY * lat_norm[i]
            + W_SIZE * size_norm[i]
            + W_COST * cost_norm[i]
        )
        breakdown = {
            "performance": round(perf_norm[i], 4),
            "latency": round(lat_norm[i], 4),
            "size": round(size_norm[i], 4),
            "cost": round(cost_norm[i], 4),
            "weighted_total": round(score, 4),
        }
        results.append((round(score, 4), breakdown))

    return results


# Ranking 
def rank_experiments(
    run_id: str,
    *,
    task_type: str,
    primary_metric: str,
    experiment_entries: list[dict[str, Any]],
) -> list[RankedExperiment]:
    """
    Full ranking pipeline: compute extended metrics → composite scores → sort.

    Args:
        run_id: Run identifier.
        task_type: The ML task type.
        primary_metric: Name of the primary metric (from the execution plan).
        experiment_entries: List of experiment dicts from experiment_results.json.

    Returns:
        Sorted list of RankedExperiment (highest composite score first).
    """
    # Compute extended metrics for each successful experiment
    extended_list: list[tuple[dict[str, Any], ExtendedMetrics]] = []
    failed_entries: list[dict[str, Any]] = []

    for entry in experiment_entries:
        em = compute_extended_metrics(run_id, entry, task_type=task_type)
        if em is not None:
            extended_list.append((entry, em))
        else:
            failed_entries.append(entry)

    if not extended_list:
        logger.warning("No successful experiments to rank.")
        # Return failed entries as unranked
        return [
            RankedExperiment(
                rank=i + 1,
                model_name=e.get("model_name", "Unknown"),
                library=e.get("library", "unknown"),
                val_metrics={},
                test_metrics={},
                inference_time_ms=0.0,
                model_size_kb=0.0,
                training_time_s=e.get("duration_seconds", 0.0),
                composite_score=0.0,
                is_winner=False,
                error_message=e.get("error"),
            )
            for i, e in enumerate(failed_entries)
        ]

    # Compute composite scores
    scores = compute_composite_scores(extended_list, primary_metric)

    # Build ranked list
    ranked: list[RankedExperiment] = []
    for (entry, em), (score, breakdown) in zip(extended_list, scores):
        ranked.append(
            RankedExperiment(
                rank=0,  # set after sort
                model_name=entry["model_name"],
                library=entry.get("library", "unknown"),
                val_metrics=em.val_metrics,
                test_metrics=em.test_metrics,
                inference_time_ms=em.inference_time_ms,
                model_size_kb=em.model_size_kb,
                training_time_s=em.training_time_s,
                composite_score=score,
                score_breakdown=breakdown,
            )
        )

    # Add failed entries at the bottom
    for entry in failed_entries:
        ranked.append(
            RankedExperiment(
                rank=0,
                model_name=entry.get("model_name", "Unknown"),
                library=entry.get("library", "unknown"),
                val_metrics={},
                test_metrics={},
                inference_time_ms=0.0,
                model_size_kb=0.0,
                training_time_s=entry.get("duration_seconds", 0.0),
                composite_score=0.0,
                is_winner=False,
                error_message=entry.get("error"),
            )
        )

    # Sort by composite score descending
    ranked.sort(key=lambda r: r.composite_score, reverse=True)

    # Assign ranks and mark winner
    for i, r in enumerate(ranked):
        r.rank = i + 1
        if i == 0 and r.composite_score > 0:
            r.is_winner = True

    return ranked
