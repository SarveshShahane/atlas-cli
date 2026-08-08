"""
Reproducible Data Splitter — Phase 5.

Partitions processed feature arrays into train / validation / test splits
with explicit random seeds and optional stratification for classification tasks.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan

logger = logging.getLogger("atlas_cli")

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    plan: ExecutionPlan,
    *,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split processed features into train / validation / test sets.

    Strategy:
      1. Hold out ``plan.evaluation.test_size`` as the final unseen test set.
      2. From the remaining data, hold out 20% as a validation set for
         early-stopping and metric computation during training.

    For classification tasks, both splits are stratified on y.

    Args:
        X: Processed feature matrix (n_samples, n_features).
        y: Target array (n_samples,).
        plan: ExecutionPlan containing evaluation parameters.
        random_seed: Base seed for reproducibility.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_size = plan.evaluation.test_size
    is_classification = plan.task_type in _CLASSIFICATION_TASKS
    stratify_y = y if is_classification else None

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y,
    )

    val_fraction = 0.2   
    stratify_remaining = y_remaining if is_classification else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining, y_remaining,
        test_size=val_fraction,
        random_state=random_seed + 1,
        stratify=stratify_remaining,
    )

    logger.info(
        f"Data split complete — "
        f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}  "
        f"(test_size={test_size}, stratified={is_classification})"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
