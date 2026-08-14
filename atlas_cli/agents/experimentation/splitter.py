"""
Reproducible Data Splitter — Phase 5.

Partitions processed feature arrays into train / validation / test splits
with explicit random seeds and leakage-aware partitioning:
  - Stratified for classification tasks
  - Group-aware partitioning to prevent identical observations from crossing
    train / validation / test split boundaries
  - Sequential/TimeSeries for temporal tasks
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan

logger = logging.getLogger("atlas_cli")

_CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification"}


def _get_feature_group_ids(X: Any) -> np.ndarray:
    """
    Assign a deterministic integer group ID to each row based on its feature values.
    Identical feature rows will receive the exact same group ID.
    """
    if isinstance(X, pd.DataFrame):
        df_x = X
    else:
        df_x = pd.DataFrame(X)

    # Use pandas groupby.ngroup to assign unique IDs to distinct feature rows
    cols = list(df_x.columns)
    group_ids = df_x.groupby(cols, sort=False).ngroup().values
    return group_ids


def _group_stratified_split(
    X: Any,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_seed: int,
    is_classification: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition indices into train, validation, and test sets such that:
      1. All identical feature rows (same group) remain in the SAME partition.
      2. Class distribution is stratified across partitions as closely as possible.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    n_samples = len(y)
    group_ids = _get_feature_group_ids(X)
    n_groups = len(np.unique(group_ids))

    # Check if duplicates exist
    has_duplicates = n_groups < n_samples
    if has_duplicates:
        num_dup_rows = n_samples - n_groups
        logger.info(
            f"Duplicate feature records detected ({num_dup_rows} duplicate rows across "
            f"{n_groups} unique feature groups). Applying group-aware partitioning to "
            "strictly prevent train/val/test data leakage."
        )

    # Build unique group table: (group_id, class_label, row_indices)
    unique_groups = np.unique(group_ids)
    group_to_indices: dict[int, list[int]] = {}
    for idx, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(idx)

    # For each class (if classification), partition its unique groups
    rng = np.random.RandomState(random_seed)

    test_indices: list[int] = []
    val_indices: list[int] = []
    train_indices: list[int] = []

    if is_classification:
        # Group by majority class per group
        class_to_groups: dict[Any, list[int]] = {}
        for gid, indices in group_to_indices.items():
            classes_in_group = y[indices]
            # Primary class label for this group
            majority_class = pd.Series(classes_in_group).mode().iloc[0]
            class_to_groups.setdefault(majority_class, []).append(gid)

        for c_label, gids in class_to_groups.items():
            gids_shuffled = np.array(gids)
            rng.shuffle(gids_shuffled)

            # Count total rows for these groups
            group_row_counts = [len(group_to_indices[g]) for g in gids_shuffled]
            total_class_rows = sum(group_row_counts)

            target_test_rows = max(1, int(round(total_class_rows * test_size)))
            target_val_rows = max(1, int(round(total_class_rows * val_size)))

            cur_test_rows = 0
            cur_val_rows = 0

            for gid, row_cnt in zip(gids_shuffled, group_row_counts):
                if cur_test_rows + row_cnt <= target_test_rows or (cur_test_rows == 0 and target_test_rows > 0):
                    test_indices.extend(group_to_indices[gid])
                    cur_test_rows += row_cnt
                elif cur_val_rows + row_cnt <= target_val_rows or (cur_val_rows == 0 and target_val_rows > 0):
                    val_indices.extend(group_to_indices[gid])
                    cur_val_rows += row_cnt
                else:
                    train_indices.extend(group_to_indices[gid])

    else:
        # Regression or clustering: group partition without class labels
        unique_groups_shuffled = np.array(unique_groups)
        rng.shuffle(unique_groups_shuffled)

        group_row_counts = [len(group_to_indices[g]) for g in unique_groups_shuffled]
        total_rows = sum(group_row_counts)

        target_test_rows = max(1, int(round(total_rows * test_size)))
        target_val_rows = max(1, int(round(total_rows * val_size)))

        cur_test_rows = 0
        cur_val_rows = 0

        for gid, row_cnt in zip(unique_groups_shuffled, group_row_counts):
            if cur_test_rows + row_cnt <= target_test_rows or (cur_test_rows == 0 and target_test_rows > 0):
                test_indices.extend(group_to_indices[gid])
                cur_test_rows += row_cnt
            elif cur_val_rows + row_cnt <= target_val_rows or (cur_val_rows == 0 and target_val_rows > 0):
                val_indices.extend(group_to_indices[gid])
                cur_val_rows += row_cnt
            else:
                train_indices.extend(group_to_indices[gid])

    # Fallback if any partition is empty
    if not train_indices:
        if len(val_indices) > 1:
            train_indices.append(val_indices.pop())
        elif len(test_indices) > 1:
            train_indices.append(test_indices.pop())

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def split_data(
    X: Any,
    y: np.ndarray,
    plan: ExecutionPlan,
    *,
    random_seed: int = 42,
) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split processed features into train / validation / test sets.

    Strategy:
      1. If time_series_forecasting or cv_strategy == "timeseries_split", perform sequential split (no shuffle) to prevent temporal data leakage.
      2. Otherwise, perform group-aware stratified partition so that identical feature observations never cross partition boundaries.

    Args:
        X: Processed feature matrix (pd.DataFrame or np.ndarray).
        y: Target array (n_samples,).
        plan: ExecutionPlan containing evaluation parameters.
        random_seed: Base seed for reproducibility.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_size = plan.evaluation.test_size
    val_fraction = 0.2  # 20% of remaining data
    val_size = (1.0 - test_size) * val_fraction

    is_time_series = (
        plan.task_type == "time_series_forecasting"
        or plan.evaluation.cv_strategy == "timeseries_split"
    )
    is_classification = plan.task_type in _CLASSIFICATION_TASKS

    if is_time_series:
        logger.info("Time Series task detected: applying sequential (non-shuffled) data split to prevent temporal leakage.")
        n_total = len(X)
        n_test = int(n_total * test_size)
        n_rem = n_total - n_test
        n_val = int(n_rem * val_fraction)
        n_train = n_rem - n_val

        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_rem)
        test_idx = np.arange(n_rem, n_total)
    else:
        train_idx, val_idx, test_idx = _group_stratified_split(
            X, y,
            test_size=test_size,
            val_size=val_size,
            random_seed=random_seed,
            is_classification=is_classification,
        )

    # Slice X and y
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
    else:
        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    logger.info(
        f"Data split complete (Group-Stratified) — "
        f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}  "
        f"(test_size={test_size:.2f}, val_size={val_size:.2f}, time_series={is_time_series}, stratified={is_classification})"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
