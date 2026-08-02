"""
Feature Engineering Orchestrator & Runner.
Loads dataset + execution plan, fits pipeline, transforms data, and serializes joblib artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from atlas_cli.agents.dataset_intelligence.loader import load_dataset
from atlas_cli.agents.dataset_intelligence.schema import ColumnSchema, SchemaReport
from atlas_cli.agents.feature_engineering.pipeline import build_feature_pipeline
from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan
from atlas_cli.core.config import settings
from atlas_cli.core.logger import logger


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def process_feature_engineering(
    run_id: str,
    file_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """
    Run feature engineering pipeline for a given run ID:
      1. Load execution_plan.json and dataset_summary.json.
      2. Load raw dataset.
      3. Construct and fit dynamic ColumnTransformer pipeline.
      4. Save fitted pipeline to .atlas_cli/runs/<run_id>/pipeline.joblib.
      5. Save features_meta.json.

    Args:
        run_id: Run identifier directory in workspace.
        file_path: Optional raw dataset file path override.

    Returns:
        (X_processed, y_processed, fitted_pipeline, feature_names)
    """
    run_dir = settings.workspace_dir / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for ID: {run_id}")

    plan_dict = _load_json(run_dir / "execution_plan.json", "Execution plan")
    summary_dict = _load_json(run_dir / "dataset_summary.json", "Dataset summary")

    plan = ExecutionPlan.model_validate(plan_dict)

    schema_dict = summary_dict.get("schema", {})
    cols_data = schema_dict.get("columns", [])
    columns = [ColumnSchema(**c) if isinstance(c, dict) else c for c in cols_data]
    schema = SchemaReport(columns=columns, num_rows=schema_dict.get("num_rows", 0))

    dataset_file = file_path or Path(summary_dict.get("file_name", ""))
    if not dataset_file.exists():
        cwd_match = Path.cwd() / dataset_file.name
        if cwd_match.exists():
            dataset_file = cwd_match
        else:
            raise FileNotFoundError(f"Source dataset file '{dataset_file}' not found.")

    df, _ = load_dataset(dataset_file)
    logger.info(f"Loaded raw dataset '{dataset_file.name}' with shape {df.shape}")

    target_col = plan.target_column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    if plan.task_type in {"binary_classification", "multiclass_classification"}:
        if not pd.api.types.is_numeric_dtype(y_raw.dtype):
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            joblib.dump(le, run_dir / "label_encoder.joblib")
        else:
            y = y_raw.values
    else:
        y = y_raw.values.astype(np.float64)

    pipeline = build_feature_pipeline(plan, schema, X_raw)
    X_processed = pipeline.fit_transform(X_raw)

    try:
        feature_names = list(pipeline.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

    pipeline_path = run_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    features_meta = {
        "num_samples": X_processed.shape[0],
        "num_features": X_processed.shape[1],
        "target_column": target_col,
        "feature_names": feature_names,
    }
    (run_dir / "features_meta.json").write_text(
        json.dumps(features_meta, indent=2), encoding="utf-8"
    )

    logger.info(
        f"Feature engineering pipeline fitted successfully. "
        f"Processed shape: {X_processed.shape}, Pipeline saved to: {pipeline_path}"
    )

    return X_processed, y, pipeline, feature_names
