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

from atlas_cli.agents.dataset_intelligence.cleaner import clean_dataset
from atlas_cli.agents.dataset_intelligence.loader import load_dataset
from atlas_cli.agents.dataset_intelligence.schema import ColumnSchema, SchemaReport, is_id_column
from atlas_cli.agents.experimentation.splitter import split_data
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
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str], pd.DataFrame, ColumnTransformer]:
    """
    Run leakage-free feature engineering pipeline for a given run ID:
      1. Load execution_plan.json, dataset_summary.json, and dataset.
      2. Auto-run clean_dataset if cleaned_data.csv does not exist.
      3. Drop ignored / ID / non-predictive columns.
      4. Perform reproducible data splitting into raw Train / Val / Test sets.
      5. Fit ColumnTransformer pipeline STRICTLY on X_train_raw to prevent data leakage.
      6. Transform X_train_raw, X_val_raw, and X_test_raw independently.
      7. Save fitted pipeline to .atlas_cli/runs/<run_id>/pipeline.joblib.
      8. Save feature_engineered_data.csv and features_meta.json.

    Args:
        run_id: Run identifier directory in workspace.
        file_path: Optional raw dataset file path override.
        random_seed: Random seed for reproducible splitting.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, fitted_pipeline, feature_names, X_train_raw, unfitted_pipeline)
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

    cleaned_csv = run_dir / "cleaned_data.csv"
    if file_path:
        dataset_file = file_path
    elif cleaned_csv.exists():
        dataset_file = cleaned_csv
    else:
        raw_file = Path(summary_dict.get("file_name", ""))
        if not raw_file.exists():
            cwd_match = Path.cwd() / raw_file.name
            if cwd_match.exists():
                raw_file = cwd_match
            else:
                raise FileNotFoundError(f"Source dataset file '{raw_file}' not found.")
        
        logger.info(f"No cleaned_data.csv found for run '{run_id}'. Auto-executing dataset cleaning...")
        ignore_cols = list(plan.preprocessing.drop_columns or [])
        clean_dataset(
            raw_file,
            target_col=plan.target_column,
            output_dir=run_dir,
            ignore_cols=ignore_cols,
        )
        cleaned_csv = run_dir / "cleaned_data.csv"
        dataset_file = cleaned_csv

    df, _ = load_dataset(dataset_file)
    logger.info(f"Loaded dataset '{dataset_file.name}' with shape {df.shape}")

    target_col = plan.target_column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    y_raw = df[target_col]

    # Collect columns to drop: explicit plan drop_columns + auto ID columns
    cols_to_drop = set(plan.preprocessing.drop_columns or [])
    for col in df.columns:
        if col != target_col and is_id_column(col, target_col):
            cols_to_drop.add(col)

    drop_list = [c for c in cols_to_drop if c in df.columns and c != target_col] + [target_col]
    X_raw = df.drop(columns=drop_list, errors="ignore")
    if drop_list:
        logger.info(f"Dropped non-predictive/ignored columns: {[c for c in drop_list if c != target_col]}")

    if plan.task_type in {"binary_classification", "multiclass_classification"}:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        joblib.dump(le, run_dir / "label_encoder.joblib")
    else:
        y = y_raw.values.astype(np.float64)

    # 1. Leakage-Free Splitting BEFORE Fitting Transformers
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
        X_raw, y, plan, random_seed=random_seed
    )

    # 2. Build Pipeline
    unfitted_pipeline = build_feature_pipeline(plan, schema, X_train_raw)
    
    # 3. Fit Pipeline ONLY on Training Split
    pipeline = build_feature_pipeline(plan, schema, X_train_raw)
    X_train = pipeline.fit_transform(X_train_raw)

    # 4. Transform Validation and Test Splits (No Re-fitting!)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    try:
        feature_names = list(pipeline.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    features_dir = run_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = run_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(pipeline, features_dir / "pipeline.joblib")

    # Export Feature Engineered Dataset CSV (Combined Transformed Dataset)
    X_all_processed = np.vstack([X_train, X_val, X_test]) if not hasattr(X_train, "toarray") else np.vstack([X_train.toarray(), X_val.toarray(), X_test.toarray()])
    y_all = np.concatenate([y_train, y_val, y_test])
    fe_df = pd.DataFrame(data=X_all_processed, columns=feature_names)
    fe_df[target_col] = y_all

    fe_csv_path = run_dir / "feature_engineered_data.csv"
    fe_df.to_csv(fe_csv_path, index=False)
    fe_df.to_csv(features_dir / "feature_engineered_data.csv", index=False)

    features_meta = {
        "num_samples": X_all_processed.shape[0],
        "num_features": X_all_processed.shape[1],
        "target_column": target_col,
        "feature_names": feature_names,
        "feature_engineered_file_path": str(fe_csv_path),
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
    }
    meta_json = json.dumps(features_meta, indent=2)
    (run_dir / "features_meta.json").write_text(meta_json, encoding="utf-8")
    (features_dir / "features_meta.json").write_text(meta_json, encoding="utf-8")

    logger.info(
        f"Feature engineering pipeline fitted strictly on training split (Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}). "
        f"Pipeline saved to: {pipeline_path}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, pipeline, feature_names, X_train_raw, unfitted_pipeline
