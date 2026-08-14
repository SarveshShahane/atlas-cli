"""
Dynamic Scikit-Learn Pipeline Builder.
Constructs reproducible preprocessing pipelines based on ExecutionPlan and SchemaReport specs.
"""
from __future__ import annotations

from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from atlas_cli.agents.feature_engineering.transformers import (
    DatetimeFeatureExtractor,
    FrequencyEncoder,
    IQROutlierCapper,
    LogTransformFeature,
)
from atlas_cli.agents.dataset_intelligence.schema import SchemaReport
from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan


def build_feature_pipeline(
    plan: ExecutionPlan,
    schema: SchemaReport,
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Construct a Scikit-Learn ColumnTransformer pipeline dynamically from an ExecutionPlan.

    Args:
        plan: Validated ML ExecutionPlan from Phase 3.
        schema: SchemaReport from Phase 2.
        X: Raw feature DataFrame (excluding target column).

    Returns:
        Configured ColumnTransformer instance.
    """
    drop_cols = set(plan.preprocessing.drop_columns)
    available_cols = [c for c in X.columns if c not in drop_cols and c != plan.target_column]

    schema_col_map = {}
    for c in schema.columns:
        if isinstance(c, dict):
            schema_col_map[c.get("name")] = c.get("inferred_type")
        else:
            schema_col_map[c.name] = c.inferred_type

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    text_cols: List[str] = []

    for col in available_cols:
        col_type = schema_col_map.get(col, "numeric")
        if col_type == "numeric":
            numeric_cols.append(col)
        elif col_type in {"categorical", "boolean", "high_cardinality"}:
            categorical_cols.append(col)
        elif col_type == "datetime":
            datetime_cols.append(col)
        elif col_type == "text":
            text_cols.append(col)
        else:
            numeric_cols.append(col)

    transformers = []

    if numeric_cols:
        num_steps = []

        missing_strat = plan.preprocessing.missing_strategy
        if missing_strat in {"mean", "median", "most_frequent"}:
            num_steps.append(("imputer", SimpleImputer(strategy=missing_strat)))
        elif missing_strat == "drop":
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        else:
            num_steps.append(("imputer", SimpleImputer(strategy="median")))

        if plan.preprocessing.outlier_strategy == "iqr_cap":
            num_steps.append(("outliers", IQROutlierCapper()))

        if "log" in plan.feature_engineering.numeric_transforms:
            num_steps.append(("log_transform", LogTransformFeature()))

        # NOTE: Scaling is intentionally NOT applied here.
        # Scaling is handled per-model-family inside the training worker:
        #   - Linear models (LogisticRegression, SVM) get StandardScaler via Pipeline wrapper.
        #   - Tree-based models (RF, XGBoost, CatBoost, LightGBM) receive unscaled data.

        transformers.append(("numeric", Pipeline(num_steps), numeric_cols))

    if categorical_cols:
        cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
        cat_strat = plan.feature_engineering.categorical_strategy

        if cat_strat == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif cat_strat == "ordinal":
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        elif cat_strat == "freq_enc":
            cat_steps.append(("encoder", FrequencyEncoder()))
        else:
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))

        transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

    if datetime_cols:
        dt_components = plan.feature_engineering.datetime_features or ["year", "month", "day", "dayofweek", "is_weekend"]
        dt_pipeline = Pipeline([
            ("extractor", DatetimeFeatureExtractor(components=dt_components)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("datetime", dt_pipeline, datetime_cols))

    if text_cols and plan.feature_engineering.text_vectorizer in {"tfidf", "bow"}:
        for text_col in text_cols:
            transformers.append((
                f"text_{text_col}",
                TfidfVectorizer(max_features=100),
                text_col,
            ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,  # Preserve readable column names
    )
