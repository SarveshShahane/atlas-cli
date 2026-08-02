"""
Pydantic v2 plan schemas — structured output models for the Pipeline Planner.
These are what the LLM is expected to produce and what gets persisted as execution_plan.json.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
    "time_series_forecasting",
]

MissingStrategy = Literal["mean", "median", "mode", "knn", "drop", "none"]
OutlierStrategy = Literal["iqr_cap", "zscore_cap", "winsorize", "none"]
ScaleStrategy = Literal["standard", "minmax", "robust", "none"]
CategoricalStrategy = Literal["onehot", "target_enc", "freq_enc", "ordinal", "none"]
CVStrategy = Literal["stratified_kfold", "kfold", "timeseries_split", "holdout"]


class PreprocessingPlan(BaseModel):
    missing_strategy: MissingStrategy = Field(
        ..., description="Strategy to handle missing numeric values."
    )
    outlier_strategy: OutlierStrategy = Field(
        ..., description="Strategy to handle numeric outliers."
    )
    scale_strategy: ScaleStrategy = Field(
        ..., description="Numeric feature scaling method."
    )
    drop_columns: list[str] = Field(
        default_factory=list,
        description="Columns to drop (zero variance, high missingness, leakage suspects).",
    )
    notes: Optional[str] = Field(None, description="Any additional preprocessing notes.")


class FeatureEngineeringPlan(BaseModel):
    numeric_transforms: list[str] = Field(
        default_factory=list,
        description="Transforms to apply to numeric columns, e.g. ['log', 'power', 'sqrt'].",
    )
    categorical_strategy: CategoricalStrategy = Field(
        ..., description="Encoding strategy for categorical columns."
    )
    datetime_features: list[str] = Field(
        default_factory=list,
        description="Datetime components to extract, e.g. ['year', 'month', 'dayofweek', 'is_weekend'].",
    )
    interaction_features: bool = Field(
        False, description="Whether to generate polynomial interaction features."
    )
    text_vectorizer: Optional[Literal["tfidf", "bow", "none"]] = Field(
        None, description="Vectorizer for text columns, if any."
    )


class ModelCandidate(BaseModel):
    name: str = Field(..., description="Human-readable model name, e.g. 'XGBoost'.")
    library: str = Field(
        ..., description="Python library identifier, e.g. 'xgboost.XGBClassifier'."
    )
    rationale: str = Field(
        ..., description="One-sentence justification for why this model suits the dataset."
    )
    priority: int = Field(
        1,
        ge=1,
        le=5,
        description="Training priority (1=highest). Determines execution order in parallel runs.",
    )


class EvaluationPlan(BaseModel):
    primary_metric: str = Field(
        ...,
        description="Primary evaluation metric, e.g. 'roc_auc', 'f1_macro', 'rmse'.",
    )
    secondary_metrics: list[str] = Field(
        default_factory=list,
        description="Additional metrics to compute, e.g. ['accuracy', 'precision', 'recall'].",
    )
    cv_strategy: CVStrategy = Field(
        ..., description="Cross-validation strategy."
    )
    n_folds: int = Field(5, ge=2, le=20, description="Number of CV folds.")
    handle_imbalance: bool = Field(
        False, description="Whether to apply class imbalance handling (SMOTE / class weights)."
    )
    test_size: float = Field(
        0.2, ge=0.05, le=0.4, description="Fraction of data to hold out as the final test set."
    )


class ExecutionPlan(BaseModel):
    """
    Top-level validated ML execution plan produced by the Pipeline Planner LLM.
    This is persisted as execution_plan.json and drives all downstream phases.
    """
    task_type: TaskType = Field(..., description="Inferred ML task type.")
    target_column: str = Field(..., description="Name of the target/label column.")
    reasoning: str = Field(
        ...,
        description="Brief natural-language explanation of how the dataset properties informed this plan.",
    )
    preprocessing: PreprocessingPlan
    feature_engineering: FeatureEngineeringPlan
    model_candidates: list[ModelCandidate] = Field(
        ..., min_length=2, max_length=6,
        description="Ordered list of model candidates to train (2–6 models).",
    )
    evaluation: EvaluationPlan

    def to_dict(self) -> dict:
        return self.model_dump()
