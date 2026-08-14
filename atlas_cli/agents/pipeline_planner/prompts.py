"""
Prompt template builders for the Pipeline Planner.
Constructs system and user messages from dataset intelligence artifacts.
"""
from __future__ import annotations

import json

MAX_RISKS_IN_PROMPT = 10

SYSTEM_PROMPT = """You are an expert Machine Learning engineer and data scientist.
Your task is to analyse a dataset's metadata and quality report provided by the user,
then produce a structured JSON execution plan for a machine learning pipeline.

Rules:
- Respond ONLY with a single valid JSON object. No markdown fences, no explanation outside the JSON.
- The JSON must exactly match the schema described by the user.
- Choose model candidates suited to the dataset size, feature types, and task type.
- Base all decisions on the dataset metadata and risks provided — do not make assumptions.
- If high multicollinearity features (VIF >= 10.0) or high correlation pairs are present, report them in `preprocessing.notes` as a diagnostic. Do NOT automatically add correlated features to `preprocessing.drop_columns` — tree-based models (Random Forest, XGBoost, CatBoost, LightGBM) handle collinearity well. Only drop truly non-predictive columns (IDs, zero-variance, data leakage suspects). For linear models, note that regularization (L1/L2) mitigates multicollinearity.
- If target imbalance is detected, set `evaluation.handle_imbalance: true` and select `evaluation.cv_strategy: "stratified_kfold"`.
- If datetime or time-series features are present, consider setting `evaluation.cv_strategy: "timeseries_split"`.
- For model_candidates, include 3–5 models ranked by expected suitability (priority 1 = best fit).
- For tabular data, strongly consider modern gradient boosted tree models (XGBoost, CatBoost, LightGBM) alongside classical baselines (Random Forest, Logistic Regression, SVC).
- Supported `library` strings:
  - Classification: `xgboost.XGBClassifier`, `catboost.CatBoostClassifier`, `lightgbm.LGBMClassifier`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.ExtraTreesClassifier`, `sklearn.ensemble.GradientBoostingClassifier`, `sklearn.linear_model.LogisticRegression`, `sklearn.svm.SVC`, `sklearn.neighbors.KNeighborsClassifier`
  - Regression: `xgboost.XGBRegressor`, `catboost.CatBoostRegressor`, `lightgbm.LGBMRegressor`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.ensemble.ExtraTreesRegressor`, `sklearn.ensemble.GradientBoostingRegressor`, `sklearn.linear_model.Ridge`, `sklearn.svm.SVR`, `sklearn.neighbors.KNeighborsRegressor`
- The "reasoning" field must explain your key decisions in 2–4 sentences."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(
    goal: str,
    dataset_summary: dict,
    quality_report: dict,
    risk_assessment: dict,
) -> str:
    """
    Construct the user message that injects all dataset context and specifies
    the JSON schema the model must return.

    Args:
        goal: The user's natural-language prediction goal.
        dataset_summary: Loaded dataset_summary.json dict.
        quality_report: Loaded quality_report.json dict.
        risk_assessment: Loaded risk_assessment.json dict.

    Returns:
        Formatted user prompt string.
    """
    all_risks = risk_assessment.get("risks", [])
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    sorted_risks = sorted(all_risks, key=lambda r: severity_order.get(r.get("severity", "INFO"), 2))
    top_risks = sorted_risks[:MAX_RISKS_IN_PROMPT]

    schema_cols = dataset_summary.get("schema", {}).get("columns", [])
    col_summary = [
        {
            "name": c["name"],
            "type": c["inferred_type"],
            "null_pct": c["null_pct"],
            "unique_count": c["unique_count"],
        }
        for c in schema_cols
    ]

    # Extract VIF multicollinearity metrics >= 10.0
    vif_list = quality_report.get("vif_metrics", [])
    high_vif = [v for v in vif_list if v.get("vif", 0) >= 10.0]

    # Extract Mutual Information top features
    mi_list = quality_report.get("mutual_information", [])[:5]

    profile_summary = {
        "num_rows": quality_report.get("num_rows"),
        "num_cols": quality_report.get("num_cols"),
        "duplicate_rows": quality_report.get("duplicate_rows"),
        "duplicate_pct": quality_report.get("duplicate_pct"),
        "target_imbalance": quality_report.get("target_imbalance"),
        "high_correlations": quality_report.get("high_correlations", [])[:5],
        "high_vif_features": high_vif,
        "top_mutual_information": mi_list,
        "multivariate_anomalies": quality_report.get("multivariate_anomalies"),
    }

    schema_description = """{
  "task_type": "<binary_classification|multiclass_classification|regression|clustering|time_series_forecasting>",
  "target_column": "<column name>",
  "reasoning": "<2-4 sentence explanation of key decisions>",
  "preprocessing": {
    "missing_strategy": "<mean|median|mode|knn|drop|none>",
    "outlier_strategy": "<iqr_cap|zscore_cap|winsorize|none>",
    "scale_strategy": "<standard|minmax|robust|none>",
    "drop_columns": ["<col1>", "..."],
    "notes": "<optional notes>"
  },
  "feature_engineering": {
    "numeric_transforms": ["<log|power|sqrt|...>"],
    "categorical_strategy": "<onehot|target_enc|freq_enc|ordinal|none>",
    "datetime_features": ["<year|month|day|dayofweek|is_weekend|...>"],
    "interaction_features": false,
    "text_vectorizer": "<tfidf|bow|none|null>"
  },
  "model_candidates": [
    {
      "name": "XGBoost Classifier",
      "library": "xgboost.XGBClassifier",
      "rationale": "High accuracy on tabular data with non-linear relationships.",
      "priority": 1
    },
    {
      "name": "CatBoost Classifier",
      "library": "catboost.CatBoostClassifier",
      "rationale": "Excellent default performance and robust categorical feature handling.",
      "priority": 2
    },
    {
      "name": "Random Forest Classifier",
      "library": "sklearn.ensemble.RandomForestClassifier",
      "rationale": "Solid ensemble baseline resistant to overfitting.",
      "priority": 3
    }
  ],
  "evaluation": {
    "primary_metric": "<roc_auc|f1_macro|accuracy|rmse|mae|r2|...>",
    "secondary_metrics": ["<metric1>", "..."],
    "cv_strategy": "<stratified_kfold|kfold|timeseries_split|holdout>",
    "n_folds": 5,
    "handle_imbalance": false,
    "test_size": 0.2
  }
}"""

    return f"""## User Goal
{goal}

## Dataset Summary
- File: {dataset_summary.get("file_name")} ({dataset_summary.get("file_format")}, {dataset_summary.get("file_size_mb")} MB)
- Shape: {dataset_summary.get("num_rows")} rows × {dataset_summary.get("num_cols")} columns
- Hash: {dataset_summary.get("dataset_hash")}

## Column Schema
{json.dumps(col_summary, indent=2)}

## Quality Profile
{json.dumps(profile_summary, indent=2)}

## Top Data Quality Risks
{json.dumps(top_risks, indent=2)}

## Required JSON Output Schema
Return ONLY this JSON structure filled with your decisions:
{schema_description}"""
