"""
Risk Detector — identifies data quality risks, leakage hazards,
multicollinearity, and outlier bounds from schema and profile reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from atlas_cli.agents.dataset_intelligence.profiler import ProfileReport
from atlas_cli.agents.dataset_intelligence.schema import SchemaReport

Severity = Literal["CRITICAL", "WARNING", "INFO"]

MISSING_CRITICAL_PCT = 50.0
MISSING_WARNING_PCT = 20.0
LEAKAGE_CORR_THRESHOLD = 0.97
LEAKAGE_MI_THRESHOLD = 0.85
OUTLIER_IQR_MULTIPLIER = 1.5
HIGH_SKEW_THRESHOLD = 2.0
VIF_CRITICAL_THRESHOLD = 10.0


@dataclass
class RiskItem:
    severity: Severity
    category: str
    column: Optional[str]
    description: str
    recommendation: str


@dataclass
class RiskAssessment:
    risks: list[RiskItem] = field(default_factory=list)
    overall_severity: Severity = "INFO"

    def to_dict(self) -> dict[str, Any]:
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        top = min(
            (r.severity for r in self.risks),
            key=lambda s: severity_order[s],
            default="INFO",
        )
        return {
            "overall_severity": top,
            "total_risks": len(self.risks),
            "risks": [
                {
                    "severity": r.severity,
                    "category": r.category,
                    "column": r.column,
                    "description": r.description,
                    "recommendation": r.recommendation,
                }
                for r in self.risks
            ],
        }


def _compute_outlier_pct(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 4:
        return 0.0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
    upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
    outlier_count = ((clean < lower) | (clean > upper)).sum()
    return float(outlier_count / len(clean) * 100)


def assess_risks(
    df: pd.DataFrame,
    schema: SchemaReport,
    profile: ProfileReport,
    target_col: Optional[str] = None,
) -> RiskAssessment:
    """
    Run all risk checks and return a structured RiskAssessment.

    Args:
        df: Loaded DataFrame.
        schema: Inferred schema report.
        profile: Statistical profile report.
        target_col: Optional target column for leakage checks.

    Returns:
        RiskAssessment with categorised risk items.
    """
    risks: list[RiskItem] = []

    # Missing Value checks
    for cp in profile.columns:
        if cp.missing_pct >= MISSING_CRITICAL_PCT:
            risks.append(
                RiskItem(
                    severity="CRITICAL",
                    category="Missing Values",
                    column=cp.name,
                    description=f"Column '{cp.name}' is {cp.missing_pct:.1f}% missing.",
                    recommendation="Consider dropping this column or applying aggressive imputation.",
                )
            )
        elif cp.missing_pct >= MISSING_WARNING_PCT:
            risks.append(
                RiskItem(
                    severity="WARNING",
                    category="Missing Values",
                    column=cp.name,
                    description=f"Column '{cp.name}' has {cp.missing_pct:.1f}% missing values.",
                    recommendation="Apply mean/median/mode imputation or KNN imputer.",
                )
            )

    # Zero Variance checks
    for cp in profile.columns:
        if cp.zero_variance:
            risks.append(
                RiskItem(
                    severity="WARNING",
                    category="Zero Variance",
                    column=cp.name,
                    description=f"Column '{cp.name}' has zero or near-zero variance.",
                    recommendation="Drop this column — it carries no predictive information.",
                )
            )

    # Skewness & Kurtosis checks
    for cp in profile.columns:
        if cp.skewness is not None and abs(cp.skewness) > HIGH_SKEW_THRESHOLD:
            risks.append(
                RiskItem(
                    severity="WARNING",
                    category="High Skewness",
                    column=cp.name,
                    description=f"Column '{cp.name}' has skewness {cp.skewness:.2f} (threshold: ±{HIGH_SKEW_THRESHOLD}).",
                    recommendation="Apply log transform, power transform (Yeo-Johnson), or Quantile Transformer.",
                )
            )

    # Duplicate rows check
    if profile.duplicate_pct > 5.0:
        risks.append(
            RiskItem(
                severity="WARNING",
                category="Duplicate Rows",
                column=None,
                description=f"{profile.duplicate_rows} duplicate rows detected ({profile.duplicate_pct:.1f}% of dataset).",
                recommendation="Deduplicate the dataset before training.",
            )
        )

    # High Correlation checks (Pearson)
    for pair in profile.high_correlations:
        risks.append(
            RiskItem(
                severity="WARNING",
                category="High Correlation",
                column=f"{pair.col_a} ↔ {pair.col_b}",
                description=f"Columns '{pair.col_a}' and '{pair.col_b}' are {pair.correlation:.2%} correlated (Pearson).",
                recommendation="Consider dropping one feature or applying PCA reduction.",
            )
        )

    # Multicollinearity (VIF)
    for vif_item in profile.vif_metrics:
        if vif_item.vif >= VIF_CRITICAL_THRESHOLD:
            risks.append(
                RiskItem(
                    severity="WARNING",
                    category="Multicollinearity",
                    column=vif_item.column,
                    description=f"Column '{vif_item.column}' has high Variance Inflation Factor (VIF = {vif_item.vif:.1f}).",
                    recommendation="Remove redundant feature or use L1 (Lasso) regularization.",
                )
            )

    # Target Imbalance
    if profile.target_imbalance and profile.target_imbalance.is_imbalanced:
        risks.append(
            RiskItem(
                severity="WARNING",
                category="Class Imbalance",
                column=profile.target_imbalance.column,
                description=(
                    f"Target column '{profile.target_imbalance.column}' is imbalanced "
                    f"(minority/majority ratio: {profile.target_imbalance.imbalance_ratio:.2%})."
                ),
                recommendation="Use SMOTE oversampling, class_weight='balanced', or stratified K-fold.",
            )
        )

    # Target Leakage — Pearson Correlation
    if target_col and target_col in df.columns:
        target = df[target_col]
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.notna().sum() > 0:
            for col in df.columns:
                if col == target_col:
                    continue
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    corr = df[col].corr(numeric_target)
                    if pd.notna(corr) and abs(corr) >= LEAKAGE_CORR_THRESHOLD:
                        risks.append(
                            RiskItem(
                                severity="CRITICAL",
                                category="Data Leakage",
                                column=col,
                                description=(
                                    f"Column '{col}' has {corr:.2%} correlation with target '{target_col}'. "
                                    "This may indicate data leakage."
                                ),
                                recommendation="Investigate whether this feature is causally valid or inadvertently derived from target.",
                            )
                        )

    # Target Leakage — Mutual Information (Non-Linear Dependencies)
    for mi_item in profile.mutual_information:
        if mi_item.mi_score >= LEAKAGE_MI_THRESHOLD and mi_item.column != target_col:
            risks.append(
                RiskItem(
                    severity="WARNING",
                    category="Non-Linear Leakage / Strong Dependency",
                    column=mi_item.column,
                    description=(
                        f"Column '{mi_item.column}' has high Mutual Information ({mi_item.mi_score:.3f}) with target."
                    ),
                    recommendation="Verify feature timing; strong non-linear coupling might indicate post-event feature collection.",
                )
            )

    # Outliers — Univariate IQR & Multivariate Isolation Forest
    for col in df.select_dtypes(include=[np.number]).columns:
        pct = _compute_outlier_pct(df[col])
        if pct > 10.0:
            risks.append(
                RiskItem(
                    severity="WARNING",
                    category="Outliers",
                    column=col,
                    description=f"Column '{col}' has {pct:.1f}% univariate outliers (IQR method).",
                    recommendation="Apply capping, Winsorization, or robust scaler.",
                )
            )

    if profile.multivariate_anomalies and profile.multivariate_anomalies.isolation_forest_pct > 5.0:
        an = profile.multivariate_anomalies
        risks.append(
            RiskItem(
                severity="WARNING",
                category="Multivariate Anomalies",
                column=None,
                description=(
                    f"Multivariate anomaly rows detected: Isolation Forest ({an.isolation_forest_count} rows, {an.isolation_forest_pct:.1f}%), "
                    f"Local Outlier Factor ({an.lof_count} rows, {an.lof_pct:.1f}%)."
                ),
                recommendation="Inspect extreme anomaly rows or use tree-based models resilient to noise.",
            )
        )

    # High Cardinality & Semantic Pattern Recommendations
    for cs in schema.columns:
        if cs.inferred_type == "high_cardinality":
            risks.append(
                RiskItem(
                    severity="INFO",
                    category="High Cardinality",
                    column=cs.name,
                    description=f"Column '{cs.name}' has {cs.unique_count} unique values ({cs.unique_pct:.1f}% of rows).",
                    recommendation="Use target encoding or hash encoding instead of one-hot encoding.",
                )
            )
        elif cs.inferred_type in ("email", "url", "ip_address", "phone_number"):
            risks.append(
                RiskItem(
                    severity="INFO",
                    category="Specialized Semantic Pattern",
                    column=cs.name,
                    description=f"Column '{cs.name}' detected as semantic format '{cs.inferred_type}'.",
                    recommendation=f"Extract sub-features (e.g. domain from email, TLD from URL, IP subnet) before encoding.",
                )
            )

    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    if risks:
        overall = min(risks, key=lambda r: severity_order[r.severity]).severity
    else:
        overall = "INFO"

    return RiskAssessment(risks=risks, overall_severity=overall)
