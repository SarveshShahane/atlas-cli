"""
Statistical Profiler — computes per-column and dataset-level
quality metrics used for pipeline planning decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

HIGH_CORRELATION_THRESHOLD = 0.95
TARGET_IMBALANCE_THRESHOLD = 0.20
VIF_WARNING_THRESHOLD = 10.0
OUTLIER_IQR_MULTIPLIER = 1.5


@dataclass
class ColumnProfile:
    name: str
    missing_count: int
    missing_pct: float
    zero_variance: bool
    skewness: Optional[float]
    kurtosis: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    mad: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    p25: Optional[float]
    p75: Optional[float]
    normality_p_value: Optional[float]
    iqr_outliers_count: int = 0
    iqr_outliers_pct: float = 0.0
    zscore_outliers_count: int = 0
    zscore_outliers_pct: float = 0.0
    modified_zscore_outliers_count: int = 0
    modified_zscore_outliers_pct: float = 0.0
    iqr_lower_bound: Optional[float] = None
    iqr_upper_bound: Optional[float] = None


@dataclass
class CorrelationPair:
    col_a: str
    col_b: str
    correlation: float


@dataclass
class TargetImbalance:
    column: str
    class_distribution: dict[str, float]
    is_imbalanced: bool
    imbalance_ratio: float


@dataclass
class MutualInfoItem:
    column: str
    mi_score: float
    spearman_corr: Optional[float]


@dataclass
class VifItem:
    column: str
    vif: float


@dataclass
class MultivariateAnomalyReport:
    isolation_forest_count: int
    isolation_forest_pct: float
    lof_count: int
    lof_pct: float


@dataclass
class ProfileReport:
    num_rows: int
    num_cols: int
    duplicate_rows: int
    duplicate_pct: float
    columns: list[ColumnProfile] = field(default_factory=list)
    high_correlations: list[CorrelationPair] = field(default_factory=list)
    spearman_correlations: list[CorrelationPair] = field(default_factory=list)
    target_imbalance: Optional[TargetImbalance] = None
    mutual_information: list[MutualInfoItem] = field(default_factory=list)
    vif_metrics: list[VifItem] = field(default_factory=list)
    multivariate_anomalies: Optional[MultivariateAnomalyReport] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "columns": [
                {
                    "name": c.name,
                    "missing_count": c.missing_count,
                    "missing_pct": round(c.missing_pct, 2),
                    "zero_variance": c.zero_variance,
                    "skewness": round(c.skewness, 4) if c.skewness is not None else None,
                    "kurtosis": round(c.kurtosis, 4) if c.kurtosis is not None else None,
                    "mean": round(c.mean, 4) if c.mean is not None else None,
                    "std": round(c.std, 4) if c.std is not None else None,
                    "mad": round(c.mad, 4) if c.mad is not None else None,
                    "min": round(c.min_val, 4) if c.min_val is not None else None,
                    "max": round(c.max_val, 4) if c.max_val is not None else None,
                    "p25": round(c.p25, 4) if c.p25 is not None else None,
                    "p75": round(c.p75, 4) if c.p75 is not None else None,
                    "normality_p_value": round(c.normality_p_value, 4) if c.normality_p_value is not None else None,
                    "iqr_outliers": {
                        "count": c.iqr_outliers_count,
                        "pct": round(c.iqr_outliers_pct, 2),
                        "lower_bound": round(c.iqr_lower_bound, 4) if c.iqr_lower_bound is not None else None,
                        "upper_bound": round(c.iqr_upper_bound, 4) if c.iqr_upper_bound is not None else None,
                    },
                    "zscore_outliers": {
                        "count": c.zscore_outliers_count,
                        "pct": round(c.zscore_outliers_pct, 2),
                    },
                    "modified_zscore_outliers": {
                        "count": c.modified_zscore_outliers_count,
                        "pct": round(c.modified_zscore_outliers_pct, 2),
                    },
                }
                for c in self.columns
            ],
            "high_correlations": [
                {"col_a": p.col_a, "col_b": p.col_b, "correlation": round(p.correlation, 4)}
                for p in self.high_correlations
            ],
            "spearman_correlations": [
                {"col_a": p.col_a, "col_b": p.col_b, "correlation": round(p.correlation, 4)}
                for p in self.spearman_correlations
            ],
            "target_imbalance": (
                {
                    "column": self.target_imbalance.column,
                    "class_distribution": self.target_imbalance.class_distribution,
                    "is_imbalanced": self.target_imbalance.is_imbalanced,
                    "imbalance_ratio": round(self.target_imbalance.imbalance_ratio, 4),
                }
                if self.target_imbalance
                else None
            ),
            "mutual_information": [
                {
                    "column": mi.column,
                    "mi_score": round(mi.mi_score, 4),
                    "spearman_corr": round(mi.spearman_corr, 4) if mi.spearman_corr is not None else None,
                }
                for mi in self.mutual_information
            ],
            "vif_metrics": [
                {"column": v.column, "vif": round(v.vif, 2)} for v in self.vif_metrics
            ],
            "multivariate_anomalies": (
                {
                    "isolation_forest_count": self.multivariate_anomalies.isolation_forest_count,
                    "isolation_forest_pct": round(self.multivariate_anomalies.isolation_forest_pct, 2),
                    "lof_count": self.multivariate_anomalies.lof_count,
                    "lof_pct": round(self.multivariate_anomalies.lof_pct, 2),
                }
                if self.multivariate_anomalies
                else None
            ),
        }


def _profile_column(series: pd.Series, n: int) -> ColumnProfile:
    missing_count = int(series.isna().sum())
    missing_pct = (missing_count / n * 100) if n > 0 else 0.0

    if pd.api.types.is_numeric_dtype(series.dtype):
        clean = series.dropna()
        len_clean = len(clean)
        zero_variance = float(clean.var()) == 0.0 if len_clean > 1 else True
        skewness = float(clean.skew()) if len_clean > 2 else None
        kurtosis = float(clean.kurtosis()) if len_clean > 3 else None
        mean = float(clean.mean()) if len_clean > 0 else None
        std = float(clean.std()) if len_clean > 1 else None
        min_val = float(clean.min()) if len_clean > 0 else None
        max_val = float(clean.max()) if len_clean > 0 else None
        p25 = float(clean.quantile(0.25)) if len_clean > 0 else None
        p75 = float(clean.quantile(0.75)) if len_clean > 0 else None

        # MAD (Median Absolute Deviation)
        median = float(clean.median()) if len_clean > 0 else 0.0
        mad = float((clean - median).abs().median()) if len_clean > 0 else None

        # Normality test
        normality_p_val = None
        if len_clean >= 20:
            try:
                _, normality_p_val = stats.normaltest(clean)
                normality_p_val = float(normality_p_val)
            except Exception:
                normality_p_val = None

        # IQR Outliers
        iqr_count = 0
        iqr_pct = 0.0
        lower_b = None
        upper_b = None
        if p25 is not None and p75 is not None:
            iqr = p75 - p25
            lower_b = p25 - OUTLIER_IQR_MULTIPLIER * iqr
            upper_b = p75 + OUTLIER_IQR_MULTIPLIER * iqr
            iqr_count = int(((clean < lower_b) | (clean > upper_b)).sum())
            iqr_pct = float(iqr_count / len_clean * 100) if len_clean > 0 else 0.0

        # Z-score Outliers (|Z| > 3)
        zscore_count = 0
        zscore_pct = 0.0
        if std and std > 0 and len_clean > 5:
            z_scores = ((clean - mean) / std).abs()
            zscore_count = int((z_scores > 3.0).sum())
            zscore_pct = float(zscore_count / len_clean * 100)

        # Modified Z-score Outliers (MAD > 3.5)
        mod_zscore_count = 0
        mod_zscore_pct = 0.0
        if mad and mad > 0 and len_clean > 5:
            mod_z = (0.6745 * (clean - median).abs() / mad)
            mod_zscore_count = int((mod_z > 3.5).sum())
            mod_zscore_pct = float(mod_zscore_count / len_clean * 100)

    else:
        zero_variance = series.nunique(dropna=True) <= 1
        skewness = kurtosis = mean = std = mad = min_val = max_val = p25 = p75 = None
        normality_p_val = None
        iqr_count = zscore_count = mod_zscore_count = 0
        iqr_pct = zscore_pct = mod_zscore_pct = 0.0
        lower_b = upper_b = None

    return ColumnProfile(
        name=series.name,
        missing_count=missing_count,
        missing_pct=missing_pct,
        zero_variance=zero_variance,
        skewness=skewness,
        kurtosis=kurtosis,
        mean=mean,
        std=std,
        mad=mad,
        min_val=min_val,
        max_val=max_val,
        p25=p25,
        p75=p75,
        normality_p_value=normality_p_val,
        iqr_outliers_count=iqr_count,
        iqr_outliers_pct=iqr_pct,
        zscore_outliers_count=zscore_count,
        zscore_outliers_pct=zscore_pct,
        modified_zscore_outliers_count=mod_zscore_count,
        modified_zscore_outliers_pct=mod_zscore_pct,
        iqr_lower_bound=lower_b,
        iqr_upper_bound=upper_b,
    )


def _compute_correlations(df: pd.DataFrame, method: str = "pearson") -> list[CorrelationPair]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []

    corr = numeric_df.corr(method=method).abs()
    pairs: list[CorrelationPair] = []
    cols = list(corr.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            val = corr.loc[col_a, col_b]
            if pd.notna(val) and val >= HIGH_CORRELATION_THRESHOLD:
                pairs.append(CorrelationPair(col_a=col_a, col_b=col_b, correlation=float(val)))
    return sorted(pairs, key=lambda p: p.correlation, reverse=True)


def _compute_vif(df: pd.DataFrame) -> list[VifItem]:
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 5:
        return []

    std_cols = numeric_df.std()
    valid_cols = std_cols[std_cols > 0].index.tolist()
    if len(valid_cols) < 2:
        return []

    X = numeric_df[valid_cols].values
    vif_items: list[VifItem] = []

    for i, col in enumerate(valid_cols):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)

        model = LinearRegression().fit(X_i, y_i)
        r_sq = model.score(X_i, y_i)
        r_sq = min(r_sq, 0.99999)

        vif_val = float(1.0 / (1.0 - r_sq))
        if vif_val >= 2.0:
            vif_items.append(VifItem(column=col, vif=vif_val))

    return sorted(vif_items, key=lambda v: v.vif, reverse=True)


def _compute_mutual_info(df: pd.DataFrame, target_col: str) -> list[MutualInfoItem]:
    if target_col not in df.columns:
        return []

    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 10:
        return []

    y = clean_df[target_col]
    X_df = clean_df.drop(columns=[target_col])

    X_processed = pd.DataFrame()
    discrete_features = []

    for idx, col in enumerate(X_df.columns):
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_processed[col] = X_df[col].fillna(X_df[col].median())
        else:
            X_processed[col] = X_df[col].astype("category").cat.codes
            discrete_features.append(idx)

    if X_processed.shape[1] == 0:
        return []

    is_classification = (
        not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10 or pd.api.types.is_bool_dtype(y)
    )

    y_encoded = y if pd.api.types.is_numeric_dtype(y) else y.astype("category").cat.codes

    try:
        if is_classification:
            mi_scores = mutual_info_classif(
                X_processed, y_encoded, discrete_features=discrete_features, random_state=42
            )
        else:
            mi_scores = mutual_info_regression(
                X_processed, y_encoded, discrete_features=discrete_features, random_state=42
            )
    except Exception:
        return []

    mi_items: list[MutualInfoItem] = []
    y_num = pd.to_numeric(y, errors="coerce")

    for col, score in zip(X_df.columns, mi_scores):
        spearman_corr = None
        if y_num.notna().sum() > 5 and pd.api.types.is_numeric_dtype(X_df[col]):
            s_corr = X_df[col].corr(y_num, method="spearman")
            if pd.notna(s_corr):
                spearman_corr = float(abs(s_corr))

        mi_items.append(
            MutualInfoItem(column=col, mi_score=float(score), spearman_corr=spearman_corr)
        )

    return sorted(mi_items, key=lambda m: m.mi_score, reverse=True)


def _compute_multivariate_anomalies(df: pd.DataFrame) -> Optional[MultivariateAnomalyReport]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 10:
        return None

    valid_cols = [col for col in numeric_df.columns if numeric_df[col].std() > 0]
    if len(valid_cols) < 2:
        return None

    clean_X = numeric_df[valid_cols].fillna(numeric_df[valid_cols].median())

    try:
        # Isolation Forest
        if_clf = IsolationForest(random_state=42, contamination="auto", n_estimators=50)
        if_preds = if_clf.fit_predict(clean_X)
        if_anomalies = int((if_preds == -1).sum())
        if_pct = float(if_anomalies / len(df) * 100)

        # Local Outlier Factor (LOF)
        lof_clf = LocalOutlierFactor(n_neighbors=min(20, len(clean_X) - 1), contamination="auto")
        lof_preds = lof_clf.fit_predict(clean_X)
        lof_anomalies = int((lof_preds == -1).sum())
        lof_pct = float(lof_anomalies / len(df) * 100)

        return MultivariateAnomalyReport(
            isolation_forest_count=if_anomalies,
            isolation_forest_pct=if_pct,
            lof_count=lof_anomalies,
            lof_pct=lof_pct,
        )
    except Exception:
        return None


def _compute_target_imbalance(
    df: pd.DataFrame, target_col: str
) -> Optional[TargetImbalance]:
    if target_col not in df.columns:
        return None

    series = df[target_col].dropna()
    vc = series.value_counts(normalize=True)

    class_distribution = {str(k): round(float(v) * 100, 2) for k, v in vc.items()}
    majority_ratio = float(vc.iloc[0]) if len(vc) > 0 else 1.0
    minority_ratio = float(vc.iloc[-1]) if len(vc) > 1 else 1.0
    imbalance_ratio = minority_ratio / majority_ratio if majority_ratio > 0 else 1.0
    is_imbalanced = imbalance_ratio < TARGET_IMBALANCE_THRESHOLD

    return TargetImbalance(
        column=target_col,
        class_distribution=class_distribution,
        is_imbalanced=is_imbalanced,
        imbalance_ratio=imbalance_ratio,
    )


def profile_dataset(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> ProfileReport:
    """
    Run statistical profiling across all columns.

    Args:
        df: Loaded DataFrame.
        target_col: Optional target column name for imbalance and leakage analysis.

    Returns:
        ProfileReport with per-column stats and dataset-level metrics.
    """
    n = len(df)
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / n * 100) if n > 0 else 0.0

    column_profiles = [_profile_column(df[col], n) for col in df.columns]
    high_correlations = _compute_correlations(df, method="pearson")
    spearman_correlations = _compute_correlations(df, method="spearman")
    vif_metrics = _compute_vif(df)
    multivariate_anomalies = _compute_multivariate_anomalies(df)

    target_imbalance = (
        _compute_target_imbalance(df, target_col) if target_col else None
    )
    mutual_information = (
        _compute_mutual_info(df, target_col) if target_col else []
    )

    return ProfileReport(
        num_rows=n,
        num_cols=len(df.columns),
        duplicate_rows=dup_count,
        duplicate_pct=dup_pct,
        columns=column_profiles,
        high_correlations=high_correlations,
        spearman_correlations=spearman_correlations,
        target_imbalance=target_imbalance,
        mutual_information=mutual_information,
        vif_metrics=vif_metrics,
        multivariate_anomalies=multivariate_anomalies,
    )
