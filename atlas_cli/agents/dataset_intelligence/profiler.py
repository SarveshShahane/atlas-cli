"""
Statistical Profiler — computes per-column and dataset-level
quality metrics used for pipeline planning decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

HIGH_CORRELATION_THRESHOLD = 0.95

TARGET_IMBALANCE_THRESHOLD = 0.20


@dataclass
class ColumnProfile:
    name: str
    missing_count: int
    missing_pct: float
    zero_variance: bool
    skewness: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    p25: Optional[float]
    p75: Optional[float]


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
class ProfileReport:
    num_rows: int
    num_cols: int
    duplicate_rows: int
    duplicate_pct: float
    columns: list[ColumnProfile] = field(default_factory=list)
    high_correlations: list[CorrelationPair] = field(default_factory=list)
    target_imbalance: Optional[TargetImbalance] = None

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
                    "mean": round(c.mean, 4) if c.mean is not None else None,
                    "std": round(c.std, 4) if c.std is not None else None,
                    "min": round(c.min_val, 4) if c.min_val is not None else None,
                    "max": round(c.max_val, 4) if c.max_val is not None else None,
                    "p25": round(c.p25, 4) if c.p25 is not None else None,
                    "p75": round(c.p75, 4) if c.p75 is not None else None,
                }
                for c in self.columns
            ],
            "high_correlations": [
                {"col_a": p.col_a, "col_b": p.col_b, "correlation": round(p.correlation, 4)}
                for p in self.high_correlations
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
        }


def _profile_column(series: pd.Series, n: int) -> ColumnProfile:
    missing_count = int(series.isna().sum())
    missing_pct = (missing_count / n * 100) if n > 0 else 0.0

    if pd.api.types.is_numeric_dtype(series.dtype):
        clean = series.dropna()
        zero_variance = float(clean.var()) == 0.0 if len(clean) > 1 else True
        skewness = float(clean.skew()) if len(clean) > 2 else None
        mean = float(clean.mean()) if len(clean) > 0 else None
        std = float(clean.std()) if len(clean) > 1 else None
        min_val = float(clean.min()) if len(clean) > 0 else None
        max_val = float(clean.max()) if len(clean) > 0 else None
        p25 = float(clean.quantile(0.25)) if len(clean) > 0 else None
        p75 = float(clean.quantile(0.75)) if len(clean) > 0 else None
    else:
        zero_variance = series.nunique(dropna=True) <= 1
        skewness = mean = std = min_val = max_val = p25 = p75 = None

    return ColumnProfile(
        name=series.name,
        missing_count=missing_count,
        missing_pct=missing_pct,
        zero_variance=zero_variance,
        skewness=skewness,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        p25=p25,
        p75=p75,
    )


def _compute_correlations(df: pd.DataFrame) -> list[CorrelationPair]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []

    corr = numeric_df.corr().abs()
    pairs: list[CorrelationPair] = []
    cols = list(corr.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            val = corr.loc[col_a, col_b]
            if pd.notna(val) and val >= HIGH_CORRELATION_THRESHOLD:
                pairs.append(CorrelationPair(col_a=col_a, col_b=col_b, correlation=float(val)))
    return sorted(pairs, key=lambda p: p.correlation, reverse=True)


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
        target_col: Optional target column name for imbalance analysis.

    Returns:
        ProfileReport with per-column stats and dataset-level metrics.
    """
    n = len(df)
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / n * 100) if n > 0 else 0.0

    column_profiles = [_profile_column(df[col], n) for col in df.columns]
    high_correlations = _compute_correlations(df)
    target_imbalance = (
        _compute_target_imbalance(df, target_col) if target_col else None
    )

    return ProfileReport(
        num_rows=n,
        num_cols=len(df.columns),
        duplicate_rows=dup_count,
        duplicate_pct=dup_pct,
        columns=column_profiles,
        high_correlations=high_correlations,
        target_imbalance=target_imbalance,
    )
