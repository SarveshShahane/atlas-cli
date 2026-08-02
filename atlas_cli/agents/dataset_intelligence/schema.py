"""
Schema Inference — classifies each column into a semantic type
and produces a structured SchemaReport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

ColumnType = Literal[
    "numeric", "categorical", "datetime", "text", "boolean", "high_cardinality", "unknown"
]

HIGH_CARDINALITY_RATIO = 0.80

TEXT_WORD_THRESHOLD = 5

DATE_NAME_HINTS = {"date", "time", "year", "month", "day", "dt", "timestamp", "ts"}


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    inferred_type: ColumnType
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list


@dataclass
class SchemaReport:
    columns: list[ColumnSchema] = field(default_factory=list)
    num_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "num_rows": self.num_rows,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "inferred_type": c.inferred_type,
                    "null_count": c.null_count,
                    "null_pct": round(c.null_pct, 2),
                    "unique_count": c.unique_count,
                    "unique_pct": round(c.unique_pct, 2),
                    "sample_values": c.sample_values,
                }
                for c in self.columns
            ],
        }


def _has_date_name_hint(col_name: str) -> bool:
    lower = col_name.lower()
    return any(hint in lower for hint in DATE_NAME_HINTS)


def _median_word_count(series: pd.Series) -> float:
    sample = series.dropna().astype(str).head(500)
    return float(sample.str.split().apply(len).median()) if len(sample) > 0 else 0.0


def _infer_column_type(series: pd.Series, col_name: str) -> ColumnType:
    n = len(series)
    if n == 0:
        return "unknown"

    dtype = series.dtype

    if pd.api.types.is_datetime64_any_dtype(dtype) or _has_date_name_hint(col_name):
        if pd.api.types.is_object_dtype(dtype):
            try:
                pd.to_datetime(series.dropna().head(50), infer_datetime_format=True)
                return "datetime"
            except Exception:
                pass
        else:
            return "datetime"

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    n_unique = series.nunique(dropna=True)
    if n_unique <= 2 and pd.api.types.is_integer_dtype(dtype):
        return "boolean"

    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"

    if pd.api.types.is_object_dtype(dtype):
        unique_ratio = n_unique / max(n, 1)

        if unique_ratio >= HIGH_CARDINALITY_RATIO:
            if _median_word_count(series) >= TEXT_WORD_THRESHOLD:
                return "text"
            return "high_cardinality"

        if _median_word_count(series) >= TEXT_WORD_THRESHOLD:
            return "text"

        return "categorical"

    return "unknown"


def infer_schema(df: pd.DataFrame) -> SchemaReport:
    """
    Infer semantic types for every column in df.

    Args:
        df: Loaded pandas DataFrame.

    Returns:
        SchemaReport with per-column classification.
    """
    n = len(df)
    columns: list[ColumnSchema] = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = (null_count / n * 100) if n > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_pct = (unique_count / n * 100) if n > 0 else 0.0
        inferred_type = _infer_column_type(series, col)

        sample = series.dropna().head(3).tolist()
        sample_values = [str(v) for v in sample]

        columns.append(
            ColumnSchema(
                name=col,
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                null_count=null_count,
                null_pct=null_pct,
                unique_count=unique_count,
                unique_pct=unique_pct,
                sample_values=sample_values,
            )
        )

    return SchemaReport(columns=columns, num_rows=n)
