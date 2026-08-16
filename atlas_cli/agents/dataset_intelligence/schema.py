"""
Schema Inference — classifies each column into a semantic type
and produces a structured SchemaReport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

import re
from typing import Literal

ColumnType = Literal[
    "numeric",
    "categorical",
    "datetime",
    "text",
    "boolean",
    "high_cardinality",
    "email",
    "ip_address",
    "url",
    "uuid",
    "phone_number",
    "spatial_coords",
    "unknown",
]

HIGH_CARDINALITY_RATIO = 0.80
TEXT_WORD_THRESHOLD = 5
DATE_NAME_HINTS = {
    "date", "time", "year", "month", "day", "dt", "timestamp", "ts",
    "created", "updated", "modified", "dob",
}

# Regex patterns for specialized semantic types
EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
IP_REGEX = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
URL_REGEX = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
UUID_REGEX = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
PHONE_REGEX = re.compile(r"^\+?[0-9]{1,4}?[-.\s]?\(?[0-9]{1,3}?\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}$")

ID_COLUMN_REGEX = re.compile(
    r"^(id|uuid|index|row_?num|row_?id|unnamed:\s*0)$"
    r"|.+(_|-|\s)id$"
    r"|.+[a-z0-9](Id|ID)$",
    re.IGNORECASE,
)


def is_id_column(col_name: str, target_col: str | None = None) -> bool:
    """Check if a column is a non-predictive identifier column (e.g. Id, user_id, PassengerId)."""
    if target_col and col_name == target_col:
        return False
    return bool(ID_COLUMN_REGEX.match(col_name.strip()))


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


def _check_regex_type(sample: list[str]) -> ColumnType | None:
    if not sample:
        return None
    if all(EMAIL_REGEX.match(s) for s in sample):
        return "email"
    if all(IP_REGEX.match(s) for s in sample):
        return "ip_address"
    if all(URL_REGEX.match(s) for s in sample):
        return "url"
    if all(UUID_REGEX.match(s) for s in sample):
        return "uuid"
    if len(sample) >= 3 and all(PHONE_REGEX.match(s) for s in sample):
        return "phone_number"
    return None


def _infer_column_type(series: pd.Series, col_name: str) -> ColumnType:
    """
    Infer semantic column type with safe precedence:
      bool → numeric/spatial → object-type (date hint → regex → datetime → categorical → text)
    """
    n = len(series)
    if n == 0:
        return "unknown"

    dtype = series.dtype

    # ── 1. Already parsed as datetime by pandas ──────────────────────────
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    # ── 2. Boolean ───────────────────────────────────────────────────────
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    n_unique = series.nunique(dropna=True)
    if n_unique <= 2 and pd.api.types.is_integer_dtype(dtype):
        return "boolean"

    # ── 3. Numeric (int / float) — checked BEFORE datetime ───────────────
    col_lower = col_name.lower()
    if pd.api.types.is_numeric_dtype(dtype):
        # Spatial coordinate hint
        if col_lower in ("lat", "latitude", "y") or col_lower in ("lon", "lng", "longitude", "x"):
            clean = series.dropna()
            if len(clean) > 0:
                min_v, max_v = clean.min(), clean.max()
                if (col_lower in ("lat", "latitude") and -90 <= min_v and max_v <= 90) or \
                   (col_lower in ("lon", "lng", "longitude") and -180 <= min_v and max_v <= 180):
                    return "spatial_coords"

        return "numeric"

    # ── 4. Object / string / categorical columns ──────────────────────────
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) or str(dtype) == "category":
        # 4a. Datetime — checked first if date-like name hint exists
        if _has_date_name_hint(col_name):
            try:
                pd.to_datetime(series.dropna().head(50))
                return "datetime"
            except Exception:
                pass

        clean_str_sample = series.dropna().astype(str).head(30).tolist()

        # 4b. Regex-based semantic types (email, IP, URL, UUID, phone)
        regex_type = _check_regex_type(clean_str_sample)
        if regex_type:
            return regex_type

        # 4c. Cardinality-based classification
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
