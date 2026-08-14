"""
Cleaner — automated dataset cleaning engine.
Applies IQR Winsorization, drops zero-variance and high-VIF collinear features,
and filters extreme noise rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from atlas_cli.agents.dataset_intelligence.loader import load_dataset
from atlas_cli.agents.dataset_intelligence.profiler import profile_dataset
from atlas_cli.agents.dataset_intelligence.schema import infer_schema, is_id_column


@dataclass
class CleanReport:
    original_rows: int
    cleaned_rows: int
    rows_removed: int
    original_cols: int
    cleaned_cols: int
    cols_dropped: list[str] = field(default_factory=list)
    capped_columns: list[str] = field(default_factory=list)
    cleaned_file_path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "original_rows": self.original_rows,
            "cleaned_rows": self.cleaned_rows,
            "rows_removed": self.rows_removed,
            "original_cols": self.original_cols,
            "cleaned_cols": self.cleaned_cols,
            "cols_dropped": self.cols_dropped,
            "capped_columns": self.capped_columns,
            "cleaned_file_path": str(self.cleaned_file_path) if self.cleaned_file_path else None,
        }


def clean_dataset(
    file_path: Path,
    target_col: Optional[str] = None,
    output_dir: Optional[Path] = None,
    ignore_cols: Optional[list[str]] = None,
    cap_outliers: bool = True,
    drop_high_vif: bool = True,
    filter_anomalies: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    """
    Apply professional data cleaning transformations on a dataset.

    Args:
        file_path: Path to raw dataset file.
        target_col: Optional target column (protected from removal/capping).
        output_dir: Optional directory to export cleaned CSV file.
        ignore_cols: Optional list of column names to ignore/drop.
        cap_outliers: Enable IQR Winsorization/capping.
        drop_high_vif: Enable dropping high-VIF collinear columns.
        filter_anomalies: Filter extreme Isolation Forest anomaly rows (<10%).

    Returns:
        Tuple of (Cleaned pandas DataFrame, CleanReport).
    """
    df, meta = load_dataset(file_path)
    schema = infer_schema(df)
    profile = profile_dataset(df, target_col=target_col)

    orig_rows, orig_cols = df.shape
    df_clean = df.copy()
    cols_dropped = []
    capped_cols = []

    # 0. Drop Explicitly Ignored & Non-predictive ID Columns
    ignore_set = set(ignore_cols or [])
    for col in list(df_clean.columns):
        if col == target_col:
            continue
        if col in ignore_set or is_id_column(col, target_col):
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col], errors="ignore")
                reason = "Explicitly Ignored" if col in ignore_set else "Non-predictive ID Column"
                cols_dropped.append(f"{col} ({reason})")

    # 1. Drop Zero-Variance Columns
    for cp in profile.columns:
        if cp.name == target_col or cp.name not in df_clean.columns:
            continue
        if cp.zero_variance:
            df_clean = df_clean.drop(columns=[cp.name], errors="ignore")
            cols_dropped.append(f"{cp.name} (Zero Variance)")

    # 2. Drop High-VIF Collinear Columns (VIF >= 10.0)
    if drop_high_vif and profile.vif_metrics:
        for vif_item in profile.vif_metrics:
            if vif_item.column == target_col:
                continue
            if vif_item.vif >= 10.0 and vif_item.column in df_clean.columns:
                df_clean = df_clean.drop(columns=[vif_item.column], errors="ignore")
                cols_dropped.append(f"{vif_item.column} (VIF={vif_item.vif:.1f})")

    # 3. IQR Outlier Winsorization / Capping
    if cap_outliers:
        numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != target_col]
        for col in numeric_cols:
            series = df_clean[col].dropna()
            if len(series) < 10:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outlier_pct = outliers / len(series) * 100

            if outlier_pct > 0.5:
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                capped_cols.append(f"{col} ({outlier_pct:.1f}% capped)")

    # 4. Filter Extreme Anomalies via Isolation Forest
    if filter_anomalies:
        numeric_df = df_clean.select_dtypes(include=[np.number])
        valid_cols = [c for c in numeric_df.columns if c != target_col and numeric_df[c].std() > 0]
        if len(valid_cols) >= 2 and len(df_clean) >= 20:
            clean_X = numeric_df[valid_cols].fillna(numeric_df[valid_cols].median())
            try:
                clf = IsolationForest(random_state=42, contamination=0.05, n_estimators=50)
                preds = clf.fit_predict(clean_X)
                anomalies = (preds == -1).sum()
                if anomalies > 0 and (anomalies / len(df_clean)) < 0.10:
                    df_clean = df_clean[preds != -1].reset_index(drop=True)
            except Exception:
                pass

    final_rows, final_cols = df_clean.shape
    cleaned_file_path = None

    if output_dir:
        output_path = Path(output_dir)
        if output_path.suffix in {".csv", ".parquet"}:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_file_path = output_path
            df_clean.to_csv(cleaned_file_path, index=False) if cleaned_file_path.suffix != ".parquet" else df_clean.to_parquet(cleaned_file_path, index=False)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            cleaned_file_path = output_path / "cleaned_data.csv"
            cleaned_sub_dir = output_path / "cleaned"
            cleaned_sub_dir.mkdir(parents=True, exist_ok=True)
            sub_file_path = cleaned_sub_dir / "cleaned_data.csv"

            if cleaned_file_path.suffix == ".parquet":
                df_clean.to_parquet(cleaned_file_path, index=False)
                df_clean.to_parquet(sub_file_path, index=False)
            else:
                df_clean.to_csv(cleaned_file_path, index=False)
                df_clean.to_csv(sub_file_path, index=False)

    report = CleanReport(
        original_rows=orig_rows,
        cleaned_rows=final_rows,
        rows_removed=orig_rows - final_rows,
        original_cols=orig_cols,
        cleaned_cols=final_cols,
        cols_dropped=cols_dropped,
        capped_columns=capped_cols,
        cleaned_file_path=cleaned_file_path,
    )

    return df_clean, report
