"""
Cleaner — automated dataset cleaning engine.
Applies IQR outlier diagnostics/Winsorization, reports zero-variance and high-VIF collinear features,
and configurable anomaly handling (report / flag / remove).

Conservative defaults:
  - VIF columns are REPORTED, not dropped (drop_high_vif=False).
  - Outliers are REPORTED, not modified (outlier_action="report").
  - Anomalies are REPORTED, not removed (anomaly_action="report").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from atlas_cli.agents.dataset_intelligence.loader import load_dataset
from atlas_cli.agents.dataset_intelligence.profiler import profile_dataset
from atlas_cli.agents.dataset_intelligence.schema import infer_schema, is_id_column

logger = logging.getLogger("atlas_cli")

AnomalyAction = Literal["report", "flag", "remove"]
OutlierAction = Literal["report", "cap"]

SMALL_DATASET_THRESHOLD = 500  # rows — below this, default to "report"


@dataclass
class AnomalyReport:
    """Detailed anomaly detection results (always computed, independent of action)."""
    isolation_forest_count: int = 0
    isolation_forest_pct: float = 0.0
    lof_count: int = 0
    lof_pct: float = 0.0
    consensus_count: int = 0  # flagged by BOTH IF and LOF
    consensus_pct: float = 0.0
    removed_indices: list[int] = field(default_factory=list)
    action_taken: str = "report"


@dataclass
class CleanReport:
    original_rows: int
    cleaned_rows: int
    rows_removed: int
    original_cols: int
    cleaned_cols: int
    cols_dropped: list[str] = field(default_factory=list)
    outlier_diagnostics: list[str] = field(default_factory=list)
    capped_columns: list[str] = field(default_factory=list)
    vif_diagnostics: list[str] = field(default_factory=list)
    anomaly_report: Optional[AnomalyReport] = None
    outlier_action: str = "report"
    anomaly_action: str = "report"
    cleaned_file_path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "original_rows": self.original_rows,
            "cleaned_rows": self.cleaned_rows,
            "rows_removed": self.rows_removed,
            "original_cols": self.original_cols,
            "cleaned_cols": self.cleaned_cols,
            "cols_dropped": self.cols_dropped,
            "outlier_diagnostics": self.outlier_diagnostics,
            "capped_columns": self.capped_columns,
            "vif_diagnostics": self.vif_diagnostics,
            "outlier_action": self.outlier_action,
            "anomaly_action": self.anomaly_action,
            "anomaly_report": {
                "isolation_forest_count": self.anomaly_report.isolation_forest_count,
                "isolation_forest_pct": round(self.anomaly_report.isolation_forest_pct, 2),
                "lof_count": self.anomaly_report.lof_count,
                "lof_pct": round(self.anomaly_report.lof_pct, 2),
                "consensus_count": self.anomaly_report.consensus_count,
                "consensus_pct": round(self.anomaly_report.consensus_pct, 2),
                "removed_indices": self.anomaly_report.removed_indices,
                "action_taken": self.anomaly_report.action_taken,
            } if self.anomaly_report else None,
            "cleaned_file_path": str(self.cleaned_file_path) if self.cleaned_file_path else None,
        }


def clean_dataset(
    file_path: Path,
    target_col: Optional[str] = None,
    output_dir: Optional[Path] = None,
    ignore_cols: Optional[list[str]] = None,
    outlier_action: OutlierAction = "report",
    drop_high_vif: bool = False,
    anomaly_action: AnomalyAction = "report",
    # Legacy flags for backwards compatibility
    cap_outliers: bool = False,
    filter_anomalies: bool = False,
) -> tuple[pd.DataFrame, CleanReport]:
    """
    Apply professional data cleaning transformations on a dataset.

    Args:
        file_path: Path to raw dataset file.
        target_col: Optional target column (protected from removal/capping).
        output_dir: Optional directory to export cleaned CSV file.
        ignore_cols: Optional list of column names to ignore/drop.
        outlier_action: "report" (diagnostic only, values preserved) or "cap" (IQR Winsorization).
        drop_high_vif: Report-only by default; if True, drop VIF >= 10 columns.
        anomaly_action: "report" (log only), "flag" (add column), "remove" (delete rows).
        cap_outliers: Legacy flag — if True, sets outlier_action="cap".
        filter_anomalies: Legacy flag — if True, sets anomaly_action="remove".

    Returns:
        Tuple of (Cleaned pandas DataFrame, CleanReport).
    """
    # Legacy compatibility
    if cap_outliers and outlier_action == "report":
        outlier_action = "cap"
    if filter_anomalies and anomaly_action == "report":
        anomaly_action = "remove"

    df, meta = load_dataset(file_path)
    schema = infer_schema(df)
    profile = profile_dataset(df, target_col=target_col)

    orig_rows, orig_cols = df.shape
    df_clean = df.copy()
    cols_dropped: list[str] = []
    outlier_diagnostics: list[str] = []
    capped_cols: list[str] = []
    vif_diagnostics: list[str] = []

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

    # 2. High-VIF Collinear Columns — DIAGNOSTIC by default
    if profile.vif_metrics:
        for vif_item in profile.vif_metrics:
            if vif_item.column == target_col:
                continue
            if vif_item.vif >= 10.0:
                diag_msg = f"{vif_item.column} (VIF={vif_item.vif:.1f})"
                vif_diagnostics.append(diag_msg)
                if drop_high_vif and vif_item.column in df_clean.columns:
                    df_clean = df_clean.drop(columns=[vif_item.column], errors="ignore")
                    cols_dropped.append(f"{vif_item.column} (VIF={vif_item.vif:.1f})")
                    logger.info(f"Dropped high-VIF column: {diag_msg}")
                else:
                    logger.info(
                        f"High VIF detected (diagnostic only, not dropping): {diag_msg}. "
                        "Tree-based models handle collinearity well."
                    )

    # 3. IQR Outlier Analysis & Optional Capping
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
            diag = f"{col} ({outliers} outliers, {outlier_pct:.1f}% outside [{lower_bound:.2f}, {upper_bound:.2f}])"
            outlier_diagnostics.append(diag)

            if outlier_action == "cap":
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                capped_cols.append(f"{col} ({outlier_pct:.1f}% capped)")
                logger.info(f"Capped outliers in {col}: {diag}")
            else:
                logger.info(f"Outlier diagnostic (values preserved): {diag}")

    # 4. Anomaly Detection — always detect, action is configurable
    anomaly_rpt = _detect_anomalies(df_clean, target_col, anomaly_action)
    if anomaly_rpt and anomaly_action == "remove" and anomaly_rpt.removed_indices:
        df_clean = df_clean.drop(index=anomaly_rpt.removed_indices).reset_index(drop=True)
        logger.info(
            f"Removed {len(anomaly_rpt.removed_indices)} consensus anomaly rows "
            f"(indices: {anomaly_rpt.removed_indices[:20]}{'...' if len(anomaly_rpt.removed_indices) > 20 else ''})"
        )
    elif anomaly_rpt and anomaly_action == "flag":
        flag_col = "_anomaly_flag"
        df_clean[flag_col] = 0
        valid_indices = [i for i in anomaly_rpt.removed_indices if i in df_clean.index]
        if valid_indices:
            df_clean.loc[valid_indices, flag_col] = 1
        logger.info(f"Flagged {len(valid_indices)} consensus anomaly rows (column: {flag_col})")

    final_rows, final_cols = df_clean.shape
    cleaned_file_path = None

    if output_dir:
        output_path = Path(output_dir)
        if output_path.suffix in {".csv", ".parquet"}:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_file_path = output_path
            if cleaned_file_path.suffix == ".parquet":
                df_clean.to_parquet(cleaned_file_path, index=False)
            else:
                df_clean.to_csv(cleaned_file_path, index=False)
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
        outlier_diagnostics=outlier_diagnostics,
        capped_columns=capped_cols,
        vif_diagnostics=vif_diagnostics,
        anomaly_report=anomaly_rpt,
        outlier_action=outlier_action,
        anomaly_action=anomaly_action,
        cleaned_file_path=cleaned_file_path,
    )

    return df_clean, report


def _detect_anomalies(
    df: pd.DataFrame,
    target_col: Optional[str],
    action: AnomalyAction,
) -> Optional[AnomalyReport]:
    """
    Run Isolation Forest + LOF anomaly detection. Always report counts.
    Only populate removed_indices when action != "report".

    For "remove" action, only remove rows flagged by BOTH detectors (consensus)
    to reduce false positives.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    valid_cols = [c for c in numeric_df.columns if c != target_col and numeric_df[c].std() > 0]

    if len(valid_cols) < 2 or len(df) < 20:
        return None

    clean_X = numeric_df[valid_cols].fillna(numeric_df[valid_cols].median())
    rpt = AnomalyReport(action_taken=action)

    try:
        if_clf = IsolationForest(random_state=42, contamination="auto", n_estimators=50)
        if_preds = if_clf.fit_predict(clean_X)
        if_mask = if_preds == -1
        rpt.isolation_forest_count = int(if_mask.sum())
        rpt.isolation_forest_pct = float(rpt.isolation_forest_count / len(df) * 100)

        lof_clf = LocalOutlierFactor(
            n_neighbors=min(20, len(clean_X) - 1), contamination="auto"
        )
        lof_preds = lof_clf.fit_predict(clean_X)
        lof_mask = lof_preds == -1
        rpt.lof_count = int(lof_mask.sum())
        rpt.lof_pct = float(rpt.lof_count / len(df) * 100)

        consensus_mask = if_mask & lof_mask
        rpt.consensus_count = int(consensus_mask.sum())
        rpt.consensus_pct = float(rpt.consensus_count / len(df) * 100)

        logger.info(
            f"Anomaly detection: IF={rpt.isolation_forest_count} ({rpt.isolation_forest_pct:.1f}%), "
            f"LOF={rpt.lof_count} ({rpt.lof_pct:.1f}%), "
            f"Consensus (both)={rpt.consensus_count} ({rpt.consensus_pct:.1f}%)"
        )

        if action in ("remove", "flag") and rpt.consensus_count > 0:
            if rpt.consensus_pct < 5.0:
                rpt.removed_indices = list(df.index[consensus_mask])
            else:
                logger.warning(
                    f"Consensus anomalies ({rpt.consensus_pct:.1f}%) exceed 5% threshold — "
                    "skipping removal to avoid discarding legitimate observations."
                )

    except Exception as exc:
        logger.warning(f"Anomaly detection failed: {exc}")

    return rpt
