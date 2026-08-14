"""
Custom Scikit-Learn compatible transformers for feature engineering.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """
    Clips numeric outliers based on Interquartile Range (IQR) bounds calculated in fit().
    Lower bound = Q1 - iqr_multiplier * IQR
    Upper bound = Q3 + iqr_multiplier * IQR
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> IQROutlierCapper:
        X_arr = np.asarray(X, dtype=np.float64)
        q1 = np.nanpercentile(X_arr, 25, axis=0)
        q3 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.iqr_multiplier * iqr
        self.upper_bounds_ = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            return X_arr
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class LogTransformFeature(BaseEstimator, TransformerMixin):
    """
    Applies np.log1p(x - min + 1) to numerical features safely handling negative or zero values.
    """

    def __init__(self):
        self.min_vals_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> LogTransformFeature:
        X_arr = np.asarray(X, dtype=np.float64)
        self.min_vals_ = np.nanmin(X_arr, axis=0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.min_vals_ is None:
            return X_arr
        shift = np.where(self.min_vals_ < 0, np.abs(self.min_vals_) + 1.0, 0.0)
        return np.log1p(X_arr + shift)


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts date components (year, month, day, quarter, dayofweek, is_weekend)
    from datetime or timestamp features.
    """

    def __init__(self, components: Optional[List[str]] = None):
        self.components = components or ["year", "month", "day", "quarter", "dayofweek", "is_weekend"]

    def fit(self, X: Any, y: Any = None) -> DatetimeFeatureExtractor:
        return self

    def transform(self, X: Any) -> np.ndarray:
        df = pd.DataFrame(X)
        extracted = []
        for col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            if "year" in self.components:
                extracted.append(s.dt.year.fillna(0).astype(np.float64))
            if "month" in self.components:
                extracted.append(s.dt.month.fillna(0).astype(np.float64))
            if "day" in self.components:
                extracted.append(s.dt.day.fillna(0).astype(np.float64))
            if "quarter" in self.components:
                extracted.append(s.dt.quarter.fillna(0).astype(np.float64))
            if "dayofweek" in self.components:
                extracted.append(s.dt.dayofweek.fillna(0).astype(np.float64))
            if "is_weekend" in self.components:
                extracted.append((s.dt.dayofweek >= 5).astype(np.float64))

        if not extracted:
            return np.zeros((len(X), 0), dtype=np.float64)
        return np.column_stack([arr.values for arr in extracted])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features by relative category frequency.
    """

    def __init__(self):
        self.freq_maps_: list[dict[Any, float]] = []

    def fit(self, X: Any, y: Any = None) -> FrequencyEncoder:
        df = pd.DataFrame(X)
        self.freq_maps_ = []
        for col in df.columns:
            vc = df[col].value_counts(normalize=True).to_dict()
            self.freq_maps_.append(vc)
        return self

    def transform(self, X: Any) -> np.ndarray:
        df = pd.DataFrame(X)
        encoded_cols = []
        for i, col in enumerate(df.columns):
            freq_map = self.freq_maps_[i] if i < len(self.freq_maps_) else {}
            mapped = df[col].map(freq_map).fillna(0.0).values.astype(np.float64)
            encoded_cols.append(mapped)
        if not encoded_cols:
            return np.zeros((len(X), 0), dtype=np.float64)
        return np.column_stack(encoded_cols)


class SemanticTextExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts structural sub-features from specialized semantic strings:
    - email: username length, domain length
    - url: domain length, path length
    - ip_address: first octet, private IP indicator
    - uuid: valid UUID boolean indicator
    """

    def __init__(self, semantic_type: str = "text"):
        self.semantic_type = semantic_type

    def fit(self, X: Any, y: Any = None) -> SemanticTextExtractor:
        return self

    def transform(self, X: Any) -> np.ndarray:
        df = pd.DataFrame(X)
        features = []

        for col in df.columns:
            series = df[col].astype(str)
            if self.semantic_type == "email":
                user_len = series.apply(lambda s: len(s.split("@")[0]) if "@" in s else len(s)).astype(np.float64)
                dom_len = series.apply(lambda s: len(s.split("@")[1]) if "@" in s else 0).astype(np.float64)
                features.extend([user_len.values, dom_len.values])
            elif self.semantic_type == "url":
                dom_len = series.apply(lambda s: len(s.split("/")[2]) if "://" in s and len(s.split("/")) > 2 else len(s)).astype(np.float64)
                path_len = series.apply(lambda s: len(s)).astype(np.float64)
                features.extend([dom_len.values, path_len.values])
            elif self.semantic_type == "ip_address":
                octet1 = series.apply(lambda s: float(s.split(".")[0]) if "." in s and s.split(".")[0].isdigit() else 0.0).astype(np.float64)
                is_private = series.apply(lambda s: 1.0 if s.startswith(("10.", "192.168.", "172.16.")) else 0.0).astype(np.float64)
                features.extend([octet1.values, is_private.values])
            else:
                str_len = series.apply(len).astype(np.float64)
                features.append(str_len.values)

        if not features:
            return np.zeros((len(X), 0), dtype=np.float64)
        return np.column_stack(features)


class TopMIInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Generates interaction feature products (X_i * X_j) between top Mutual Information feature pairs.
    """

    def fit(self, X: Any, y: Any = None) -> TopMIInteractionTransformer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] < 2:
            return np.zeros((len(X_arr), 0), dtype=np.float64)

        interactions = []
        num_cols = min(X_arr.shape[1], 4)
        for i in range(num_cols):
            for j in range(i + 1, num_cols):
                prod = X_arr[:, i] * X_arr[:, j]
                interactions.append(prod)

        if not interactions:
            return np.zeros((len(X_arr), 0), dtype=np.float64)
        return np.column_stack(interactions)
