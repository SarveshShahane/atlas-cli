"""
Dataset Loader — supports CSV, Parquet, and JSON input formats.
Returns a loaded DataFrame and file-level metadata.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}


@dataclass
class FileMeta:
    """Metadata captured at load time about the raw dataset file."""
    file_name: str
    file_format: str
    file_size_mb: float
    num_rows: int
    num_cols: int
    dataset_hash: str


def _compute_hash(path: Path) -> str:
    """MD5 hash of raw file bytes for reproducibility fingerprinting."""
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_dataset(file_path: Path) -> tuple[pd.DataFrame, FileMeta]:
    """
    Load a dataset from CSV, Parquet, or JSON file.

    Args:
        file_path: Absolute or relative path to the dataset file.

    Returns:
        (df, meta) tuple where df is the loaded DataFrame and meta contains file-level metadata.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".csv":
        df = pd.read_csv(file_path, low_memory=False)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    elif ext in {".json", ".jsonl"}:
        try:
            df = pd.read_json(file_path)
        except ValueError:
            df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unhandled extension: {ext}")

    file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 4)
    dataset_hash = _compute_hash(file_path)

    meta = FileMeta(
        file_name=file_path.name,
        file_format=ext.lstrip(".").upper(),
        file_size_mb=file_size_mb,
        num_rows=len(df),
        num_cols=len(df.columns),
        dataset_hash=dataset_hash,
    )
    return df, meta
