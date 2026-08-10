"""
Snapshot Metadata Manager — Phase 10 Reproducibility Engine.

Captures dataset SHA256 hashes, Git commit hash, dependency versions,
random seeds, and pipeline parameters into reproducibility_snapshot.json.
Verifies snapshots during replay.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")

_KEY_DEPENDENCIES = [
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "pandas",
    "numpy",
    "joblib",
    "pydantic",
    "atlas-cli",
]


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA256 hex digest of a file."""
    path = Path(file_path)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash() -> str:
    """Return the current Git commit hash, or 'uncommitted/no-git'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            return res.stdout.strip()[:12]
    except Exception:
        pass
    return "no-git-repo"


def get_installed_dependencies() -> dict[str, str]:
    """Capture versions of key machine learning dependencies."""
    import importlib.metadata
    deps = {}
    for pkg in _KEY_DEPENDENCIES:
        try:
            deps[pkg] = importlib.metadata.version(pkg)
        except Exception:
            deps[pkg] = "not-installed"
    return deps


@dataclass
class SnapshotMetadata:
    """Reproducibility snapshot data structure."""

    run_id: str
    dataset_path: str
    dataset_sha256: str
    git_commit: str
    python_version: str
    os_info: str
    dependencies: dict[str, str] = field(default_factory=dict)
    random_seed: int = 42
    created_at: str = ""
    pipeline_config: dict[str, Any] = field(default_factory=dict)


def create_snapshot(
    run_id: str,
    dataset_path: str | Path,
    *,
    random_seed: int = 42,
    pipeline_config: dict[str, Any] | None = None,
) -> SnapshotMetadata:
    """
    Create and save a reproducibility snapshot for a run.

    Args:
        run_id: Run identifier.
        dataset_path: Path to dataset file.
        random_seed: Random seed used for splits and model training.
        pipeline_config: Optional pipeline planner configuration dict.

    Returns:
        SnapshotMetadata instance.
    """
    from datetime import datetime

    run_dir = settings.workspace_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot = SnapshotMetadata(
        run_id=run_id,
        dataset_path=str(dataset_path),
        dataset_sha256=compute_file_sha256(dataset_path),
        git_commit=get_git_commit_hash(),
        python_version=platform.python_version(),
        os_info=f"{platform.system()} {platform.release()}",
        dependencies=get_installed_dependencies(),
        random_seed=random_seed,
        created_at=datetime.utcnow().isoformat(),
        pipeline_config=pipeline_config or {},
    )

    data = {
        "run_id": snapshot.run_id,
        "dataset_path": snapshot.dataset_path,
        "dataset_sha256": snapshot.dataset_sha256,
        "git_commit": snapshot.git_commit,
        "python_version": snapshot.python_version,
        "os_info": snapshot.os_info,
        "dependencies": snapshot.dependencies,
        "random_seed": snapshot.random_seed,
        "created_at": snapshot.created_at,
        "pipeline_config": snapshot.pipeline_config,
    }

    out_path = run_dir / "reproducibility_snapshot.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Snapshot created: {out_path}")

    return snapshot


def load_snapshot(run_id: str) -> Optional[SnapshotMetadata]:
    """
    Load saved snapshot metadata for a run.

    Returns:
        SnapshotMetadata or None if missing.
    """
    run_dir = settings.workspace_dir / "runs" / run_id
    snapshot_path = run_dir / "reproducibility_snapshot.json"

    if not snapshot_path.exists():
        return None

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return SnapshotMetadata(
            run_id=data.get("run_id", run_id),
            dataset_path=data.get("dataset_path", ""),
            dataset_sha256=data.get("dataset_sha256", ""),
            git_commit=data.get("git_commit", "unknown"),
            python_version=data.get("python_version", ""),
            os_info=data.get("os_info", ""),
            dependencies=data.get("dependencies", {}),
            random_seed=data.get("random_seed", 42),
            created_at=data.get("created_at", ""),
            pipeline_config=data.get("pipeline_config", {}),
        )
    except Exception as exc:
        logger.warning(f"Failed to load snapshot for run {run_id}: {exc}")
        return None


def verify_snapshot_integrity(snapshot: SnapshotMetadata) -> tuple[bool, list[str]]:
    """
    Verify if the target dataset exists and matches the original SHA256 digest.

    Returns:
        Tuple of (is_valid, list_of_warning_messages).
    """
    warnings = []
    is_valid = True

    # Verify dataset existence and hash
    ds_path = Path(snapshot.dataset_path)
    if not ds_path.exists():
        is_valid = False
        warnings.append(f"Dataset file not found at original path: {snapshot.dataset_path}")
    else:
        current_hash = compute_file_sha256(ds_path)
        if snapshot.dataset_sha256 and current_hash != snapshot.dataset_sha256:
            is_valid = False
            warnings.append(
                f"Dataset hash mismatch! Original: {snapshot.dataset_sha256[:12]}..., "
                f"Current: {current_hash[:12]}..."
            )

    # Check python version drift
    curr_py = platform.python_version()
    if snapshot.python_version and curr_py != snapshot.python_version:
        warnings.append(f"Python version drift: run was recorded on {snapshot.python_version}, current is {curr_py}.")

    return is_valid, warnings
