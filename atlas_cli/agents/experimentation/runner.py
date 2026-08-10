"""
Parallel Experimentation Runner — Phase 5 Orchestrator.

Ties together feature engineering, data splitting, parallel model training,
metric logging, and artifact persistence.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sqlmodel import Session

from atlas_cli.agents.experimentation.splitter import split_data
from atlas_cli.agents.experimentation.worker import (
    ExperimentResult,
    WorkerArgs,
    train_single_model,
)
from atlas_cli.agents.feature_engineering.runner import process_feature_engineering
from atlas_cli.agents.pipeline_planner.schemas import ExecutionPlan
from atlas_cli.core.config import settings
from atlas_cli.db.models import Experiment, MetricLog
from atlas_cli.db.session import create_db_and_tables, get_engine

logger = logging.getLogger("atlas_cli")


def _load_execution_plan(run_dir: Path) -> ExecutionPlan:
    """Load and validate the execution plan JSON."""
    plan_path = run_dir / "execution_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"No execution_plan.json found in {run_dir}. Run 'atlas plan' first."
        )
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return ExecutionPlan.model_validate(data)


def _persist_experiment(
    session: Session,
    run_id: str,
    result: ExperimentResult,
    artifact_path: Optional[str],
) -> str:
    """Write Experiment + MetricLog rows to SQLite and return the experiment ID."""
    experiment = Experiment(
        run_id=run_id,
        model_name=result.model_name,
        model_type=result.library,
        hyperparams_json=json.dumps(result.hyperparams, default=str),
        status=result.status,
        duration_seconds=result.duration_seconds,
        metrics_json=json.dumps(result.metrics),
        model_artifact_path=artifact_path,
        error_message=result.error_message,
    )
    session.add(experiment)
    session.flush()  

    for metric_name, metric_value in result.metrics.items():
        log = MetricLog(
            experiment_id=experiment.id,
            run_id=run_id,
            metric_name=metric_name,
            metric_value=metric_value,
            split_type="val",
        )
        session.add(log)

    return experiment.id


def run_experiments(
    run_id: str,
    *,
    max_workers: int = 4,
    file_path: Optional[Path] = None,
    random_seed: int = 42,
) -> list[ExperimentResult]:
    """
    Execute the full experimentation pipeline for a run:

      1. Load execution plan.
      2. Run Phase 4 feature engineering (if not already cached).
      3. Split data into train / val / test.
      4. Train all model candidates in parallel via ThreadPoolExecutor.
      5. Persist experiments, metrics, and model artifacts.
      6. Save experiment_results.json summary.

    Args:
        run_id: Run identifier (maps to .atlas_cli/runs/<run_id>/).
        max_workers: Maximum number of concurrent training threads.
        file_path: Optional dataset file path override.
        random_seed: Base random seed for reproducibility.

    Returns:
        List of ExperimentResult objects.
    """
    run_dir = settings.workspace_dir / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    plan = _load_execution_plan(run_dir)
    logger.info(f"Loaded execution plan: {plan.task_type}, {len(plan.model_candidates)} candidates")

    logger.info("Running feature engineering pipeline...")
    X, y, pipeline, feature_names = process_feature_engineering(run_id, file_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, plan, random_seed=random_seed)

    test_dir = run_dir / "splits"
    test_dir.mkdir(parents=True, exist_ok=True)
    np.save(test_dir / "X_test.npy", X_test)
    np.save(test_dir / "y_test.npy", y_test)
    np.save(test_dir / "X_train.npy", X_train)
    np.save(test_dir / "y_train.npy", y_train)
    np.save(test_dir / "X_val.npy", X_val)
    np.save(test_dir / "y_val.npy", y_val)

    worker_args_list: list[WorkerArgs] = []
    for candidate in sorted(plan.model_candidates, key=lambda m: m.priority):
        worker_args_list.append(WorkerArgs(
            candidate=candidate,
            task_type=plan.task_type,
            handle_imbalance=plan.evaluation.handle_imbalance,
            random_seed=random_seed,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        ))

    results: list[ExperimentResult] = []
    effective_workers = min(max_workers, len(worker_args_list))
    # Pre-import packages in main thread to prevent thread import lock contention and circular imports
    for _mod in ("polars", "sklearn", "lightgbm", "xgboost", "catboost", "joblib"):
        try:
            __import__(_mod)
        except ImportError:
            pass

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_name = {
            executor.submit(train_single_model, args): args.candidate.name
            for args in worker_args_list
        }
        for future in as_completed(future_to_name):
            model_name = future_to_name[future]
            try:
                result = future.result()
            except Exception as exc:
                result = ExperimentResult(
                    model_name=model_name,
                    library="unknown",
                    status="failed",
                    error_message=str(exc),
                )
            results.append(result)

    total_time = time.perf_counter() - t0
    logger.info(f"All experiments finished in {total_time:.2f}s")

    create_db_and_tables() 
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with Session(get_engine()) as session:
        for result in results:
            artifact_path: Optional[str] = None

            if result.status == "success" and result.estimator is not None:
                safe_name = result.model_name.lower().replace(" ", "_").replace("/", "_")
                model_path = models_dir / f"{safe_name}.joblib"
                joblib.dump(result.estimator, model_path)
                artifact_path = str(model_path)
                logger.info(f"Saved model artifact: {model_path}")

            _persist_experiment(session, run_id, result, artifact_path)

        session.commit()

    summary = {
        "run_id": run_id,
        "task_type": plan.task_type,
        "total_duration_seconds": round(total_time, 2),
        "num_candidates": len(results),
        "num_succeeded": sum(1 for r in results if r.status == "success"),
        "num_failed": sum(1 for r in results if r.status == "failed"),
        "primary_metric": plan.evaluation.primary_metric,
        "experiments": [
            {
                "model_name": r.model_name,
                "library": r.library,
                "status": r.status,
                "duration_seconds": round(r.duration_seconds, 2),
                "metrics": r.metrics,
                "train_metrics": r.train_metrics,
                "error": r.error_message[:200] if r.error_message else None,
            }
            for r in results
        ],
    }

    summary_path = run_dir / "experiment_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Experiment summary saved: {summary_path}")

    return results
