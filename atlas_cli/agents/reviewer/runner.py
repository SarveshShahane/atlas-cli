"""
AI Reviewer & Auto-Critique Loop Orchestrator — Phase 7.

Executes a 1-pass critique-and-refine retry experiment for the top model candidate:
  1. Inspects initial model training & validation metrics.
  2. Runs automated diagnostic audit (overfitting, underfitting, imbalance).
  3. Generates LLM / rule-based RefinementPlan with regularized parameters.
  4. Retrains one refined model candidate on the training set.
  5. Computes updated metrics on val & test splits.
  6. Compares initial vs refined performance and logs CritiqueReport.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sqlmodel import Session

from atlas_cli.agents.comparator.scorer import compute_extended_metrics
from atlas_cli.agents.experimentation.metrics import compute_metrics
from atlas_cli.agents.experimentation.worker import (
    ExperimentResult,
    WorkerArgs,
    train_single_model,
)
from atlas_cli.agents.pipeline_planner.schemas import ModelCandidate
from atlas_cli.agents.reviewer.diagnostics import diagnose_model
from atlas_cli.agents.reviewer.llm_critique import generate_critique
from atlas_cli.agents.reviewer.schemas import (
    CritiqueReport,
    RefinedExperimentComparison,
)
from atlas_cli.core.config import settings
from atlas_cli.db.models import Experiment, MetricLog
from atlas_cli.db.session import create_db_and_tables, get_engine
from atlas_cli.models.wrappers import resolve_estimator

logger = logging.getLogger("atlas_cli")


def _persist_refined_experiment(
    session: Session,
    run_id: str,
    result: ExperimentResult,
    artifact_path: Optional[str],
) -> str:
    """Write refined Experiment + MetricLog rows to SQLite."""
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
            split_type="val_refined",
        )
        session.add(log)

    return experiment.id


def review_and_refine(
    run_id: str,
    *,
    random_seed: int = 42,
    model_override: str | None = None,
) -> CritiqueReport:
    """
    Execute 1-pass AI Reviewer critique & refinement retry experiment.

    Args:
        run_id: The run identifier (maps to .atlas_cli/runs/<run_id>/).
        random_seed: Seed for training reproducibility.
        model_override: Optional LLM model override for reflection.

    Returns:
        CritiqueReport summarizing diagnosis, refinement plan, and final metrics.
    """
    run_dir = settings.workspace_dir / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # 1. Load execution plan
    plan_path = run_dir / "execution_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"No execution_plan.json in run {run_id}.")
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    task_type = plan_data.get("task_type", "binary_classification")
    primary_metric = plan_data.get("evaluation", {}).get("primary_metric", "accuracy")

    # 2. Load experiment results
    exp_path = run_dir / "experiment_results.json"
    if not exp_path.exists():
        raise FileNotFoundError(f"No experiment_results.json in run {run_id}.")
    exp_data = json.loads(exp_path.read_text(encoding="utf-8"))
    experiments = exp_data.get("experiments", [])

    successful = [e for e in experiments if e.get("status") == "success"]
    if not successful:
        raise ValueError(f"No successful experiments found in run {run_id} to review.")

    # Select target candidate (use comparison winner if present, else highest primary metric candidate)
    comp_path = run_dir / "comparison_results.json"
    target_entry = None
    if comp_path.exists():
        comp_data = json.loads(comp_path.read_text(encoding="utf-8"))
        winner_info = comp_data.get("winner")
        if winner_info:
            target_entry = next(
                (e for e in successful if e.get("model_name") == winner_info["model_name"]),
                None,
            )

    if not target_entry:
        target_entry = max(
            successful,
            key=lambda e: e.get("metrics", {}).get(primary_metric, 0.0),
        )

    model_name = target_entry["model_name"]
    library = target_entry["library"]
    val_metrics = target_entry.get("metrics", {})
    train_metrics = target_entry.get("train_metrics", {})

    logger.info(f"Targeting model '{model_name}' ({library}) for AI review...")

    # 3. Load splits
    splits_dir = run_dir / "splits"
    X_train = np.load(splits_dir / "X_train.npy")
    y_train = np.load(splits_dir / "y_train.npy")
    X_val = np.load(splits_dir / "X_val.npy")
    y_val = np.load(splits_dir / "y_val.npy")
    X_test = np.load(splits_dir / "X_test.npy")
    y_test = np.load(splits_dir / "y_test.npy")

    # If train_metrics were not computed previously, compute them now
    if not train_metrics:
        safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
        model_path = run_dir / "models" / f"{safe_name}.joblib"
        if model_path.exists():
            estimator = joblib.load(model_path)
            y_train_pred = estimator.predict(X_train)
            train_metrics = compute_metrics(
                y_train, y_train_pred, task_type=task_type, estimator=estimator, X_val=X_train
            )

    # Compute initial test metrics
    initial_em = compute_extended_metrics(run_id, target_entry, task_type=task_type, primary_metric=primary_metric)
    initial_test_metrics = initial_em.test_metrics if initial_em else {}

    # 4. Diagnose model
    diagnosis = diagnose_model(
        model_name,
        library,
        task_type=task_type,
        primary_metric=primary_metric,
        val_metrics=val_metrics,
        train_metrics=train_metrics,
    )

    # 5. Generate critique & refinement plan
    refinement_plan = generate_critique(
        model_name,
        library,
        task_type=task_type,
        primary_metric=primary_metric,
        val_metrics=val_metrics,
        train_metrics=train_metrics,
        diagnosis=diagnosis,
        model_override=model_override,
    )

    # 6. Execute Refined Retry Candidate Training
    refined_candidate = ModelCandidate(
        name=f"{model_name} (Refined)",
        library=library,
        rationale=f"Auto-critique regularized refinement: {refinement_plan.proposed_adjustments}",
        priority=1,
    )

    logger.info(f"Retraining refined model with params: {refinement_plan.refined_hyperparams}")

    t0 = time.perf_counter()
    try:
        # Resolve estimator with refined hyperparams
        estimator = resolve_estimator(
            refined_candidate,
            task_type=task_type,
            handle_imbalance=plan_data.get("evaluation", {}).get("handle_imbalance", False),
            random_seed=random_seed + 10,
            extra_params=refinement_plan.refined_hyperparams,
        )

        estimator.fit(X_train, y_train)
        duration_s = time.perf_counter() - t0

        # Compute metrics for refined model
        y_val_pred = estimator.predict(X_val)
        refined_val_metrics = compute_metrics(
            y_val, y_val_pred, task_type=task_type, estimator=estimator, X_val=X_val
        )

        y_train_pred = estimator.predict(X_train)
        refined_train_metrics = compute_metrics(
            y_train, y_train_pred, task_type=task_type, estimator=estimator, X_val=X_train
        )

        y_test_pred = estimator.predict(X_test)
        refined_test_metrics = compute_metrics(
            y_test, y_test_pred, task_type=task_type, estimator=estimator, X_val=X_test
        )

        refined_result = ExperimentResult(
            model_name=f"{model_name} (Refined)",
            library=library,
            status="success",
            metrics=refined_val_metrics,
            train_metrics=refined_train_metrics,
            duration_seconds=duration_s,
            estimator=estimator,
            hyperparams=refinement_plan.refined_hyperparams,
        )

    except Exception as exc:
        logger.error(f"Refined model training failed: {exc}")
        refined_result = ExperimentResult(
            model_name=f"{model_name} (Refined)",
            library=library,
            status="failed",
            error_message=str(exc),
        )
        refined_val_metrics = {}
        refined_train_metrics = {}
        refined_test_metrics = {}

    # 7. Persist refined model artifact & database entry
    if refined_result.status == "success" and refined_result.estimator is not None:
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
        refined_model_path = models_dir / f"{safe_name}_refined.joblib"
        joblib.dump(refined_result.estimator, refined_model_path)

        create_db_and_tables()
        with Session(get_engine()) as session:
            _persist_refined_experiment(session, run_id, refined_result, str(refined_model_path))
            session.commit()

    # 8. Compare initial vs refined
    init_train_val_gap = train_metrics.get(primary_metric, 0.0) - val_metrics.get(primary_metric, 0.0)
    ref_train_val_gap = refined_train_metrics.get(primary_metric, 0.0) - refined_val_metrics.get(primary_metric, 0.0)
    gap_reduced = init_train_val_gap - ref_train_val_gap

    init_test = initial_test_metrics.get(primary_metric, 0.0)
    ref_test = refined_test_metrics.get(primary_metric, 0.0)
    test_improved = ref_test >= init_test

    if gap_reduced > 0.02 and test_improved:
        verdict = (
            f"REFINE SUCCESS: Overfitting gap reduced by {gap_reduced:.4f} "
            f"while maintaining/improving test {primary_metric.upper()} ({init_test:.4f} → {ref_test:.4f})."
        )
        successful_refinement = True
    elif test_improved:
        verdict = (
            f"REFINE STABLE: Test {primary_metric.upper()} improved/maintained "
            f"({init_test:.4f} → {ref_test:.4f})."
        )
        successful_refinement = True
    else:
        verdict = (
            f"REFINE NEUTRAL: Initial model remains optimal; regularized model "
            f"test {primary_metric.upper()} ({ref_test:.4f}) did not exceed initial ({init_test:.4f})."
        )
        successful_refinement = False

    comparison = RefinedExperimentComparison(
        model_name=model_name,
        primary_metric_name=primary_metric,
        initial_train_metric=round(train_metrics.get(primary_metric, 0.0), 4),
        initial_val_metric=round(val_metrics.get(primary_metric, 0.0), 4),
        initial_test_metric=round(init_test, 4),
        refined_train_metric=round(refined_train_metrics.get(primary_metric, 0.0), 4),
        refined_val_metric=round(refined_val_metrics.get(primary_metric, 0.0), 4),
        refined_test_metric=round(ref_test, 4),
        gap_reduced=round(gap_reduced, 4),
        test_improved=test_improved,
        verdict=verdict,
    )

    # 9. Assemble report
    report = CritiqueReport(
        run_id=run_id,
        task_type=task_type,
        primary_metric=primary_metric,
        initial_winner_name=model_name,
        diagnosis=diagnosis,
        refinement_plan=refinement_plan,
        comparison=comparison,
        refinement_successful=successful_refinement,
        opt_rationale=f"{refinement_plan.critique_summary} {verdict}",
    )

    # Promote refined model if it improved test performance
    if successful_refinement and ref_test > init_test and refined_result.estimator is not None:
        try:
            primary_model_path = models_dir / f"{safe_name}.joblib"
            joblib.dump(refined_result.estimator, primary_model_path)
            logger.info(f"Promoted refined model as winner: {primary_model_path}")

            # Sync comparison_results.json winner test metrics
            comparison_path = run_dir / "comparison_results.json"
            if comparison_path.exists():
                comp_data = json.loads(comparison_path.read_text(encoding="utf-8"))
                if comp_data.get("winner") and comp_data["winner"].get("model_name") == model_name:
                    comp_data["winner"]["primary_metric_test"] = ref_test
                    comp_data["winner"]["val_metrics"] = refined_val_metrics
                    comparison_path.write_text(json.dumps(comp_data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not promote refined model artifact: {exc}")

    # 10. Save critique_report.json
    critique_path = run_dir / "critique_report.json"
    critique_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"Critique report saved: {critique_path}")

    return report
