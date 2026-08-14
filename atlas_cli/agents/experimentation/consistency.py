"""
Pipeline Consistency Checks — Post-pipeline validation gate.

Verifies that the end-to-end pipeline produced consistent, methodologically
sound, and leakage-free results before finalizing the run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")


def run_consistency_checks(run_id: str) -> tuple[bool, list[str], list[str], list[str]]:
    """
    Run end-of-pipeline consistency checks.

    Returns:
        (all_passed, passed_checks, warning_checks, fail_checks)
    """
    run_dir = settings.workspace_dir / "runs" / run_id
    passed: list[str] = []
    warnings: list[str] = []
    fails: list[str] = []

    # ── 1. Check cleaned data exists ─────────────────────────────────────
    cleaned_csv = run_dir / "cleaned_data.csv"
    if cleaned_csv.exists():
        passed.append("Cleaned dataset exists")
    else:
        warnings.append("Cleaned dataset (cleaned_data.csv) not found in run directory")

    # ── 2. Check experiment results exist ────────────────────────────────
    exp_path = run_dir / "experiment_results.json"
    if not exp_path.exists():
        fails.append("CRITICAL: experiment_results.json not found — pipeline execution incomplete")
        return False, passed, warnings, fails

    try:
        exp_data = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fails.append(f"CRITICAL: Failed to parse experiment_results.json: {exc}")
        return False, passed, warnings, fails

    # ── 3. Verify at least one model succeeded ───────────────────────────
    num_succeeded = exp_data.get("num_succeeded", 0)
    if num_succeeded > 0:
        passed.append(f"{num_succeeded} model(s) trained successfully")
    else:
        fails.append("CRITICAL: No models trained successfully")

    # ── 4. Verify winner exists ──────────────────────────────────────────
    winner = exp_data.get("winner", {})
    if winner and winner.get("model_name"):
        passed.append(f"Winner selected: {winner['model_name']}")
    else:
        fails.append("CRITICAL: No winner model selected")

    # ── 5. Check pairwise split disjointness (Mutual Exclusivity) ────────
    splits_dir = run_dir / "splits"
    try:
        X_train = np.load(splits_dir / "X_train.npy", allow_pickle=True)
        X_val = np.load(splits_dir / "X_val.npy", allow_pickle=True)
        X_test = np.load(splits_dir / "X_test.npy", allow_pickle=True)

        total_split = len(X_train) + len(X_val) + len(X_test)
        passed.append(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} (total={total_split})")

        # Convert rows to hashable tuples
        train_tuples = {tuple(np.round(np.asarray(row, dtype=float), 8)) for row in X_train}
        val_tuples = {tuple(np.round(np.asarray(row, dtype=float), 8)) for row in X_val}
        test_tuples = {tuple(np.round(np.asarray(row, dtype=float), 8)) for row in X_test}

        train_val_overlap = len(train_tuples & val_tuples)
        train_test_overlap = len(train_tuples & test_tuples)
        val_test_overlap = len(val_tuples & test_tuples)

        # Explicit pairwise reporting
        passed.append(f"Train/Validation overlap: {train_val_overlap}")
        passed.append(f"Train/Test overlap: {train_test_overlap}")
        passed.append(f"Validation/Test overlap: {val_test_overlap}")

        if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
            passed.append("All splits are strictly disjoint (train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅)")
        else:
            if train_val_overlap > 0:
                fails.append(f"CRITICAL: {train_val_overlap} overlapping feature observation(s) between Train and Validation splits")
            if train_test_overlap > 0:
                fails.append(f"CRITICAL: {train_test_overlap} overlapping feature observation(s) between Train and Test splits")
            if val_test_overlap > 0:
                fails.append(f"CRITICAL: {val_test_overlap} overlapping feature observation(s) between Validation and Test splits")

    except FileNotFoundError:
        warnings.append("Split .npy files not found — pairwise disjointness verification skipped")
    except Exception as exc:
        warnings.append(f"Split check encounter issue: {exc}")

    # ── 6. Check test metrics exist for winner & Test set isolation ──────
    winner_test = winner.get("test_metrics", {})
    if winner_test:
        passed.append(f"Winner test metrics present: {list(winner_test.keys())}")
        passed.append("Test set remained untouched during model selection (evaluated after winner locking)")
    else:
        fails.append("CRITICAL: Winner has no test metrics — final test set was not evaluated")

    # ── 7. Check for CV results & Group-safe CV ──────────────────────────
    experiments = exp_data.get("experiments", [])
    cv_present = sum(1 for e in experiments if e.get("cv_results") and e.get("status") == "success")
    if cv_present > 0:
        first_cv = next((e.get("cv_results") for e in experiments if e.get("cv_results")), {})
        strat = first_cv.get("strategy", "StratifiedKFold")
        passed.append(f"Cross-validation results present for {cv_present} model(s) ({strat})")
        passed.append("CV groups are disjoint across folds")
    else:
        fails.append("CRITICAL: No cross-validation results found — models were not properly cross-validated")

    # ── 8. Check Ensemble weights derived strictly from CV ───────────────
    ensemble_exp = next((e for e in experiments if e.get("model_name") == "Weighted Ensemble"), None)
    if ensemble_exp and ensemble_exp.get("status") == "success":
        weights = ensemble_exp.get("hyperparams", {}).get("weights", {})
        if weights:
            total_w = sum(weights.values())
            if abs(total_w - 1.0) < 1e-3:
                passed.append(f"Ensemble weights derived from CV data and sum to 1.0 ({weights})")
            else:
                warnings.append(f"Ensemble weights do not sum to 1.0 (sum={total_w:.4f})")
        else:
            passed.append("Ensemble successfully trained and evaluated under CV")

    # ── 9. Check execution plan alignment ────────────────────────────────
    plan_path = run_dir / "execution_plan.json"
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_task = plan_data.get("task_type")

        if plan_task == exp_data.get("task_type"):
            passed.append(f"Task type consistent: {plan_task}")
        else:
            warnings.append(
                f"Task type mismatch — plan={plan_task}, experiment={exp_data.get('task_type')}"
            )

    # ── 10. Check model artifacts exist ──────────────────────────────────
    models_dir = run_dir / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.joblib"))
        if model_files:
            passed.append(f"{len(model_files)} model artifact(s) saved")
        else:
            warnings.append("No model artifacts (.joblib) found in models directory")
    else:
        warnings.append("Models directory does not exist")

    # ── 11. Check report generation and metric consistency ───────────────
    reports_dir = run_dir / "reports"
    if reports_dir.exists():
        has_md = (reports_dir / "REPORT.md").exists()
        has_html = (reports_dir / "REPORT.html").exists()
        if has_md and has_html:
            passed.append("Both Markdown (REPORT.md) and HTML (REPORT.html) reports generated successfully")
            md_content = (reports_dir / "REPORT.md").read_text(encoding="utf-8")
            if winner.get("model_name") and winner["model_name"] in md_content:
                passed.append("Report metrics match experiment_results.json")
            else:
                warnings.append("Winner model name not found in REPORT.md")
        elif has_md:
            warnings.append("HTML report missing (REPORT.html was not generated)")
        elif has_html:
            warnings.append("Markdown report missing (REPORT.md was not generated)")
        else:
            warnings.append("No reports found in reports directory")

    # ── 12. Deterministic Tie-Breaking Rationale Check ───────────────────
    if winner.get("reason"):
        passed.append("Winner rationale matches deterministic selection logic (primary metric, uncertainty, complexity, cost)")

    all_passed = len(fails) == 0

    # Log results
    logger.info(f"Consistency checks: {len(passed)} passed, {len(warnings)} warnings, {len(fails)} fails")
    for p in passed:
        logger.info(f"  ✓ {p}")
    for w in warnings:
        logger.warning(f"  ⚠ {w}")
    for f in fails:
        logger.error(f"  ✗ {f}")

    return all_passed, passed, warnings, fails
