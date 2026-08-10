"""
Model Wrapper Adapters — Phase 5 Module 4.

Factory that maps ModelCandidate.library strings (e.g. "xgboost.XGBClassifier")
to configured scikit-learn-compatible estimator instances.

Supported libraries:
  - scikit-learn: RandomForest, ExtraTrees, LogisticRegression, Ridge
  - XGBoost: XGBClassifier / XGBRegressor
  - LightGBM: LGBMClassifier / LGBMRegressor
  - CatBoost: CatBoostClassifier / CatBoostRegressor
"""
from __future__ import annotations

import logging
from typing import Any

from atlas_cli.agents.pipeline_planner.schemas import ModelCandidate

logger = logging.getLogger("atlas_cli")
_REGISTRY: dict[str, tuple[str, str]] = {

    "sklearn.RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "sklearn.RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "sklearn.ExtraTreesClassifier": ("sklearn.ensemble", "ExtraTreesClassifier"),
    "sklearn.ExtraTreesRegressor": ("sklearn.ensemble", "ExtraTreesRegressor"),
    "sklearn.LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "sklearn.Ridge": ("sklearn.linear_model", "Ridge"),
    "sklearn.GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "sklearn.GradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor"),

    "sklearn.ensemble.RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "sklearn.ensemble.RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "sklearn.ensemble.ExtraTreesClassifier": ("sklearn.ensemble", "ExtraTreesClassifier"),
    "sklearn.ensemble.ExtraTreesRegressor": ("sklearn.ensemble", "ExtraTreesRegressor"),
    "sklearn.ensemble.GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "sklearn.ensemble.GradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "sklearn.linear_model.LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "sklearn.linear_model.Ridge": ("sklearn.linear_model", "Ridge"),
    "sklearn.svm.SVC": ("sklearn.svm", "SVC"),
    "sklearn.svm.SVR": ("sklearn.svm", "SVR"),
    "sklearn.neighbors.KNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier"),
    "sklearn.neighbors.KNeighborsRegressor": ("sklearn.neighbors", "KNeighborsRegressor"),
    "sklearn.tree.DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier"),
    "sklearn.tree.DecisionTreeRegressor": ("sklearn.tree", "DecisionTreeRegressor"),
    "sklearn.naive_bayes.GaussianNB": ("sklearn.naive_bayes", "GaussianNB"),

    "xgboost.XGBClassifier": ("xgboost", "XGBClassifier"),
    "xgboost.XGBRegressor": ("xgboost", "XGBRegressor"),
    "xgboost.XGBoostClassifier": ("xgboost", "XGBClassifier"),
    "xgboost.XGBoostRegressor": ("xgboost", "XGBRegressor"),
    "XGBClassifier": ("xgboost", "XGBClassifier"),
    "XGBRegressor": ("xgboost", "XGBRegressor"),

    "lightgbm.LGBMClassifier": ("lightgbm", "LGBMClassifier"),
    "lightgbm.LGBMRegressor": ("lightgbm", "LGBMRegressor"),
    "LGBMClassifier": ("lightgbm", "LGBMClassifier"),
    "LGBMRegressor": ("lightgbm", "LGBMRegressor"),

    "catboost.CatBoostClassifier": ("catboost", "CatBoostClassifier"),
    "catboost.CatBoostRegressor": ("catboost", "CatBoostRegressor"),
    "CatBoostClassifier": ("catboost", "CatBoostClassifier"),
    "CatBoostRegressor": ("catboost", "CatBoostRegressor"),
}

_RANDOM_STATE_KEY: dict[str, str] = {
    "catboost": "random_seed",
}

_SUPPORTS_CLASS_WEIGHT = {
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "LogisticRegression",
    "SVC",
    "DecisionTreeClassifier",
}

_SUPPORTS_SCALE_POS_WEIGHT = {"XGBClassifier"}

_SUPPORTS_IS_UNBALANCE = {"LGBMClassifier"}

_SUPPORTS_AUTO_CLASS_WEIGHTS = {"CatBoostClassifier"}

_NO_RANDOM_STATE = {"GaussianNB", "KNeighborsClassifier", "KNeighborsRegressor"}


def _import_class(module_path: str, class_name: str) -> type:
    """Dynamically import a class from its module path."""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _resolve_library_key(library_key: str, task_type: str = "binary_classification") -> tuple[str, str]:
    """
    Resolve a library string to (module_path, class_name).

    Handles exact registry matches, shorthand family names (xgboost, catboost, lightgbm),
    and dynamic import fallbacks.
    """
    key = library_key.strip()
    if key in _REGISTRY:
        return _REGISTRY[key]

    is_regression = task_type == "regression"
    key_lower = key.lower()

    if key_lower in {"xgboost", "xgb", "xgboost.xgb"}:
        return ("xgboost", "XGBRegressor" if is_regression else "XGBClassifier")
    if key_lower in {"catboost", "catboost.catboost", "cb"}:
        return ("catboost", "CatBoostRegressor" if is_regression else "CatBoostClassifier")
    if key_lower in {"lightgbm", "lgb", "lgbm", "lightgbm.lgbm"}:
        return ("lightgbm", "LGBMRegressor" if is_regression else "LGBMClassifier")
    if key_lower in {"randomforest", "rf", "sklearn.randomforest"}:
        return ("sklearn.ensemble", "RandomForestRegressor" if is_regression else "RandomForestClassifier")

    parts = key.rsplit(".", 1)
    if len(parts) == 2:
        module_path, class_name = parts
        try:
            _import_class(module_path, class_name)
            return module_path, class_name
        except (ImportError, AttributeError):
            pass

    raise ValueError(
        f"Unknown model library '{library_key}'. "
        f"Supported: {sorted(_REGISTRY.keys())} (or any valid Python import path)"
    )


def resolve_estimator(
    candidate: ModelCandidate,
    *,
    task_type: str,
    handle_imbalance: bool = False,
    random_seed: int = 42,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    """
    Resolve a ModelCandidate into a configured estimator instance.

    Args:
        candidate: ModelCandidate from the execution plan.
        task_type: One of the TaskType literals.
        handle_imbalance: Whether to enable class-weight balancing.
        random_seed: Seed for reproducibility.
        extra_params: Optional overrides for constructor kwargs.

    Returns:
        Configured estimator instance (sklearn API compatible).

    Raises:
        ValueError: If the library string is not recognised.
    """
    library_key = candidate.library.strip()
    module_path, class_name = _resolve_library_key(library_key, task_type=task_type)
    cls = _import_class(module_path, class_name)

    params: dict[str, Any] = {}

    if class_name not in _NO_RANDOM_STATE:
        family = module_path.split(".")[0]
        seed_key = _RANDOM_STATE_KEY.get(family, "random_state")
        params[seed_key] = random_seed

    if handle_imbalance:
        if class_name in _SUPPORTS_CLASS_WEIGHT:
            params["class_weight"] = "balanced"
        elif class_name in _SUPPORTS_SCALE_POS_WEIGHT and task_type == "binary_classification":
            params["scale_pos_weight"] = 1
        elif class_name in _SUPPORTS_IS_UNBALANCE and task_type == "binary_classification":
            params["is_unbalance"] = True
        elif class_name in _SUPPORTS_AUTO_CLASS_WEIGHTS:
            params["auto_class_weights"] = "Balanced"

    family = module_path.split(".")[0]
    if family == "catboost":
        params["verbose"] = 0

    if family == "xgboost":
        params["verbosity"] = 0

    if family == "lightgbm":
        params["verbose"] = -1

    if extra_params:
        params.update(extra_params)

    logger.info(f"Resolving estimator: {library_key} → {class_name}({params})")
    return cls(**params)


def list_supported_models() -> list[str]:
    """Return all supported library strings."""
    return sorted(_REGISTRY.keys())

