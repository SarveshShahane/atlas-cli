"""
FastAPI Microservice Scaffolder — Phase 10 Deployment Generator.

Scaffolds a production-ready FastAPI inference microservice directory:
  - main.py (FastAPI application with /health, /info, /predict, /predict_batch)
  - schemas.py (Pydantic request/response schemas matching dataset features)
  - model/ (Model binary + preprocessing pipeline joblib artifacts)
  - requirements.txt (Production microservice dependencies)
  - Dockerfile (Slim multi-stage container build)
  - docker-compose.yml (One-command service orchestration)
  - test_api.py (Automated verification script)
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from atlas_cli.core.config import settings

logger = logging.getLogger("atlas_cli")


@dataclass
class FeatureColumn:
    """Feature column schema specification."""

    name: str
    dtype: str
    inferred_type: str
    pydantic_type: str
    sample_value: Any


@dataclass
class ScaffoldResult:
    """Output of the deployment scaffolder."""

    run_id: str
    model_name: str
    library: str
    task_type: str
    primary_metric: str
    output_dir: Path
    scaffolded_files: list[str] = field(default_factory=list)
    endpoint_urls: list[str] = field(default_factory=list)


def _map_dtype_to_pydantic(dtype: str, inferred_type: str) -> str:
    """Map Pandas/dataset dtype to Python/Pydantic type annotation."""
    d = dtype.lower()
    inf = inferred_type.lower()

    if "int" in d:
        return "int"
    if "float" in d or "double" in d or inf == "numeric":
        return "float"
    if "bool" in d or inf == "boolean":
        return "bool"
    return "str"


def _extract_feature_columns(run_dir: Path) -> list[FeatureColumn]:
    """Extract feature schema from dataset_summary.json (excluding target)."""
    columns: list[FeatureColumn] = []

    plan_path = run_dir / "execution_plan.json"
    target_col = None
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        target_col = plan.get("target_column")

    ds_path = run_dir / "dataset_summary.json"
    if ds_path.exists():
        try:
            ds_data = json.loads(ds_path.read_text(encoding="utf-8"))
            raw_cols = ds_data.get("schema", {}).get("columns", [])
            for c in raw_cols:
                name = c.get("name")
                if name and name != target_col:
                    dtype = c.get("dtype", "float64")
                    inf_type = c.get("inferred_type", "numeric")
                    pydantic_type = _map_dtype_to_pydantic(dtype, inf_type)

                    sample_vals = c.get("sample_values", [])
                    sample_val = sample_vals[0] if sample_vals else 0
                    if pydantic_type == "int":
                        try:
                            sample_val = int(sample_val)
                        except Exception:
                            sample_val = 0
                    elif pydantic_type == "float":
                        try:
                            sample_val = float(sample_val)
                        except Exception:
                            sample_val = 0.0
                    elif pydantic_type == "bool":
                        sample_val = bool(sample_val)

                    columns.append(
                        FeatureColumn(
                            name=name,
                            dtype=dtype,
                            inferred_type=inf_type,
                            pydantic_type=pydantic_type,
                            sample_value=sample_val,
                        )
                    )
        except Exception as exc:
            logger.warning(f"Could not parse dataset_summary.json for schema: {exc}")

    # Fallback to features_meta.json
    if not columns:
        meta_path = run_dir / "features_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for name in meta.get("feature_names", []):
                    columns.append(
                        FeatureColumn(
                            name=name,
                            dtype="float64",
                            inferred_type="numeric",
                            pydantic_type="float",
                            sample_value=0.0,
                        )
                    )
            except Exception as exc:
                logger.warning(f"Could not parse features_meta.json: {exc}")

    return columns


def scaffold_deployment(
    run_id: str,
    output_dir: str | Path,
    *,
    experiment_id: Optional[str] = None,
) -> ScaffoldResult:
    """
    Scaffold complete FastAPI microservice for winning or specified model.

    Args:
        run_id: Run identifier.
        output_dir: Target microservice directory path.
        experiment_id: Optional model name / experiment ID to deploy.

    Returns:
        ScaffoldResult.

    Raises:
        FileNotFoundError: If run directory or model artifact is missing.
    """
    run_dir = settings.workspace_dir / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_dir = out_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target model
    model_name = "Winning Model"
    library = "sklearn"
    safe_name = "model"
    primary_metric = "roc_auc"
    task_type = "binary_classification"

    comp_path = run_dir / "comparison_results.json"
    if comp_path.exists():
        comp_data = json.loads(comp_path.read_text(encoding="utf-8"))
        primary_metric = comp_data.get("primary_metric", "roc_auc")
        task_type = comp_data.get("task_type", "binary_classification")
        winner = comp_data.get("winner")
        if winner and not experiment_id:
            model_name = winner.get("model_name", "Winning Model")
            library = winner.get("library", "sklearn")
            safe_name = model_name.lower().replace(" ", "_").replace("/", "_")

    if experiment_id:
        safe_name = experiment_id.lower().replace(" ", "_").replace("/", "_")
        model_name = experiment_id

    # Copy model artifact
    src_model_path = run_dir / "models" / f"{safe_name}.joblib"
    if not src_model_path.exists():
        src_model_path = run_dir / "models" / f"{safe_name}_refined.joblib"
    if not src_model_path.exists():
        # Fallback to any model joblib
        models_in_run = list((run_dir / "models").glob("*.joblib"))
        if models_in_run:
            src_model_path = models_in_run[0]
        else:
            raise FileNotFoundError(f"No model joblib artifacts found in {run_dir / 'models'}")

    dst_model_path = model_dir / "model.joblib"
    shutil.copy2(src_model_path, dst_model_path)
    logger.info(f"Model artifact copied: {dst_model_path}")

    # Copy preprocessing pipeline if present
    src_pipeline = run_dir / "pipeline.joblib"
    has_pipeline = src_pipeline.exists()
    if has_pipeline:
        shutil.copy2(src_pipeline, model_dir / "pipeline.joblib")
        logger.info(f"Preprocessing pipeline copied: {model_dir / 'pipeline.joblib'}")

    # Extract schema features
    features = _extract_feature_columns(run_dir)

    scaffolded_files: list[str] = []

    # 1. Generate schemas.py
    schemas_code = _generate_schemas_py(features, task_type=task_type)
    (out_path / "schemas.py").write_text(schemas_code, encoding="utf-8")
    scaffolded_files.append("schemas.py")

    # 2. Generate main.py
    main_code = _generate_main_py(
        model_name=model_name,
        library=library,
        task_type=task_type,
        primary_metric=primary_metric,
        run_id=run_id,
        has_pipeline=has_pipeline,
        features=features,
    )
    (out_path / "main.py").write_text(main_code, encoding="utf-8")
    scaffolded_files.append("main.py")

    # 3. Generate requirements.txt
    reqs_code = _generate_requirements_txt(library)
    (out_path / "requirements.txt").write_text(reqs_code, encoding="utf-8")
    scaffolded_files.append("requirements.txt")

    # 4. Generate Dockerfile
    dockerfile_code = _generate_dockerfile()
    (out_path / "Dockerfile").write_text(dockerfile_code, encoding="utf-8")
    scaffolded_files.append("Dockerfile")

    # 5. Generate docker-compose.yml
    compose_code = _generate_docker_compose(model_name)
    (out_path / "docker-compose.yml").write_text(compose_code, encoding="utf-8")
    scaffolded_files.append("docker-compose.yml")

    # 6. Generate test_api.py
    test_code = _generate_test_api(features, task_type=task_type)
    (out_path / "test_api.py").write_text(test_code, encoding="utf-8")
    scaffolded_files.append("test_api.py")

    scaffolded_files.append("model/model.joblib")
    if has_pipeline:
        scaffolded_files.append("model/pipeline.joblib")

    return ScaffoldResult(
        run_id=run_id,
        model_name=model_name,
        library=library,
        task_type=task_type,
        primary_metric=primary_metric,
        output_dir=out_path,
        scaffolded_files=scaffolded_files,
        endpoint_urls=[
            "GET  /health",
            "GET  /info",
            "POST /predict",
            "POST /predict_batch",
        ],
    )


def _generate_schemas_py(features: list[FeatureColumn], *, task_type: str) -> str:
    """Generate Pydantic schemas file for API requests & responses."""
    fields_code = []
    sample_dict = {}

    for f in features:
        fields_code.append(f"    {f.name}: {f.pydantic_type} = Field(..., description=\"Feature '{f.name}'\")")
        sample_dict[f.name] = f.sample_value

    sample_json = json.dumps(sample_dict, indent=8)

    is_classification = task_type in {"binary_classification", "multiclass_classification"}
    prob_field = "    probabilities: Optional[list[float]] = None\n" if is_classification else ""

    return f'''"""
Pydantic Request & Response Schemas.
Auto-generated by Atlas CLI Deployment Generator.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input payload for a single prediction request."""
{chr(10).join(fields_code) if fields_code else "    features: list[float] = Field(..., description='Raw feature vector')"}

    model_config = {{
        "json_schema_extra": {{
            "examples": [
{sample_json}
            ]
        }}
    }}


class PredictionBatchInput(BaseModel):
    """Batch input payload for multiple prediction requests."""
    instances: list[PredictionInput] = Field(..., description="List of feature records")


class PredictionOutput(BaseModel):
    """Output prediction response."""
    prediction: Any = Field(..., description="Predicted label or value")
{prob_field}    model_name: str = Field(..., description="Serving model name")


class PredictionBatchOutput(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionOutput] = Field(..., description="List of prediction responses")
    count: int = Field(..., description="Number of instances processed")


class ServiceInfo(BaseModel):
    """Service metadata information."""
    service_name: str = "Atlas CLI Model Microservice"
    version: str = "1.0.0"
    model_name: str
    library: str
    task_type: str
    status: str = "healthy"
'''


def _generate_main_py(
    model_name: str,
    library: str,
    task_type: str,
    primary_metric: str,
    run_id: str,
    has_pipeline: bool,
    features: list[FeatureColumn],
) -> str:
    """Generate main.py FastAPI application code."""
    is_classification = task_type in {"binary_classification", "multiclass_classification"}

    feature_extraction = ""
    if features:
        feature_names = [f.name for f in features]
        feature_extraction = f"""
def _record_to_features(inp: PredictionInput) -> list:
    return [{', '.join(f'inp.{name}' for name in feature_names)}]
"""
    else:
        feature_extraction = """
def _record_to_features(inp: PredictionInput) -> list:
    return getattr(inp, 'features', [])
"""

    return f'''"""
Production FastAPI Microservice for {model_name}.
Auto-generated by Atlas CLI Deployment Generator.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    PredictionBatchInput,
    PredictionBatchOutput,
    PredictionInput,
    PredictionOutput,
    ServiceInfo,
)

logger = logging.getLogger("model_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="{model_name} Inference API",
    description="Production ML microservice generated by Atlas CLI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"

estimator: Any = None
pipeline: Any = None

@app.on_event("startup")
def load_artifacts():
    global estimator, pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {{MODEL_PATH}}")
    estimator = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model: {{MODEL_PATH}}")

    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        logger.info(f"Loaded preprocessing pipeline: {{PIPELINE_PATH}}")

{feature_extraction}

def _transform_and_predict(features_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if pipeline is not None:
        features_matrix = pipeline.transform(features_matrix)

    preds = estimator.predict(features_matrix)
    probas = None
    if hasattr(estimator, "predict_proba"):
        try:
            probas = estimator.predict_proba(features_matrix)
        except Exception:
            pass
    return preds, probas


@app.get("/health", tags=["Health"])
def health():
    return {{"status": "healthy", "model_loaded": estimator is not None}}


@app.get("/info", response_model=ServiceInfo, tags=["Metadata"])
def info():
    return ServiceInfo(
        model_name="{model_name}",
        library="{library}",
        task_type="{task_type}",
        status="healthy" if estimator is not None else "degraded",
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(inp: PredictionInput):
    if estimator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        raw_feats = _record_to_features(inp)
        X = np.array([raw_feats], dtype=object if any(isinstance(v, str) for v in raw_feats) else float)

        # Build DataFrame if features are named
        if hasattr(inp, "model_dump"):
            df = pd.DataFrame([inp.model_dump()])
            preds, probas = _transform_and_predict(df if pipeline is not None else X)
        else:
            preds, probas = _transform_and_predict(X)

        pred_val = preds[0]
        if isinstance(pred_val, (np.generic, np.ndarray)):
            pred_val = pred_val.item()

        prob_list = None
        if probas is not None:
            prob_list = [float(p) for p in probas[0]]

        return PredictionOutput(
            prediction=pred_val,
            probabilities=prob_list,
            model_name="{model_name}",
        )
    except Exception as exc:
        logger.error(f"Inference error: {{exc}}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict_batch", response_model=PredictionBatchOutput, tags=["Inference"])
def predict_batch(batch: PredictionBatchInput):
    if estimator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        records = [inp.model_dump() for inp in batch.instances]
        df = pd.DataFrame(records)

        preds, probas = _transform_and_predict(df)

        outputs = []
        for i, pred in enumerate(preds):
            pval = pred.item() if isinstance(pred, (np.generic, np.ndarray)) else pred
            plist = [float(p) for p in probas[i]] if probas is not None else None
            outputs.append(
                PredictionOutput(
                    prediction=pval,
                    probabilities=plist,
                    model_name="{model_name}",
                )
            )

        return PredictionBatchOutput(predictions=outputs, count=len(outputs))
    except Exception as exc:
        logger.error(f"Batch inference error: {{exc}}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
'''


def _generate_requirements_txt(library: str) -> str:
    """Generate production microservice requirements.txt."""
    family = library.split(".")[0].lower()
    extra_pkg = ""
    if family == "xgboost":
        extra_pkg = "xgboost>=2.0.0\n"
    elif family == "lightgbm":
        extra_pkg = "lightgbm>=4.0.0\n"
    elif family == "catboost":
        extra_pkg = "catboost>=1.2.0\n"

    return f"""fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
scikit-learn>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
{extra_pkg}requests>=2.28.0
"""


def _generate_dockerfile() -> str:
    """Generate multi-stage slim Dockerfile."""
    return """FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


def _generate_docker_compose(model_name: str) -> str:
    """Generate docker-compose.yml file."""
    safe_service_name = model_name.lower().replace(" ", "-").replace("/", "-")
    return f"""version: '3.8'

services:
  {safe_service_name}:
    build: .
    ports:
      - "8000:8000"
    restart: always
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""


def _generate_test_api(features: list[FeatureColumn], *, task_type: str) -> str:
    """Generate automated API testing script test_api.py."""
    sample_payload = {f.name: f.sample_value for f in features}
    sample_json = json.dumps(sample_payload, indent=4)

    return f'''"""
Automated Test Script for Model Microservice.
Run with: python test_api.py
"""
import json
import urllib.request
import sys

BASE_URL = "http://localhost:8000"
SAMPLE_PAYLOAD = {sample_json}


def test_endpoint(path: str, method: str = "GET", data: dict = None):
    url = f"{{BASE_URL}}{{path}}"
    print(f"Testing {{method}} {{url}}...")
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data else None,
        headers={{"Content-Type": "application/json"}} if data else {{}},
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print(f"  ✅ [{{response.status}}] Success!")
            print(f"  Response: {{json.dumps(res_data, indent=2)}}")
            return res_data
    except Exception as exc:
        print(f"  ❌ Error: {{exc}}")
        return None


if __name__ == "__main__":
    print("=== Testing FastAPI Microservice ===")
    h = test_endpoint("/health")
    i = test_endpoint("/info")
    p = test_endpoint("/predict", method="POST", data=SAMPLE_PAYLOAD)

    if h and p:
        print("\\n🎉 All tests passed successfully!")
    else:
        print("\\n⚠️ Tests failed. Ensure the server is running on http://localhost:8000")
        sys.exit(1)
'''
