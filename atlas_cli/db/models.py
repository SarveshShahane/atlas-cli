from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel
import uuid

def generate_uuid() -> str:
    return str(uuid.uuid4())[:8]

class Run(SQLModel, table=True):
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    name: str = Field(default="Default Run")
    dataset_path: str
    dataset_hash: Optional[str] = None
    goal: str = Field(default="Predictive Modeling")
    status: str = Field(default="created")
    reproducibility_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class DatasetMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    file_path: str
    num_rows: int = 0
    num_cols: int = 0
    dataset_hash: str = ""
    summary_json: str = "{}"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Experiment(SQLModel, table=True):
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    model_name: str
    model_type: str
    hyperparams_json: str = "{}"
    status: str = Field(default="pending")
    duration_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MetricLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: str = Field(foreign_key="experiment.id")
    run_id: str = Field(foreign_key="run.id")
    metric_name: str
    metric_value: float
    split_type: str = Field(default="val")
    created_at: datetime = Field(default_factory=datetime.utcnow)
