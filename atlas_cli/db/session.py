from sqlmodel import SQLModel, create_engine, Session
from atlas_cli.core.config import settings

def get_engine():
    db_url = settings.db_url
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, echo=False, connect_args=connect_args)

engine = get_engine()

def create_db_and_tables() -> None:
    """Ensure workspace directory exists and initialize database schema."""
    settings.ensure_workspace()
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
