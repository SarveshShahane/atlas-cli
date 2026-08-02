"""
Database session management.
Engine is initialized lazily on first access to prevent premature DB connections.
"""
from __future__ import annotations

from typing import Generator

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from atlas_cli.core.config import settings

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return (or lazily create) the database engine."""
    global _engine
    if _engine is None:
        db_url = settings.db_url
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        _engine = create_engine(db_url, echo=False, connect_args=connect_args)
    return _engine


def create_db_and_tables() -> None:
    """Ensure workspace directory exists and initialize database schema."""
    settings.ensure_workspace()
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Generator[Session, None, None]:
    """Yield a database session context."""
    with Session(get_engine()) as session:
        yield session
