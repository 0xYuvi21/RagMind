"""
SQLAlchemy engine/session plumbing. One engine per process, built from Settings
(injected, not read from the environment here) so tests can point at a
throwaway SQLite file or an in-memory database instead of data/app.db.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def build_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def build_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(engine) -> None:
    """Create all tables. Safe to call repeatedly (no-op if they already exist)."""
    from Backend import db_models  # noqa: F401 - import registers models on Base.metadata

    Base.metadata.create_all(bind=engine)
