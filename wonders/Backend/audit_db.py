"""
SQLAlchemy engine/session plumbing for the audit log store — deliberately a
SEPARATE PostgreSQL database from the app's own DATABASE_URL (Backend/db.py),
so audit history is never lost/rolled back alongside app data (e.g. a test
run against a throwaway SQLite file must not also wipe audit history) and so
the audit store can live on its own instance/credentials in production.

Mirrors Backend/db.py's shape (one engine per process, built from injected
settings, never read from the environment directly here) for the same
testability reason: callers can point this at a temp/throwaway Postgres (or
override the dependency with a no-op logger — see Backend/audit_logger.py's
NullAuditLogger) instead of touching a real audit database in tests.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class AuditBase(DeclarativeBase):
    pass


def build_audit_engine(audit_database_url: str):
    return create_engine(audit_database_url, pool_pre_ping=True)


def build_audit_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_audit_db(engine) -> None:
    """Create all audit tables. Safe to call repeatedly (no-op if they
    already exist)."""
    from Backend import audit_models  # noqa: F401 - import registers models on AuditBase.metadata

    AuditBase.metadata.create_all(bind=engine)
