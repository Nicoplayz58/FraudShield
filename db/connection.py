from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_database_url(explicit_url: str | None = None) -> str:
    database_url = explicit_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL no esta configurada.")
    return database_url


def create_database_engine(database_url: str | None = None) -> Engine:
    return create_engine(get_database_url(database_url), pool_pre_ping=True, future=True)
