"""PostgreSQL read/write wrapper.

Thin layer on top of SQLAlchemy + pandas.  Every module that needs DB access
goes through this file, so switching engines later touches one place only.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from utils.config import get_db_url, get_settings
from utils.logger import get_logger

log = get_logger(__name__)

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Lazily create a singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = get_db_url()
        _engine = create_engine(url, pool_pre_ping=True, future=True)
        log.debug("DB engine created: %s", url.replace(
            get_settings()["database"]["password"], "***"
        ))
    return _engine


@contextmanager
def get_conn() -> Iterator:
    eng = get_engine()
    with eng.connect() as conn:
        yield conn


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    with get_conn() as conn:
        return pd.read_sql_query(text(sql), conn, params=params or {})


def execute(sql: str, params: dict | None = None) -> None:
    """Run a DDL/DML statement (CREATE / INSERT / UPDATE / DELETE)."""
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})


def upsert_df(
    df: pd.DataFrame,
    table: str,
    pk_cols: Iterable[str],
    chunksize: int = 5000,
) -> int:
    """Upsert a DataFrame via ON CONFLICT. Returns rows written."""
    if df.empty:
        return 0

    cols = list(df.columns)
    col_list = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    update_cols = [c for c in cols if c not in set(pk_cols)]
    update_clause = (
        ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        if update_cols else ""
    )
    pk_clause = ", ".join(pk_cols)

    if update_clause:
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({pk_clause}) DO UPDATE SET {update_clause}"
        )
    else:
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({pk_clause}) DO NOTHING"
        )

    total = 0
    with get_engine().begin() as conn:
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start : start + chunksize]
            records = chunk.to_dict(orient="records")
            conn.execute(text(sql), records)
            total += len(records)
    log.info("Upserted %d rows into %s", total, table)
    return total


def table_exists(table: str) -> bool:
    sql = (
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables "
        "  WHERE table_name = :t"
        ")"
    )
    with get_conn() as conn:
        return bool(conn.execute(text(sql), {"t": table}).scalar())
