"""Initialise the PostgreSQL database.

Steps:
 1. Connect to the server's default `postgres` DB.
 2. CREATE DATABASE quantmate (if not exists).
 3. Connect to quantmate, run schema.sql.
 4. Verify all tables exist.

Run:
    python data/db/init_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import create_engine, text   # noqa: E402

from utils.config import get_settings        # noqa: E402
from utils.logger import get_logger          # noqa: E402

log = get_logger("init_db")

SCHEMA_SQL = Path(__file__).with_name("schema.sql")
EXPECTED_TABLES = [
    "stock_basic",
    "daily_basic",
    "adj_factor",
    "suspend",
    "fina_indicator",
    "moneyflow",
    "factor_values",
]


def _server_url(s: dict) -> str:
    db = s["database"]
    return (
        f"postgresql+psycopg2://{db['user']}:{db['password']}"
        f"@{db['host']}:{db['port']}/postgres"
    )


def _target_url(s: dict) -> str:
    db = s["database"]
    return (
        f"postgresql+psycopg2://{db['user']}:{db['password']}"
        f"@{db['host']}:{db['port']}/{db['database']}"
    )


def ensure_database() -> None:
    s = get_settings()
    target_name = s["database"]["database"]

    eng = create_engine(_server_url(s), isolation_level="AUTOCOMMIT", future=True)
    with eng.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"),
            {"n": target_name},
        ).scalar()
        if exists:
            log.info("Database %s already exists — skip creation.", target_name)
        else:
            # pg doesn't support parameterised CREATE DATABASE
            conn.execute(text(f'CREATE DATABASE "{target_name}"'))
            log.info("Created database %s.", target_name)


def apply_schema() -> None:
    s = get_settings()
    eng = create_engine(_target_url(s), future=True)
    ddl = SCHEMA_SQL.read_text(encoding="utf-8")
    with eng.begin() as conn:
        # run each statement separately so logs are more useful
        for stmt in [x.strip() for x in ddl.split(";") if x.strip()]:
            conn.execute(text(stmt))
    log.info("Schema applied from %s", SCHEMA_SQL)


def verify_schema() -> None:
    s = get_settings()
    eng = create_engine(_target_url(s), future=True)
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        ).fetchall()
    present = {r[0] for r in rows}
    missing = [t for t in EXPECTED_TABLES if t not in present]
    if missing:
        raise RuntimeError(f"Schema verification failed; missing: {missing}")
    log.info("Schema OK — tables present: %s", sorted(present & set(EXPECTED_TABLES)))


def main() -> None:
    ensure_database()
    apply_schema()
    verify_schema()
    log.info("Database initialised successfully.")


if __name__ == "__main__":
    main()
