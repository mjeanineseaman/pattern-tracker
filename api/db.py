"""SQLite connection helper.

Executes `schema.DDL` on every connection. That's cheap and safe because
every statement in DDL is `CREATE TABLE IF NOT EXISTS` — on an existing
database it's a no-op; on a fresh one it bootstraps the schema. No separate
migration step for Phase 0.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from api.schema import DDL

# Loaded here since this is currently the first module most entry points
# (tests, scripts, eventually main.py) import. Only ANTHROPIC_API_KEY lives
# in .env today (see CLAUDE.md); db.py itself doesn't read it, but nothing
# downstream has to remember to call this themselves. A no-op if .env is
# absent, so it's harmless in CI or a fresh checkout.
load_dotenv()

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "pattern_tracker.db"

# Columns added to tables that already exist in databases out in the wild.
# `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a new
# column in DDL never reaches one — it needs an ALTER. Each entry is
# (table, column, full ALTER statement); applied only when the column is
# genuinely absent, so running this repeatedly is safe.
#
# Deliberately additive-only. Dropping or retyping a column would need a
# real migration that also rewrites stored revision deltas (CLAUDE.md:
# "ids are stable and permanent"), which is a different, much more careful
# job than this list is for.
MIGRATIONS: list[tuple[str, str, str]] = [
    ("patterns", "garment_type", "ALTER TABLE patterns ADD COLUMN garment_type TEXT"),
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, column, statement in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if existing and column not in existing:
            conn.execute(statement)
            conn.commit()


def get_connection(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with the schema ensured.

    Pass `path=":memory:"` for tests or throwaway scripts.
    """
    # check_same_thread=False: FastAPI's get_db() is a generator dependency,
    # and anyio's threadpool runs its setup (up to `yield`) and teardown
    # (conn.close()) as two SEPARATE run_sync calls that aren't guaranteed
    # the same worker thread — sqlite3's default same-thread check then
    # raises ProgrammingError the moment two requests land on different
    # threads and the close() lands on a thread that didn't open the
    # connection. Each request still gets its own connection, closed once,
    # so this doesn't introduce real cross-thread sharing.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(DDL)
    _apply_migrations(conn)
    return conn
