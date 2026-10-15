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
    return conn
