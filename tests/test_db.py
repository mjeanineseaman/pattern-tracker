"""Phase 0 smoke test: get_connection() bootstraps the schema."""
import threading

from api.db import get_connection


def test_get_connection_creates_schema():
    conn = get_connection(":memory:")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {
        "patterns",
        "pattern_versions",
        "revisions",
        "revision_embeddings",
    } <= tables


def test_migrations_add_a_column_to_an_already_existing_table(tmp_path):
    """CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a new
    column in DDL never reaches a database someone already has. Simulates
    one by creating the pre-Phase-9 patterns table by hand."""
    import sqlite3
    from api.db import _apply_migrations

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE patterns (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
        "current_version INTEGER NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO patterns VALUES ('pat_old', 'before phase 9', 1, 'then')")
    conn.commit()
    assert "garment_type" not in {r[1] for r in conn.execute("PRAGMA table_info(patterns)")}

    _apply_migrations(conn)

    assert "garment_type" in {r[1] for r in conn.execute("PRAGMA table_info(patterns)")}
    # the existing row survives, with the new column simply unset
    assert conn.execute("SELECT garment_type FROM patterns WHERE id = 'pat_old'").fetchone() == (None,)
    conn.close()


def test_migrations_are_idempotent():
    conn = get_connection(":memory:")
    from api.db import _apply_migrations

    _apply_migrations(conn)
    _apply_migrations(conn)

    columns = [r[1] for r in conn.execute("PRAGMA table_info(patterns)")]
    assert columns.count("garment_type") == 1


def test_connection_can_be_closed_from_a_different_thread():
    """FastAPI's get_db() generator dependency has its setup (up to yield)
    and teardown (conn.close()) run as two separate anyio threadpool calls,
    with no guarantee they land on the same worker thread — this is exactly
    what sqlite3's default check_same_thread=True forbids. Reproduces that
    shape directly rather than relying on a real concurrent HTTP request to
    happen to land on two different threads."""
    conn = get_connection(":memory:")
    errors = []

    def close_elsewhere():
        try:
            conn.close()
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=close_elsewhere)
    t.start()
    t.join()

    assert errors == []
