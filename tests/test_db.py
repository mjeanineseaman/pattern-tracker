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
