"""Tests for garment learnings (PLAN.md Phase 9.2).

The gate is the important part: only a muslin fitting is evidence about a
body. Everything else is a design choice or a guess, and letting those
through would propagate an unvalidated correction into every future
garment.
"""
import sqlite3

from api.learnings import (
    is_evidence, learning_text, learnings_for, record_learning,
)
from api.schema import DDL, ChangeType, FieldDelta, Revision, RevisionSource


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(DDL)
    conn.execute(
        "INSERT INTO patterns (id, name, current_version, created_at, garment_type) "
        "VALUES ('pat_001', 'x', 1, 'now', 'bodice')",
    )
    return conn


def _revision(
    id_="rev_1",
    source=RevisionSource.MUSLIN_FITTING,
    reason="widened the back waist dart",
    fit_notes="the back pulled across the shoulder blade",
) -> Revision:
    revision = Revision(
        id=id_, pattern_id="pat_001", version=2, parent_version=1,
        change_type=ChangeType.DART, source=source,
        deltas=[FieldDelta("pieces[bodice_back].darts[back_waist_dart].leg_a", "a", "b")],
        reason=reason, fit_notes=fit_notes,
    )
    return revision


def _store(conn: sqlite3.Connection, revision: Revision) -> None:
    """A learning references a real revision row, so the FK has something
    to point at."""
    conn.execute(
        "INSERT INTO revisions (id, pattern_id, version, parent_version, change_type, "
        "source, deltas_json, reason, fit_notes, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, ?)",
        (
            revision.id, revision.pattern_id, revision.version, revision.parent_version,
            revision.change_type.value, revision.source.value,
            revision.reason, revision.fit_notes, revision.created_at,
        ),
    )


# --------------------------------------------------------------------------
# The evidence gate
# --------------------------------------------------------------------------
def test_a_muslin_fitting_is_evidence():
    assert is_evidence(_revision(source=RevisionSource.MUSLIN_FITTING))


def test_a_manual_edit_is_not_evidence():
    """A design change isn't an observation about a body."""
    assert not is_evidence(_revision(source=RevisionSource.MANUAL))


def test_an_ai_adjustment_is_not_evidence():
    """Letting the model's own guesses become learnings would compound
    them into every later garment with nothing checking them."""
    assert not is_evidence(_revision(source=RevisionSource.AI_ADJUST))


def test_a_fitting_with_nothing_written_down_is_not_evidence():
    assert not is_evidence(_revision(reason="", fit_notes=""))


def test_record_learning_skips_a_non_fitting_revision():
    conn = _db()
    revision = _revision(source=RevisionSource.MANUAL)
    _store(conn, revision)

    assert record_learning(conn, revision, "bodice") is None
    assert learnings_for(conn, "bodice") == []


# --------------------------------------------------------------------------
# Writing and confirming
# --------------------------------------------------------------------------
def test_a_fitting_becomes_a_learning_traceable_to_its_revision():
    conn = _db()
    revision = _revision()
    _store(conn, revision)

    learning = record_learning(conn, revision, "bodice")

    assert learning.times_confirmed == 1
    assert learning.source_revision_id == "rev_1"
    assert "shoulder blade" in learning.learning
    assert [l.id for l in learnings_for(conn, "bodice")] == [learning.id]


def test_learning_text_keeps_both_the_reason_and_what_the_body_did():
    text = learning_text(_revision(reason="let out the side seam", fit_notes="tight across the ribs"))

    assert "let out the side seam" in text
    assert "tight across the ribs" in text


def test_the_same_lesson_in_different_words_confirms_rather_than_duplicates():
    """Two fittings finding the same thing should raise confidence, not
    fill the prompt with near-identical lines."""
    conn = _db()
    first = _revision(id_="rev_1")
    second = _revision(
        id_="rev_2",
        reason="widened the back waist dart",
        fit_notes="the back pulled across the shoulder blades again",
    )
    _store(conn, first)
    _store(conn, second)

    record_learning(conn, first, "bodice")
    confirmed = record_learning(conn, second, "bodice")

    stored = learnings_for(conn, "bodice")
    assert len(stored) == 1
    assert stored[0].times_confirmed == 2
    assert confirmed.times_confirmed == 2


def test_an_unrelated_lesson_is_recorded_separately():
    conn = _db()
    first = _revision(id_="rev_1")
    second = _revision(
        id_="rev_2",
        reason="shortened the centre front hem",
        fit_notes="it dipped below the waist at the front",
    )
    _store(conn, first)
    _store(conn, second)

    record_learning(conn, first, "bodice")
    record_learning(conn, second, "bodice")

    assert len(learnings_for(conn, "bodice")) == 2


def test_learnings_are_scoped_to_their_garment_type():
    """A dress lesson must not steer a skirt draft — opening retrieval up
    across patterns (9.3) makes garment_type the only thing preventing it."""
    conn = _db()
    revision = _revision()
    _store(conn, revision)
    record_learning(conn, revision, "dress")

    assert learnings_for(conn, "skirt") == []
    assert len(learnings_for(conn, "dress")) == 1


def test_most_confirmed_learnings_come_first():
    conn = _db()
    once = _revision(id_="rev_once", reason="raised the neckline", fit_notes="it gaped at the front")
    twice_a = _revision(id_="rev_a", reason="widened the back dart", fit_notes="back pulled")
    twice_b = _revision(id_="rev_b", reason="widened the back dart", fit_notes="back pulled")
    for r in (once, twice_a, twice_b):
        _store(conn, r)

    record_learning(conn, once, "bodice")
    record_learning(conn, twice_a, "bodice")
    record_learning(conn, twice_b, "bodice")

    ranked = learnings_for(conn, "bodice")
    assert ranked[0].times_confirmed == 2
    assert "back dart" in ranked[0].learning


def test_limit_caps_what_reaches_the_prompt():
    conn = _db()
    for i in range(5):
        r = _revision(id_=f"rev_{i}", reason=f"distinct lesson number {i}", fit_notes="")
        _store(conn, r)
        record_learning(conn, r, "bodice")

    assert len(learnings_for(conn, "bodice", limit=3)) == 3
