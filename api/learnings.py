"""Garment learnings (PLAN.md Phase 9.2): what past fittings taught us
about this body, scoped by garment type.

Distinct from the revisions table. A revision is a *fact* — what changed
on this pattern on this date. A learning is an *inference* — what this
body seems to need, carried forward to the next garment of the same kind.
Every learning keeps the revision id it came from, so an inference can
always be traced back to the observation behind it.

**What gates a learning, and why it matters.** Only a revision recorded as
`MUSLIN_FITTING` becomes one. A `MANUAL` or `AI_ADJUST` revision is a
design change or a guess — worth keeping in history and answerable via
/ask, but not evidence about a body. Without that gate a single
speculative edit propagates into every future garment with no muslin ever
having validated it, which is exactly what CLAUDE.md's "things that need a
human" section exists to prevent.

No LLM call here. The learning text is the sewist's own `reason` and
`fit_notes` — the words written right after taking a muslin off a body are
already the lesson, and paraphrasing them through a model would add a
hallucination surface to the one part of this system that is pure
first-hand observation.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from api.rag.embed import embed_text
from api.rag.retrieve import cosine_similarity
from api.schema import Revision, RevisionSource

# Above this lexical similarity, a new observation is treated as the same
# lesson learned again rather than a new one. Tuned conservatively: a false
# "same" silently merges two distinct corrections, which is worse than
# carrying a near-duplicate.
_DUPLICATE_SIMILARITY = 0.85

# Only this source is evidence about a body. See the module docstring.
_EVIDENCE_SOURCES = {RevisionSource.MUSLIN_FITTING}


@dataclass
class Learning:
    id: str
    garment_type: str
    learning: str
    source_revision_id: str
    times_confirmed: int
    created_at: str


def learning_text(revision: Revision) -> str:
    """The lesson, in the sewist's own words. `reason` is why the change
    was made; `fit_notes` is what the body actually did — both matter, and
    the second is usually the more transferable half."""
    parts = [revision.reason.strip(), revision.fit_notes.strip()]
    return " ".join(p for p in parts if p)


def is_evidence(revision: Revision) -> bool:
    return revision.source in _EVIDENCE_SOURCES and bool(learning_text(revision))


def _row_to_learning(row: tuple) -> Learning:
    return Learning(*row)


def learnings_for(
    db: sqlite3.Connection, garment_type: str, limit: int = 10,
) -> list[Learning]:
    """What to feed the drafting step, most-confirmed first.

    Ordered by `times_confirmed` rather than ranked by embedding
    similarity. At personal scale a garment type accumulates a handful of
    lessons, not thousands — "the corrections this body has needed most
    often" is both more useful and more honest than a cosine ranking over
    a set small enough to read in full. See PLAN.md 9.3.
    """
    rows = db.execute(
        """
        SELECT id, garment_type, learning, source_revision_id, times_confirmed, created_at
        FROM garment_learnings
        WHERE garment_type = ?
        ORDER BY times_confirmed DESC, created_at DESC
        LIMIT ?
        """,
        (garment_type, limit),
    ).fetchall()
    return [_row_to_learning(row) for row in rows]


def record_learning(
    db: sqlite3.Connection, revision: Revision, garment_type: str,
) -> Learning | None:
    """Turn a fitting observation into a learning, or confirm an existing one.

    Returns the learning written or confirmed, or None when the revision
    isn't evidence (see the gate in the module docstring). Does not commit
    — the caller owns the transaction, so a learning and the revision that
    produced it land together or not at all.
    """
    if not is_evidence(revision):
        return None

    text = learning_text(revision)
    existing = _find_similar(db, garment_type, text)
    if existing is not None:
        db.execute(
            "UPDATE garment_learnings SET times_confirmed = times_confirmed + 1 WHERE id = ?",
            (existing.id,),
        )
        return Learning(
            id=existing.id,
            garment_type=existing.garment_type,
            learning=existing.learning,
            source_revision_id=existing.source_revision_id,
            times_confirmed=existing.times_confirmed + 1,
            created_at=existing.created_at,
        )

    learning = Learning(
        id=uuid.uuid4().hex,
        garment_type=garment_type,
        learning=text,
        source_revision_id=revision.id,
        times_confirmed=1,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    db.execute(
        """
        INSERT INTO garment_learnings
            (id, garment_type, learning, source_revision_id, times_confirmed, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            learning.id, learning.garment_type, learning.learning,
            learning.source_revision_id, learning.times_confirmed, learning.created_at,
        ),
    )
    return learning


def _find_similar(
    db: sqlite3.Connection, garment_type: str, text: str,
) -> Learning | None:
    """The same lesson written twice in slightly different words should
    raise its confirmation count, not add a second row.

    Reuses the local hashed bag-of-words embedder rather than exact string
    matching: "the back pulled across the shoulder blade again" and "back
    pulled across the shoulder blades" are the same observation, and
    neither is a substring of the other. Lexical overlap is a good fit
    here specifically because both texts come from one person describing
    one body in their own consistent vocabulary (see CLAUDE.md).
    """
    candidate = embed_text(text)
    best: Learning | None = None
    best_score = 0.0
    for learning in learnings_for(db, garment_type, limit=1000):
        score = cosine_similarity(candidate, embed_text(learning.learning))
        if score > best_score:
            best, best_score = learning, score
    return best if best_score >= _DUPLICATE_SIMILARITY else None
