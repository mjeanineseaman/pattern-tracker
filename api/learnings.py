"""What past fittings taught about this body, scoped by garment type.

A revision is a fact — what changed, when. A learning is an inference —
what this body needs, carried to the next garment of the same kind. Each
keeps its source revision id so the inference stays traceable.

Only a MUSLIN_FITTING revision becomes one. A MANUAL or AI_ADJUST revision
is a design change or a guess: worth keeping in history and answerable via
/ask, but not evidence about a body. Without that gate one speculative
edit propagates into every future garment unvalidated.

No LLM call here — the words written right after taking a muslin off a
body are already the lesson, and paraphrasing them would add a
hallucination surface to the one purely first-hand part of this system.
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
    """`reason` is why the change was made, `fit_notes` is what the body
    did — the second is usually the more transferable half."""
    parts = [revision.reason.strip(), revision.fit_notes.strip()]
    return " ".join(p for p in parts if p)


def is_evidence(revision: Revision) -> bool:
    return revision.source in _EVIDENCE_SOURCES and bool(learning_text(revision))


def _row_to_learning(row: tuple) -> Learning:
    return Learning(*row)


def learnings_for(
    db: sqlite3.Connection, garment_type: str, limit: int = 10,
) -> list[Learning]:
    """Most-confirmed first, rather than ranked by embedding similarity: a
    garment type accumulates a handful of lessons, not thousands, and "what
    this body needed most often" beats a cosine ranking over a set small
    enough to read in full.
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
    """Returns the learning written or confirmed, or None when the revision
    isn't evidence. Does not commit: the caller owns the transaction, so a
    learning and the revision behind it land together or not at all.
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
    """The same lesson in different words should raise the confirmation
    count, not add a row.

    Uses the local embedder rather than string matching: "pulled across the
    shoulder blade again" and "pulled across the shoulder blades" are the
    same observation and neither contains the other. Lexical overlap works
    here because both come from one person's consistent vocabulary.
    """
    candidate = embed_text(text)
    best: Learning | None = None
    best_score = 0.0
    for learning in learnings_for(db, garment_type, limit=1000):
        score = cosine_similarity(candidate, embed_text(learning.learning))
        if score > best_score:
            best, best_score = learning, score
    return best if best_score >= _DUPLICATE_SIMILARITY else None
