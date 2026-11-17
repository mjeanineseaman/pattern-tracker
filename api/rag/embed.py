"""Local hashed bag-of-words embeddings, not a semantic embeddings API.

Anthropic has no embeddings endpoint, and Voyage/OpenAI were rejected as a
second paid provider plus a network call per revision, for a local-only
project at personal scale. See CLAUDE.md for the full reasoning.

No vocabulary table, no training step: each token hashes into one of
DIMENSIONS buckets, so a text embeds independently of everything else.
MODEL_NAME is stored per vector so a later swap doesn't require guessing
which rows came from which method.
"""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from array import array

from api.schema import Revision, revision_from_row

DIMENSIONS = 512
MODEL_NAME = "hashed-bow-v1"

# The revision-text template's own boilerplate (schema.py's
# Revision.embedding_text()) — words that appear in EVERY revision because
# of how that string is built, not because of what the revision was about.
# Dropping them stops them from diluting every vector toward the same
# "generic revision" point. Deliberately NOT a general English stopword
# list — short connector words like "to"/"of" that might appear inside a
# real `reason`/`fit_notes` are left alone, since those carry real signal
# here that a generic list would strip out along with the noise.
_STOPWORDS = frozenset({
    "change", "via", "on", "pattern", "reason", "fit", "notes",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything non-alphanumeric (which conveniently
    splits dotted/bracketed paths and underscore_joined ids into their
    meaningful parts, e.g. "back_waist_dart" -> "back", "waist", "dart" —
    see CLAUDE.md for why that specific behaviour is the point, not an
    accident), then drop the template boilerplate."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _bucket(token: str, dimensions: int) -> int:
    # hashlib, not Python's builtin hash(): the builtin is salted per
    # process (PYTHONHASHSEED), so the same token would land in a
    # different bucket on every run — fatal for comparing vectors stored
    # on one run against a query embedded on another.
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dimensions


def embed_text(text: str, dimensions: int = DIMENSIONS) -> array:
    """Text -> an L2-normalized float32 vector of length `dimensions`, as a
    stdlib `array` (no numpy dependency — see CLAUDE.md's stack notes).
    An empty/all-stopword text embeds to the zero vector; cosine similarity
    against it is always 0, which is the correct "no signal" answer rather
    than an error."""
    vec = [0.0] * dimensions
    for token in tokenize(text):
        vec[_bucket(token, dimensions)] += 1.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return array("f", vec)


def embed_revision(revision: Revision, dimensions: int = DIMENSIONS) -> array:
    return embed_text(revision.embedding_text(), dimensions)


def store_embedding(db: sqlite3.Connection, revision_id: str, vector: array, model: str = MODEL_NAME) -> None:
    db.execute(
        "INSERT OR REPLACE INTO revision_embeddings (revision_id, model, vector) VALUES (?, ?, ?)",
        (revision_id, model, vector.tobytes()),
    )


def embed_and_store_revision(db: sqlite3.Connection, revision: Revision) -> None:
    """Does not commit — callers (api.main's /revise, backfill below) are
    expected to do that themselves as part of their own transaction."""
    store_embedding(db, revision.id, embed_revision(revision))
