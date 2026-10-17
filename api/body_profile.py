"""The body model (PLAN.md Phase 9.1): current best known measurements for
one body.

Distinct from a pattern version's `Measurements` snapshot, which records
what was true when that version was drafted. The profile is what's true
*now* — it's what a new pattern starts from, and what a fitting
correction updates.

Holds measured facts only. "This body needs 4mm more back shoulder dart"
is an inference drawn from a fitting, not a measurement, and belongs in
api.learnings instead. Keeping the two apart is what makes the profile
auditable: every number in it was measured, not derived.

No LLM call anywhere in this module. A measurement is a fact you read off
a tape, and the one person using this tool is the one who measured it.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from pydantic import TypeAdapter

from api.schema import FieldDelta, Measurements

DEFAULT_PROFILE_ID = "default"

_measurements_adapter = TypeAdapter(Measurements)


def get_profile(
    db: sqlite3.Connection, profile_id: str = DEFAULT_PROFILE_ID,
) -> Optional[Measurements]:
    """The stored body model, or None if nothing's been recorded yet."""
    row = db.execute(
        "SELECT measurements_json FROM body_profiles WHERE id = ?", (profile_id,),
    ).fetchone()
    if row is None:
        return None
    return _measurements_adapter.validate_python(json.loads(row[0]))


def save_profile(
    db: sqlite3.Connection,
    measurements: Measurements,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> str:
    """Upsert the body model. Returns the `updated_at` stamp written."""
    updated_at = datetime.now(timezone.utc).isoformat()
    db.execute(
        """
        INSERT INTO body_profiles (id, measurements_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            measurements_json = excluded.measurements_json,
            updated_at = excluded.updated_at
        """,
        (
            profile_id,
            json.dumps(_measurements_adapter.dump_python(measurements, mode="json")),
            updated_at,
        ),
    )
    return updated_at


def seed_profile_if_missing(
    db: sqlite3.Connection,
    measurements: Measurements,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> bool:
    """Record the first pattern's measurements as the initial body model.

    Only fills a gap — an existing profile is never overwritten here,
    since a later pattern drafted with deliberately different numbers
    (someone else's bodice, a test) shouldn't silently redefine whose body
    this is. Returns whether it wrote anything.
    """
    if get_profile(db, profile_id) is not None:
        return False
    save_profile(db, measurements, profile_id)
    return True


def apply_measurement_deltas(
    db: sqlite3.Connection,
    deltas: list[FieldDelta],
    profile_id: str = DEFAULT_PROFILE_ID,
) -> list[str]:
    """Step 5 of the intended flow: a revision that changes a measurement
    changes the body, so it updates the model.

    Only `measurements.*` paths are considered — a dart moving is a
    pattern change, not a body change, even though both arrive in the same
    revision. Returns the field names updated, so the caller can say what
    happened rather than silently mutating the profile.

    No-ops when there's no profile yet: there's nothing to correct, and
    inventing one from a partial delta set would produce a profile whose
    untouched fields were never actually measured.
    """
    current = get_profile(db, profile_id)
    if current is None:
        return []

    known_fields = {f.name for f in dataclasses.fields(Measurements)}
    updates: dict[str, float | None] = {}
    for delta in deltas:
        if not delta.path.startswith("measurements."):
            continue
        field = delta.path.split(".", 1)[1]
        if field in known_fields:
            updates[field] = delta.new

    if not updates:
        return []

    save_profile(db, dataclasses.replace(current, **updates), profile_id)
    return sorted(updates)
