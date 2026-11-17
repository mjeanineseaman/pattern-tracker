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


def profile_changes(
    before: Optional[Measurements], after: Measurements,
) -> list[dict[str, object]]:
    """What a save would change, as {field, from, to} entries.

    Exists so an overwrite is never silent. Submitting the create form
    replaces the profile wholesale, which is what makes the form the one
    place you maintain your measurements — but it also means a submission
    missing `armscye_depth` clears a value that was genuinely measured,
    and the draft quietly goes back to estimating it. Reporting the change
    (including a value becoming None) makes that visible and recoverable
    instead of a silent regression in fit.
    """
    changes: list[dict[str, object]] = []
    for f in dataclasses.fields(Measurements):
        new = getattr(after, f.name)
        old = getattr(before, f.name) if before is not None else None
        if _differs(old, new):
            changes.append({"field": f.name, "from": old, "to": new})
    return changes


def _differs(old: Optional[float], new: Optional[float]) -> bool:
    """Whether two stored measurements are meaningfully different.

    Not `!=`: a value converted from inches is stored as 800.0999999999999,
    shown in the UI as 80.01cm, and submitted back as 800.1. Plain equality
    reports that as an edit, so touching one field appeared to change three.
    The epsilon is far below both the 0.1mm display rounding and anything a
    tape measure resolves, so a real edit (0.1cm minimum through the form)
    can't hide underneath it.
    """
    if old is None or new is None:
        return old is not new
    return abs(old - new) > 0.01


def update_profile(
    db: sqlite3.Connection,
    measurements: Measurements,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> list[dict[str, object]]:
    """Replace the body model with these measurements, and report what
    changed.

    Unconditional by design: the measurements you submit when creating a
    pattern become your profile. An earlier version only seeded when no
    profile existed, which meant the form showed your measurements,
    accepted an edit to them, and then silently discarded it — the profile
    could only be corrected through a revision or a direct PUT.
    """
    changes = profile_changes(get_profile(db, profile_id), measurements)
    save_profile(db, measurements, profile_id)
    return changes


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
