"""The body model: current best known measurements for one body.

Distinct from a pattern version's `Measurements` snapshot, which is what
was true when that version was drafted.

Holds measured facts only — "needs 4mm more back shoulder dart" is an
inference from a fitting and belongs in api.learnings. Keeping them apart
is what makes the profile auditable.
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

    A submission omitting an optional measurement clears one that was
    genuinely taken, sending the draft back to estimating it. Reporting
    that makes the regression visible rather than silent.
    """
    changes: list[dict[str, object]] = []
    for f in dataclasses.fields(Measurements):
        new = getattr(after, f.name)
        old = getattr(before, f.name) if before is not None else None
        if _differs(old, new):
            changes.append({"field": f.name, "from": old, "to": new})
    return changes


def _differs(old: Optional[float], new: Optional[float]) -> bool:
    """Not `!=`: a value converted from inches stores as 800.0999999999999,
    displays as 80.01cm and returns as 800.1, so equality reported editing
    one field as changing three. The epsilon sits below the 0.1mm display
    rounding and the 0.1cm minimum edit the form can make.
    """
    if old is None or new is None:
        return old is not new
    return abs(old - new) > 0.01


def update_profile(
    db: sqlite3.Connection,
    measurements: Measurements,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> list[dict[str, object]]:
    """Replace the body model and report what changed.

    Unconditional: the measurements submitted when creating a pattern
    become the profile, so an edit made in that form sticks.
    """
    changes = profile_changes(get_profile(db, profile_id), measurements)
    save_profile(db, measurements, profile_id)
    return changes


def apply_measurement_deltas(
    db: sqlite3.Connection,
    deltas: list[FieldDelta],
    profile_id: str = DEFAULT_PROFILE_ID,
) -> list[str]:
    """A revision that changes a measurement changes the body.

    Only `measurements.*` paths count — a dart moving is a pattern change,
    not a body change, even arriving in the same revision. No-ops without a
    profile: inventing one from a partial delta set would leave untouched
    fields holding numbers nobody measured.
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
