"""Tests for the body model (PLAN.md Phase 9.1). No LLM, no network —
this module is pure storage plus one filter over revision deltas.
"""
import sqlite3

from api.body_profile import (
    apply_measurement_deltas, get_profile, save_profile, seed_profile_if_missing,
)
from api.schema import DDL, FieldDelta, Measurements


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(DDL)
    return conn


def _measurements(**overrides) -> Measurements:
    base = dict(
        bust=920, waist=740, hip=980, shoulder_width=380,
        back_length=400, bust_height=240, bust_span=180,
    )
    return Measurements(**{**base, **overrides})


def test_get_profile_is_none_before_anything_is_recorded():
    assert get_profile(_db()) is None


def test_save_and_read_back_round_trips():
    conn = _db()
    save_profile(conn, _measurements(bust=955))

    assert get_profile(conn).bust == 955


def test_save_profile_overwrites_rather_than_adding_a_second_row():
    conn = _db()
    save_profile(conn, _measurements(bust=920))
    save_profile(conn, _measurements(bust=955))

    assert conn.execute("SELECT COUNT(*) FROM body_profiles").fetchone() == (1,)
    assert get_profile(conn).bust == 955


def test_optional_fields_round_trip_including_none():
    """Measurements carries a lot of Optional fields (CLAUDE.md: extend as
    garment types need them). Storing JSON rather than columns means an
    unset field has to come back as None, not as a missing attribute."""
    conn = _db()
    save_profile(conn, _measurements(neck_circumference=360))

    profile = get_profile(conn)
    assert profile.neck_circumference == 360
    assert profile.arm_length is None


def test_seed_fills_a_missing_profile():
    conn = _db()

    assert seed_profile_if_missing(conn, _measurements(bust=920)) is True
    assert get_profile(conn).bust == 920


def test_seed_never_overwrites_an_existing_profile():
    """A later pattern drafted with deliberately different numbers — someone
    else's bodice, a test — must not silently redefine whose body this is."""
    conn = _db()
    save_profile(conn, _measurements(bust=920))

    assert seed_profile_if_missing(conn, _measurements(bust=700)) is False
    assert get_profile(conn).bust == 920


def test_measurement_deltas_update_the_profile():
    conn = _db()
    save_profile(conn, _measurements(bust=920, waist=740))

    updated = apply_measurement_deltas(conn, [
        FieldDelta("measurements.bust", 920, 955),
        FieldDelta("measurements.waist", 740, 765),
    ])

    assert updated == ["bust", "waist"]
    profile = get_profile(conn)
    assert profile.bust == 955
    assert profile.waist == 765


def test_non_measurement_deltas_leave_the_profile_alone():
    """A dart moving is a pattern change, not a body change — even though
    both arrive in the same revision."""
    conn = _db()
    save_profile(conn, _measurements(bust=920))

    updated = apply_measurement_deltas(conn, [
        FieldDelta("pieces[bodice_back].points[back_waist_dart_leg_a].x", 100.0, 104.0),
        FieldDelta("pieces[bodice_front].seam_allowance_mm", 15.0, 10.0),
    ])

    assert updated == []
    assert get_profile(conn).bust == 920


def test_a_mixed_revision_applies_only_its_measurement_half():
    conn = _db()
    save_profile(conn, _measurements(bust=920))

    updated = apply_measurement_deltas(conn, [
        FieldDelta("measurements.bust", 920, 955),
        FieldDelta("pieces[bodice_back].points[cb_neck].y", 0.0, 2.0),
    ])

    assert updated == ["bust"]
    assert get_profile(conn).bust == 955


def test_unknown_measurement_fields_are_ignored_not_crashed_on():
    """Stored revisions carry old paths forever (CLAUDE.md). A delta naming
    a field that no longer exists on Measurements must be skipped, not
    raise on dataclasses.replace."""
    conn = _db()
    save_profile(conn, _measurements(bust=920))

    updated = apply_measurement_deltas(conn, [
        FieldDelta("measurements.a_field_that_was_removed", 1, 2),
    ])

    assert updated == []
    assert get_profile(conn).bust == 920


def test_deltas_no_op_when_no_profile_exists_yet():
    """Nothing to correct, and inventing a profile from a partial delta set
    would leave untouched fields holding numbers nobody ever measured."""
    conn = _db()

    assert apply_measurement_deltas(conn, [FieldDelta("measurements.bust", 920, 955)]) == []
    assert get_profile(conn) is None
