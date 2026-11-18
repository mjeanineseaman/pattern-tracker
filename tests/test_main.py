"""Endpoint tests: a real in-memory SQLite DB seeded with a drafted
pattern, hit through the actual FastAPI routes rather than by calling the
underlying functions — so db.py, pattern_from_json and the exporters are
exercised wired together as they really run.

No network and no API key: `get_anthropic_client` is overridden with a
stub whose `.messages.create()` returns canned text.
"""
import sqlite3
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from api.drafting.bodice import draft_bodice
from api.main import app, get_anthropic_client, get_db
from api.schema import DDL, DesignIntent, FabricClass, GarmentType, Measurements, Pattern


class _FakeAnthropicClient:
    """Stands in for anthropic.Anthropic in tests: records the call it
    received and returns whatever text the test configured, without any
    network access or API key."""
    def __init__(self, reply_text: str):
        self._reply_text = reply_text
        self.last_create_kwargs: dict | None = None

        class _Messages:
            def create(inner_self, **kwargs):
                self.last_create_kwargs = kwargs
                return SimpleNamespace(content=[SimpleNamespace(text=self._reply_text)])

        self.messages = _Messages()


def _seeded_db():
    # FastAPI runs sync routes in a worker thread; a connection opened on the
    # test's own thread needs check_same_thread=False to be usable there.
    # get_connection() doesn't expose that (production opens one connection
    # per request, naturally thread-local), so this test builds its own.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(DDL)
    measurements = Measurements(
        bust=920, waist=740, hip=980, shoulder_width=380,
        back_length=400, bust_height=240, bust_span=180,
    )
    intent = DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)
    pattern = Pattern(
        id="pat_001", name="Test bodice", version=1,
        measurements=measurements, intent=intent,
        pieces=draft_bodice(measurements, intent),
    )
    conn.execute(
        "INSERT INTO patterns (id, name, current_version, created_at, garment_type) "
        "VALUES (?, ?, ?, datetime('now'), ?)",
        (pattern.id, pattern.name, pattern.version, intent.garment_type.value),
    )
    conn.execute(
        "INSERT INTO pattern_versions (pattern_id, version, snapshot_json) VALUES (?, ?, ?)",
        (pattern.id, pattern.version, pattern.to_json()),
    )
    conn.commit()
    return conn, pattern


def _client() -> TestClient:
    client, _conn, _pattern = _client_with_db()
    return client


def _client_with_db() -> tuple[TestClient, sqlite3.Connection, Pattern]:
    conn, pattern = _seeded_db()

    def override_get_db():
        yield conn

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app), conn, pattern


def _client_with_db_and_vision(reply_text: str) -> tuple[TestClient, sqlite3.Connection, Pattern, _FakeAnthropicClient]:
    client, conn, pattern = _client_with_db()
    fake_client = _FakeAnthropicClient(reply_text)

    def override_get_anthropic_client():
        return fake_client

    app.dependency_overrides[get_anthropic_client] = override_get_anthropic_client
    return client, conn, pattern, fake_client


@pytest.mark.parametrize("path", [
    "/patterns/pat_001/versions/99",
    "/patterns/pat_001/versions/99/pdf",
    "/patterns/pat_001/versions/99/outline",
])
def test_reading_a_missing_version_404s(path):
    """Each of these resolves its own snapshot, so each needs its own
    guard — listed together so a new version-scoped endpoint is obviously
    missing from the list."""
    assert _client().get(path).status_code == 404


@pytest.mark.parametrize("path", [
    "/patterns/does_not_exist",
    "/patterns/does_not_exist/revisions",
    "/patterns/does_not_exist/versions/1/pdf",
])
def test_reading_a_missing_pattern_404s(path):
    assert _client().get(path).status_code == 404


def test_pdf_endpoint_returns_pdf_for_an_existing_version():
    client = _client()
    response = client.get("/patterns/pat_001/versions/1/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_revise_creates_a_new_version_and_a_revision_row():
    client, conn, pattern = _client_with_db()
    revised = pattern.to_dict()
    revised["pieces"][0]["seam_allowance_mm"] = 20.0

    response = client.post(
        "/patterns/pat_001/revise",
        json={"pattern": revised, "reason": "wider seam allowance for a heavier fabric"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 2
    assert body["parent_version"] == 1
    assert body["change_type"] == "seam_allowance"
    assert body["source"] == "manual"
    assert body["reason"] == "wider seam allowance for a heavier fabric"
    assert len(body["deltas"]) == 1
    assert "seam_allowance_mm" in body["deltas"][0]["path"]

    assert conn.execute("SELECT current_version FROM patterns WHERE id = 'pat_001'").fetchone() == (2,)
    assert conn.execute(
        "SELECT version FROM pattern_versions WHERE pattern_id = 'pat_001' ORDER BY version",
    ).fetchall() == [(1,), (2,)]
    assert conn.execute("SELECT COUNT(*) FROM revisions WHERE pattern_id = 'pat_001'").fetchone() == (1,)


def test_revise_rejects_an_empty_diff():
    client, _conn, pattern = _client_with_db()

    response = client.post(
        "/patterns/pat_001/revise",
        json={"pattern": pattern.to_dict(), "reason": "nothing actually changed"},
    )

    assert response.status_code == 400


def test_revise_rejects_a_blank_reason():
    client, _conn, pattern = _client_with_db()
    revised = pattern.to_dict()
    revised["pieces"][0]["seam_allowance_mm"] = 20.0

    response = client.post("/patterns/pat_001/revise", json={"pattern": revised, "reason": "   "})

    assert response.status_code == 400


def test_revise_404s_for_a_missing_pattern():
    client, _conn, pattern = _client_with_db()

    response = client.post(
        "/patterns/does_not_exist/revise",
        json={"pattern": pattern.to_dict(), "reason": "doesn't matter"},
    )

    assert response.status_code == 404


def test_get_revisions_returns_newest_first():
    client, _conn, pattern = _client_with_db()
    first = pattern.to_dict()
    first["pieces"][0]["seam_allowance_mm"] = 20.0
    client.post("/patterns/pat_001/revise", json={"pattern": first, "reason": "first bump"})
    second = first.copy()
    second["pieces"] = [dict(p) for p in first["pieces"]]
    second["pieces"][0]["seam_allowance_mm"] = 25.0
    client.post("/patterns/pat_001/revise", json={"pattern": second, "reason": "second bump"})

    response = client.get("/patterns/pat_001/revisions")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert [r["version"] for r in body] == [3, 2]  # newest first
    assert body[0]["reason"] == "second bump"
    assert body[1]["reason"] == "first bump"


def test_get_revisions_is_an_empty_list_for_a_pattern_with_no_revisions_yet():
    client, _conn, _pattern = _client_with_db()

    response = client.get("/patterns/pat_001/revisions")

    assert response.status_code == 200
    assert response.json() == []


def test_diff_aggregates_directly_between_snapshots_not_per_revision():
    """A field nudged across two revisions (15mm -> 20mm -> 25mm) should
    show up in a from=1,to=3 diff as the one end-to-end change (15 -> 25),
    not two redundant entries concatenated from each revision's own
    stored deltas."""
    client, _conn, pattern = _client_with_db()
    first = pattern.to_dict()
    first["pieces"][0]["seam_allowance_mm"] = 20.0
    client.post("/patterns/pat_001/revise", json={"pattern": first, "reason": "first bump"})
    second = first.copy()
    second["pieces"] = [dict(p) for p in first["pieces"]]
    second["pieces"][0]["seam_allowance_mm"] = 25.0
    client.post("/patterns/pat_001/revise", json={"pattern": second, "reason": "second bump"})

    response = client.get("/patterns/pat_001/diff", params={"from": 1, "to": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["from"] == 1
    assert body["to"] == 3
    seam_deltas = [d for d in body["deltas"] if "seam_allowance_mm" in d["path"]]
    assert len(seam_deltas) == 1
    assert seam_deltas[0]["old"] == 15.0
    assert seam_deltas[0]["new"] == 25.0


def test_diff_404s_for_a_missing_version():
    client, _conn, _pattern = _client_with_db()

    response = client.get("/patterns/pat_001/diff", params={"from": 1, "to": 99})

    assert response.status_code == 404


def test_ask_endpoint_answers_from_a_revision_made_through_this_api():
    """End-to-end: a real revision created via POST /revise (which embeds
    it automatically) is retrievable and answerable via POST /ask — not a
    hand-seeded row, the actual path a real revision takes."""
    client, _conn, pattern, fake_client = _client_with_db_and_vision(
        "You widened the back waist dart by 5mm after the first muslin pulled.",
    )
    revised = pattern.to_dict()
    revised["pieces"][0]["seam_allowance_mm"] = 20.0
    client.post(
        "/patterns/pat_001/revise",
        json={"pattern": revised, "reason": "wider seam allowance for a heavier fabric"},
    )

    response = client.post("/patterns/pat_001/ask", json={"question": "why did the seam allowance change?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You widened the back waist dart by 5mm after the first muslin pulled."
    assert len(body["revisions_used"]) == 1
    assert body["revisions_used"][0]["reason"] == "wider seam allowance for a heavier fabric"
    assert fake_client.last_create_kwargs is not None


def test_ask_endpoint_404s_for_a_missing_pattern():
    client, _conn, _pattern, _fake = _client_with_db_and_vision("irrelevant")

    response = client.post("/patterns/does_not_exist/ask", json={"question": "anything"})

    assert response.status_code == 404


def test_ask_endpoint_does_not_call_the_llm_for_a_pattern_with_no_revisions():
    client, _conn, _pattern, fake_client = _client_with_db_and_vision("should never be returned")

    response = client.post("/patterns/pat_001/ask", json={"question": "anything"})

    assert response.status_code == 200
    assert response.json()["revisions_used"] == []
    assert fake_client.last_create_kwargs is None


def test_mark_sloper_marks_the_version_under_its_own_garment_type():
    client, conn, _pattern = _client_with_db()

    response = client.post("/patterns/pat_001/versions/1/mark-sloper")

    assert response.status_code == 200
    body = response.json()
    assert body["garment_type"] == "bodice"
    assert body["pattern_id"] == "pat_001"
    assert body["version"] == 1
    assert body["marked_at"]  # non-empty timestamp

    row = conn.execute(
        "SELECT garment_type, pattern_id, version FROM reference_slopers WHERE garment_type = 'bodice'",
    ).fetchone()
    assert row == ("bodice", "pat_001", 1)


def test_mark_sloper_replaces_the_previous_sloper_for_the_same_garment_type():
    """One sloper per garment_type (schema.py's PRIMARY KEY on garment_type)
    — marking a second version for the same type replaces the first, it
    doesn't add a second candidate row."""
    client, conn, _pattern = _client_with_db()
    conn.execute(
        "INSERT INTO pattern_versions (pattern_id, version, snapshot_json) "
        "SELECT pattern_id, 2, snapshot_json FROM pattern_versions WHERE pattern_id = 'pat_001' AND version = 1",
    )
    conn.commit()

    client.post("/patterns/pat_001/versions/1/mark-sloper")
    response = client.post("/patterns/pat_001/versions/2/mark-sloper")

    assert response.status_code == 200
    rows = conn.execute("SELECT version FROM reference_slopers WHERE garment_type = 'bodice'").fetchall()
    assert rows == [(2,)]


def test_mark_sloper_404s_for_a_missing_version():
    client = _client()
    response = client.post("/patterns/pat_001/versions/99/mark-sloper")

    assert response.status_code == 404


def test_get_sloper_returns_the_marked_patterns_pieces_measurements_and_intent():
    """Phase 4's actual use case: everything draft_bodice_from_sloper needs
    (sloper_pieces, sloper_measurements, sloper_intent) comes back in one
    round trip, not just a pointer to go resolve."""
    client, _conn, _pattern = _client_with_db()
    client.post("/patterns/pat_001/versions/1/mark-sloper")

    response = client.get("/slopers/bodice")

    assert response.status_code == 200
    body = response.json()
    assert body["garment_type"] == "bodice"
    assert body["pattern_id"] == "pat_001"
    assert body["version"] == 1
    assert body["marked_at"]
    assert body["pattern"]["measurements"]["bust"] == 920
    assert body["pattern"]["intent"]["garment_type"] == "bodice"
    piece_ids = {p["id"] for p in body["pattern"]["pieces"]}
    assert piece_ids == {"bodice_front", "bodice_back"}


def test_get_sloper_404s_when_none_marked_for_that_garment_type():
    client, _conn, _pattern = _client_with_db()  # seeded, but mark-sloper never called

    response = client.get("/slopers/bodice")

    assert response.status_code == 404


def test_get_sloper_422s_for_an_unrecognised_garment_type():
    client = _client()
    response = client.get("/slopers/not_a_real_garment_type")

    assert response.status_code == 422


# --------------------------------------------------------------------------
# POST /patterns, GET /patterns, GET /patterns/{id}, GET /patterns/{id}/versions/{v}
# (Phase 7's join point: the create-a-pattern step the frontend needs, which
# didn't exist until now even though every later phase assumed it).
# --------------------------------------------------------------------------
_NEW_PATTERN_BODY = {
    "name": "New bodice",
    "intent": {"garment_type": "bodice", "fabric_class": "woven"},
    "measurements": {
        "bust": 920, "waist": 740, "hip": 980, "shoulder_width": 380,
        "back_length": 400, "bust_height": 240, "bust_span": 180,
    },
}


def test_create_pattern_drafts_from_standard_formulas_when_no_sloper_is_marked():
    client, conn, _pattern = _client_with_db()

    response = client.post("/patterns", json=_NEW_PATTERN_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "New bodice"
    assert body["version"] == 1
    piece_ids = {p["id"] for p in body["pieces"]}
    assert piece_ids == {"bodice_front", "bodice_back"}

    row = conn.execute(
        "SELECT reason, source FROM revisions WHERE pattern_id = ?", (body["id"],),
    ).fetchone()
    assert "standard formulas" in row[0]
    assert row[1] == "croquis_draft"


def test_create_pattern_drafts_from_the_marked_sloper_when_one_exists():
    client, conn, _pattern = _client_with_db()
    client.post("/patterns/pat_001/versions/1/mark-sloper")

    response = client.post("/patterns", json=_NEW_PATTERN_BODY)

    assert response.status_code == 200
    body = response.json()
    row = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (body["id"],),
    ).fetchone()
    assert "reference sloper" in row[0]
    assert "pat_001 v1" in row[0]


def test_create_pattern_500s_if_the_marked_sloper_version_no_longer_exists():
    """Mirrors GET /slopers/{type}'s own "shouldn't be reachable" guard —
    nothing in this app deletes pattern versions, but reference_slopers
    pointing at a snapshot that's gone should 500 clearly, not crash on an
    unguarded None."""
    client, conn, _pattern = _client_with_db()
    client.post("/patterns/pat_001/versions/1/mark-sloper")
    conn.execute("DELETE FROM pattern_versions WHERE pattern_id = 'pat_001' AND version = 1")
    conn.commit()

    response = client.post("/patterns", json=_NEW_PATTERN_BODY)

    assert response.status_code == 500


def test_create_pattern_persists_so_it_can_be_fetched_back():
    client, _conn, _pattern = _client_with_db()
    created = client.post("/patterns", json=_NEW_PATTERN_BODY).json()

    response = client.get(f"/patterns/{created['id']}")

    assert response.status_code == 200
    # draft_note, measurement_needs and body_profile_updated describe *this
    # drafting run*, not the stored pattern, so they're on the create
    # response only.
    stored_fields = {k: v for k, v in created.items()
                     if k not in ("draft_note", "measurement_needs", "body_profile_updated")}
    assert response.json() == stored_fields


def test_list_patterns_returns_every_pattern_newest_first():
    client, _conn, _pattern = _client_with_db()
    client.post("/patterns", json=_NEW_PATTERN_BODY)

    response = client.get("/patterns")

    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert "pat_001" in ids
    assert len(ids) == 2


def test_get_pattern_returns_the_current_version():
    client = _client()
    response = client.get("/patterns/pat_001")

    assert response.status_code == 200
    assert response.json()["id"] == "pat_001"


def test_get_pattern_version_returns_a_specific_version():
    client = _client()
    response = client.get("/patterns/pat_001/versions/1")

    assert response.status_code == 200
    assert response.json()["version"] == 1


def test_get_outline_returns_flattened_polylines_for_every_piece():
    client = _client()
    response = client.get("/patterns/pat_001/versions/1/outline")

    assert response.status_code == 200
    body = response.json()
    assert body["pattern_id"] == "pat_001"
    assert body["version"] == 1
    piece_ids = {p["id"] for p in body["pieces"]}
    assert piece_ids == {"bodice_front", "bodice_back"}
    for piece in body["pieces"]:
        assert len(piece["stitching"]) > 2
        assert len(piece["cut"]) > 2
        assert all(len(pt) == 2 for pt in piece["stitching"])


def test_the_ui_is_served_at_the_root():
    """One command starts the whole tool, and the page ends up same-origin
    with the API it calls."""
    response = _client().get("/")

    assert response.status_code == 200
    assert "Pattern Tracker" in response.text


def test_the_static_mount_does_not_shadow_the_api():
    """A mount at "/" matches anything not already claimed, so declaring it
    before the routes would swallow every endpoint. This is the regression
    that would make the whole API disappear at once."""
    client = _client()

    for path in ("/patterns", "/garment-types", "/body-profile"):
        assert client.get(path).status_code in (200, 404), path
        assert "text/html" not in client.get(path).headers.get("content-type", "")


# --------------------------------------------------------------------------
# GET/PUT /body-profile and its two hooks (PLAN.md Phase 9.1)
# --------------------------------------------------------------------------
def test_get_body_profile_404s_before_anything_is_recorded():
    client = _client()
    assert client.get("/body-profile").status_code == 404


def test_put_then_get_body_profile_round_trips():
    client = _client()
    measurements = dict(_NEW_PATTERN_BODY["measurements"], bust=955)

    put = client.put("/body-profile", json=measurements)
    assert put.status_code == 200
    assert put.json()["updated_at"]

    assert client.get("/body-profile").json()["bust"] == 955


def test_creating_a_later_pattern_updates_the_body_profile():
    """Unconditional by design: this form is where measurements are
    maintained, so an edit made in it has to stick."""
    client = _client()
    client.post("/patterns", json=_NEW_PATTERN_BODY)

    response = client.post("/patterns", json={
        **_NEW_PATTERN_BODY,
        "measurements": dict(_NEW_PATTERN_BODY["measurements"], bust=955),
    })

    assert client.get("/body-profile").json()["bust"] == 955
    assert {"field": "bust", "from": 920, "to": 955} in response.json()["body_profile_updated"]


def test_creating_a_pattern_reports_a_measurement_it_cleared():
    """A submission omitting an optional measurement wipes one that was
    really taken. That has to come back visibly, not silently."""
    client = _client()
    client.post("/patterns", json={
        **_NEW_PATTERN_BODY,
        "measurements": dict(_NEW_PATTERN_BODY["measurements"], armscye_depth=190.5),
    })

    response = client.post("/patterns", json=_NEW_PATTERN_BODY)

    assert {"field": "armscye_depth", "from": 190.5, "to": None} \
        in response.json()["body_profile_updated"]


def test_revising_a_measurement_updates_the_body_profile():
    client, _conn, pattern = _client_with_db()
    client.put("/body-profile", json=_NEW_PATTERN_BODY["measurements"])

    edited = pattern.to_dict()
    edited["measurements"]["bust"] = 955

    response = client.post(f"/patterns/{pattern.id}/revise", json={
        "pattern": edited, "reason": "re-measured after losing the muslin ease",
    })

    assert response.status_code == 200
    assert response.json()["body_profile_updated"] == ["bust"]
    assert client.get("/body-profile").json()["bust"] == 955


def test_revising_geometry_alone_leaves_the_body_profile_untouched():
    client, _conn, pattern = _client_with_db()
    client.put("/body-profile", json=_NEW_PATTERN_BODY["measurements"])

    edited = pattern.to_dict()
    edited["pieces"][0]["points"][0]["x"] += 5

    response = client.post(f"/patterns/{pattern.id}/revise", json={
        "pattern": edited, "reason": "nudged a point",
    })

    assert response.json()["body_profile_updated"] == []
    assert client.get("/body-profile").json()["bust"] == 920


# --------------------------------------------------------------------------
# Learnings recorded from revisions (PLAN.md Phase 9.2)
# --------------------------------------------------------------------------
def _revise(client, pattern, reason, source, fit_notes="", nudge=1.0):
    edited = pattern.to_dict()
    edited["pieces"][0]["points"][0]["x"] += nudge
    return client.post(f"/patterns/{pattern.id}/revise", json={
        "pattern": edited, "reason": reason, "fit_notes": fit_notes, "source": source,
    })


def test_a_muslin_fitting_revision_records_a_learning():
    client, conn, pattern = _client_with_db()

    response = _revise(
        client, pattern,
        reason="widened the back waist dart",
        fit_notes="the back pulled across the shoulder blade",
        source="muslin_fitting",
    )

    assert response.json()["learning_recorded"]["times_confirmed"] == 1
    rows = conn.execute("SELECT garment_type, learning FROM garment_learnings").fetchall()
    assert rows[0][0] == "bodice"
    assert "shoulder blade" in rows[0][1]


def test_a_manual_revision_records_no_learning():
    """A design change isn't an observation about a body, so it must not
    propagate into later garments."""
    client, conn, pattern = _client_with_db()

    response = _revise(client, pattern, reason="raised the neckline for style", source="manual")

    assert response.json()["learning_recorded"] is None
    assert conn.execute("SELECT COUNT(*) FROM garment_learnings").fetchone() == (0,)


def test_patterns_created_before_phase_9_record_no_learning_rather_than_guessing():
    """_seeded_db inserts pat_001 without a garment_type, standing in for a
    row written before the column existed."""
    client, conn, pattern = _client_with_db()
    conn.execute("UPDATE patterns SET garment_type = NULL WHERE id = ?", (pattern.id,))
    conn.commit()

    response = _revise(
        client, pattern, reason="let out the side seam",
        fit_notes="tight across the ribs", source="muslin_fitting",
    )

    assert response.status_code == 200
    assert response.json()["learning_recorded"] is None


def test_creating_a_pattern_records_its_garment_type():
    client, conn, _pattern = _client_with_db()

    created = client.post("/patterns", json=_NEW_PATTERN_BODY).json()

    row = conn.execute("SELECT garment_type FROM patterns WHERE id = ?", (created["id"],)).fetchone()
    assert row[0] == "bodice"


# --------------------------------------------------------------------------
# Asking for measurements rather than guessing them
# --------------------------------------------------------------------------
def test_creating_a_pattern_reports_what_it_had_to_estimate():
    """Reported on the response and recorded in the revision reason —
    months later, "which numbers were real here?" is answerable from the
    history alone."""
    client, conn, _pattern = _client_with_db()

    created = client.post("/patterns", json=_NEW_PATTERN_BODY).json()

    fields = [n["field"] for n in created["measurement_needs"]]
    assert fields[0] == "shoulder_length"  # the fixed guess, asked for first
    assert "armscye_depth" in fields

    reason = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (created["id"],),
    ).fetchone()[0]
    assert "Estimated (not measured)" in reason
    assert "shoulder_length" in reason


def test_supplying_every_measurement_leaves_nothing_estimated():
    client, conn, _pattern = _client_with_db()
    complete = dict(
        _NEW_PATTERN_BODY["measurements"],
        shoulder_length=127, armscye_depth=190, back_width=330,
        nape_to_bust=280, neck_circumference=380,
    )

    created = client.post(
        "/patterns", json={**_NEW_PATTERN_BODY, "measurements": complete},
    ).json()

    assert created["measurement_needs"] == []
    reason = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (created["id"],),
    ).fetchone()[0]
    assert "Estimated" not in reason


def test_body_profile_needs_lists_what_to_measure_next():
    client = _client()
    client.put("/body-profile", json=_NEW_PATTERN_BODY["measurements"])

    response = client.get("/body-profile/needs")

    assert response.status_code == 200
    needs = response.json()["needs"]
    assert needs[0]["field"] == "shoulder_length"
    assert needs[0]["how_to_measure"]
    assert needs[0]["affects"]


def test_body_profile_needs_404s_without_a_profile():
    assert _client().get("/body-profile/needs").status_code == 404


# --------------------------------------------------------------------------
# What can actually be drafted, and what each type will need
# --------------------------------------------------------------------------
def test_drafting_an_unimplemented_garment_type_is_refused():
    """Drafting a bodice and labelling it "skirt" would be confidently the
    wrong garment, which is worse than refusing."""
    client = _client()

    response = client.post("/patterns", json={
        **_NEW_PATTERN_BODY,
        "intent": {"garment_type": "skirt", "fabric_class": "woven"},
    })

    assert response.status_code == 422
    assert "isn't drafted yet" in response.json()["detail"]
    assert "hip" in response.json()["detail"]  # says what it'll need when it is


def test_the_refusal_names_the_measurements_that_type_will_need():
    client = _client()

    detail = client.post("/patterns", json={
        **_NEW_PATTERN_BODY,
        "intent": {"garment_type": "shirt", "fabric_class": "woven"},
    }).json()["detail"]

    assert "shoulder_width" in detail
    assert "arm_length" in detail
    # names the missing block, not just the garment: half a shirt already
    # drafts, so "not supported" would hide what's actually outstanding
    assert "sleeve block" in detail


def test_a_bodice_drafts_without_hip_or_shoulder_width():
    """A bodice block with no sleeve is a sleeveless top — neither
    measurement is involved."""
    client = _client()
    body = {**_NEW_PATTERN_BODY}
    body["measurements"] = {
        k: v for k, v in body["measurements"].items()
        if k not in ("hip", "shoulder_width")
    }

    assert client.post("/patterns", json=body).status_code == 200


def test_garment_types_endpoint_reports_what_is_draftable_and_what_is_missing():
    """The realistic state: a profile measured for bodice work, with hip and
    shoulder_width genuinely not taken yet."""
    client = _client()
    bodice_only = {
        k: v for k, v in _NEW_PATTERN_BODY["measurements"].items()
        if k not in ("hip", "shoulder_width")
    }
    client.put("/body-profile", json=bodice_only)

    types = {t["garment_type"]: t for t in client.get("/garment-types").json()}

    assert types["bodice"]["draftable"] is True
    assert types["bodice"]["missing"] == []
    assert types["skirt"]["draftable"] is False
    assert "hip" in types["skirt"]["missing"]

    shirt = types["shirt"]
    assert shirt["blocks"] == ["bodice", "sleeve"]
    assert shirt["missing_blocks"] == ["sleeve"]
    assert "shoulder_width" in shirt["missing"]
    assert "bust" not in shirt["missing"]  # the bodice half is measured


# --------------------------------------------------------------------------
# The Phase 9 loop end to end (PLAN.md Phase 9 "Done when")
# --------------------------------------------------------------------------
def test_a_second_garment_starts_from_what_the_first_fitting_taught():
    """The whole point of Phase 9: draft, fit, record what the muslin did,
    and the next garment of that type starts adjusted rather than from the
    formulas alone — with the reason saying exactly what was applied."""
    reply = (
        '[{"piece": "bodice_back", "point": "cb_neck", "dx": 0, "dy": 4, '
        '"reason": "back pulled across the shoulder blade last time"}]'
    )
    client, conn, pattern, _fake = _client_with_db_and_vision(reply)

    # a real fitting on the first pattern becomes a learning
    edited = pattern.to_dict()
    edited["pieces"][0]["points"][0]["x"] += 2
    client.post(f"/patterns/{pattern.id}/revise", json={
        "pattern": edited,
        "reason": "widened the back waist dart",
        "fit_notes": "the back pulled across the shoulder blade",
        "source": "muslin_fitting",
    })
    assert conn.execute("SELECT COUNT(*) FROM garment_learnings").fetchone() == (1,)

    # the next bodice is drafted with that learning in hand
    created = client.post("/patterns", json=_NEW_PATTERN_BODY).json()

    reason = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (created["id"],),
    ).fetchone()[0]
    assert "Adjusted from fitting history" in reason
    assert "shoulder blade" in reason

    # and the geometry really did move
    plain = draft_bodice(pattern.measurements, pattern.intent)
    plain_cb_neck = next(
        p for p in next(x for x in plain if x.id == "bodice_back").points if p.id == "cb_neck"
    )
    created_cb_neck = next(
        p for p in next(x for x in created["pieces"] if x["id"] == "bodice_back")["points"]
        if p["id"] == "cb_neck"
    )
    assert created_cb_neck["y"] == plain_cb_neck.y + 4


def test_a_first_garment_is_drafted_without_calling_the_model_at_all():
    """No fitting history means nothing to apply, so Phase 9.4
    short-circuits rather than spending a call to be told nothing."""
    client, _conn, _pattern, fake = _client_with_db_and_vision("[]")

    created = client.post("/patterns", json=_NEW_PATTERN_BODY).json()

    assert fake.last_create_kwargs is None
    assert created["pieces"]


def test_creation_still_succeeds_when_the_model_call_fails():
    """Drafting must not become unavailable because an API call did."""
    client, conn, pattern = _client_with_db()

    class _ExplodingClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("Could not resolve authentication method")

    app.dependency_overrides[get_anthropic_client] = lambda: _ExplodingClient()

    edited = pattern.to_dict()
    edited["pieces"][0]["points"][0]["x"] += 2
    client.post(f"/patterns/{pattern.id}/revise", json={
        "pattern": edited, "reason": "let out the side seam",
        "fit_notes": "tight across the ribs", "source": "muslin_fitting",
    })

    response = client.post("/patterns", json=_NEW_PATTERN_BODY)

    assert response.status_code == 200
    reason = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (response.json()["id"],),
    ).fetchone()[0]
    assert "the adjustment step failed" in reason


# --------------------------------------------------------------------------
# The Phase 10 design step, wired into POST /patterns ahead of Phase 9.4's
# fit step. _PNG_BYTES and the CROQUIS_DIR monkeypatch are shared with the
# POST /croquis tests below; defined there, used here too.
# --------------------------------------------------------------------------
def _uploaded_croquis_ref(client, tmp_path, monkeypatch) -> str:
    import api.main as main_module
    monkeypatch.setattr(main_module, "CROQUIS_DIR", tmp_path / "croquis")
    response = client.post(
        "/croquis", files={"file": ("sketch.png", _PNG_BYTES, "image/png")},
    )
    return response.json()["croquis_ref"]


def test_create_pattern_designs_from_an_uploaded_croquis(tmp_path, monkeypatch):
    """The whole point of Phase 10: a sketch on a brand-new pattern — no
    fitting history at all — still changes the geometry, because the
    design step isn't gated on learnings the way the fit step is."""
    reply = '{"neckline_drop_front_mm": 80, "hem_change_mm": -40}'
    client, conn, _pattern, fake_client = _client_with_db_and_vision(reply)
    croquis_ref = _uploaded_croquis_ref(client, tmp_path, monkeypatch)
    body = {**_NEW_PATTERN_BODY, "croquis_ref": croquis_ref}

    created = client.post("/patterns", json=body).json()

    assert fake_client.last_create_kwargs is not None
    reason = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (created["id"],),
    ).fetchone()[0]
    assert "Designed from the croquis" in reason
    assert "front neckline 80mm lower" in reason

    plain = draft_bodice(Measurements(**_NEW_PATTERN_BODY["measurements"]),
                          DesignIntent(GarmentType.BODICE, FabricClass.WOVEN))
    plain_front = next(p for p in plain if p.id == "bodice_front")
    created_front = next(p for p in created["pieces"] if p["id"] == "bodice_front")
    plain_cf_neck = next(p for p in plain_front.points if p.id == "cf_neck")
    created_cf_neck = next(p for p in created_front["points"] if p["id"] == "cf_neck")
    assert created_cf_neck["y"] == plain_cf_neck.y + 80


def test_create_pattern_with_no_croquis_skips_the_design_step():
    """No sketch on file -- same as a fresh pattern with no fitting
    history for the fit step -- is nothing to design from, not a failure."""
    client, conn, _pattern, fake_client = _client_with_db_and_vision("irrelevant")

    created = client.post("/patterns", json=_NEW_PATTERN_BODY).json()

    assert fake_client.last_create_kwargs is None  # the fit step short-circuits too
    reason = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (created["id"],),
    ).fetchone()[0]
    assert "nothing to design from" in reason


def test_create_pattern_stores_the_eased_intent_a_released_dart_adds(tmp_path, monkeypatch):
    """waist_suppression releases a dart's intake, which adds girth the
    validator has to be told about -- the stored snapshot's intent has to
    carry that ease forward, or re-validating this exact pattern later
    would check its geometry against the wrong number and fail it."""
    reply = '{"waist_suppression": 0.0}'
    client, _conn, _pattern, _fake = _client_with_db_and_vision(reply)
    croquis_ref = _uploaded_croquis_ref(client, tmp_path, monkeypatch)
    body = {**_NEW_PATTERN_BODY, "croquis_ref": croquis_ref}

    created = client.post("/patterns", json=body).json()

    assert created["intent"]["ease_waist_mm"] > DesignIntent(
        GarmentType.BODICE, FabricClass.WOVEN,
    ).ease_waist_mm
    # and what's fetched back matches what was stored, same ease included
    fetched = client.get(f"/patterns/{created['id']}").json()
    assert fetched["intent"]["ease_waist_mm"] == created["intent"]["ease_waist_mm"]


def test_a_bad_design_proposal_does_not_discard_a_good_fit_correction(tmp_path, monkeypatch):
    """PLAN.md's whole reason for two calls instead of one: 'Failure
    isolates. A design proposal that fails validation shouldn't throw away
    a good fit correction, or vice versa.' The design call's garbled reply
    falls back on its own; the fit call right after it, on the same
    pattern, still gets to apply a real learning."""
    import api.main as main_module

    class _SequentialClient:
        """Returns a different canned reply each call -- the design step's
        call first, then the fit step's -- rather than one reply forced to
        answer both, which is all the shared _FakeAnthropicClient can do."""
        def __init__(self, replies):
            self._replies = iter(replies)
            self.calls: list[dict] = []

            class _Messages:
                def create(inner_self, **kwargs):
                    self.calls.append(kwargs)
                    return SimpleNamespace(content=[SimpleNamespace(text=next(self._replies))])

            self.messages = _Messages()

    fake_client = _SequentialClient([
        "not a design object, nothing here parses as JSON",
        '[{"piece": "bodice_back", "point": "cb_neck", "dx": 0, "dy": 4, '
        '"reason": "back pulled across the shoulder blade last time"}]',
    ])
    app.dependency_overrides[main_module.get_anthropic_client] = lambda: fake_client
    client, conn, pattern = _client_with_db()
    edited = pattern.to_dict()
    edited["pieces"][0]["points"][0]["x"] += 2
    client.post(f"/patterns/{pattern.id}/revise", json={
        "pattern": edited, "reason": "widened the back waist dart",
        "fit_notes": "the back pulled across the shoulder blade",
        "source": "muslin_fitting",
    })
    croquis_ref = _uploaded_croquis_ref(client, tmp_path, monkeypatch)
    body = {**_NEW_PATTERN_BODY, "croquis_ref": croquis_ref}

    created = client.post("/patterns", json=body)

    assert created.status_code == 200
    assert len(fake_client.calls) == 2
    reason = conn.execute(
        "SELECT reason FROM revisions WHERE pattern_id = ?", (created.json()["id"],),
    ).fetchone()[0]
    assert "the design proposal was unusable" in reason
    assert "Adjusted from fitting history" in reason
    assert "shoulder blade" in reason


# --------------------------------------------------------------------------
# POST /croquis — a plain upload since Phase 9 ("Design intent is
# author-written"): it stores the sketch and returns a ref, and makes no
# vision call at all. The sketch reaches a model later, in the drafting
# step, where it informs geometry rather than deciding what the garment is.
# CROQUIS_DIR is isolated to a pytest tmp_path so tests never write into
# the real repo's croquis/ folder.
# --------------------------------------------------------------------------
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)  # smallest possible valid 1x1 PNG, not a real sketch -- just needs to be non-empty bytes


def test_croquis_endpoint_stores_the_image_and_returns_its_ref(tmp_path, monkeypatch):
    import api.main as main_module
    monkeypatch.setattr(main_module, "CROQUIS_DIR", tmp_path / "croquis")
    client = _client()

    response = client.post(
        "/croquis", files={"file": ("sketch.png", _PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()

    # exactly {croquis_ref} — uploading a sketch must not come back with
    # garment_type/fit guesses for a person to correct. Intent is authored.
    assert set(body) == {"croquis_ref"}

    stored = (tmp_path / "croquis").parent / body["croquis_ref"]
    assert stored.exists()
    assert stored.read_bytes() == _PNG_BYTES


def test_croquis_endpoint_needs_no_anthropic_client_at_all(tmp_path, monkeypatch):
    """Storing a sketch makes no LLM call, so it works with no API key and
    no network — the dependency override is deliberately NOT installed
    here, and a real Anthropic() would raise if one were constructed."""
    import api.main as main_module
    monkeypatch.setattr(main_module, "CROQUIS_DIR", tmp_path / "croquis")
    app.dependency_overrides.pop(get_anthropic_client, None)
    client = _client()

    response = client.post(
        "/croquis", files={"file": ("sketch.png", _PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200


def test_croquis_endpoint_400s_for_an_empty_file(tmp_path, monkeypatch):
    import api.main as main_module
    monkeypatch.setattr(main_module, "CROQUIS_DIR", tmp_path / "croquis")
    client = _client()

    response = client.post("/croquis", files={"file": ("sketch.png", b"", "image/png")})

    assert response.status_code == 400
