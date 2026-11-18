"""FastAPI app. Currently exposes pattern creation/listing/reading, the
Phase 3 PDF export endpoint, all of Phase 5's revision API (revise,
revisions, diff, mark-sloper), the Phase 4 lookup that reads mark-sloper's
data, a sketch of Phase 4's own croquis intake, and Phase 6's /ask
endpoint over the RAG layer.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from anthropic import Anthropic
from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass

from api.body_profile import (
    apply_measurement_deltas, get_profile, save_profile, update_profile,
)
from api.db import get_connection
from api.drafting.ai_adjust import adjust_draft
from api.drafting.bodice import draft_bodice, draft_bodice_from_sloper
from api.drafting.design_from_croquis import design_from_croquis
from api.drafting.needs import (
    garment_blocks, is_draftable, measurement_needs, missing_blocks,
    missing_requirements, needs_to_dict, required_measurements,
)
from api.learnings import learnings_for, record_learning
from api.export.geometry import dart_lines, offset_polyline, piece_outline
from api.export.pdf import render_pattern_pdf
from api.rag.answer import answer_question
from api.rag.embed import embed_and_store_revision
from api.schema import (
    ChangeType, DesignIntent, FieldDelta, GarmentType, Measurements, Pattern, Revision,
    RevisionSource, diff_patterns, infer_change_type, pattern_from_json, revision_from_row,
)

app = FastAPI(title="Pattern Tracker")

# The UI in web/ is served by this same app, so the ordinary setup is
# same-origin and needs no CORS at all. This stays for the cases that
# aren't: opening web/index.html straight off disk during development, or
# pointing the UI's API field at a differently-ported instance. Scoped to
# localhost/127.0.0.1 rather than "*", since this API only ever runs on
# your own machine (CLAUDE.md: never deployed live) and no non-local page
# has any business calling it.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

CROQUIS_DIR = Path(__file__).resolve().parent.parent / "croquis"
_measurements_adapter = TypeAdapter(Measurements)
_revision_adapter = TypeAdapter(Revision)
_field_delta_list_adapter = TypeAdapter(list[FieldDelta])


def get_db() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _croquis_path(croquis_ref: str | None) -> Path | None:
    """Resolve a stored croquis ref to a real path, refusing anything that
    escapes CROQUIS_DIR. The ref comes back from POST /croquis, but it
    arrives here as an ordinary client-supplied string, so it's treated as
    untrusted input rather than trusted because of where it usually
    originates."""
    if not croquis_ref:
        return None
    root = CROQUIS_DIR.resolve()
    candidate = (CROQUIS_DIR.parent / croquis_ref).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


def get_anthropic_client() -> Anthropic:
    """Reads ANTHROPIC_API_KEY from the environment (api.db's load_dotenv()
    call, already imported above, has put it there by the time any route
    runs). Overridden in tests with a stub so the vision call never
    actually hits the network or needs a real key."""
    return Anthropic()


@pydantic_dataclass
class CreatePatternRequest:
    """The Phase 4/6 join point this project didn't have yet: an approved
    DesignIntent (the croquis review screen's output, once a human has
    confirmed/corrected it — never applied automatically, see
    api.main.post_croquis) plus measurements, turned into an actual
    pattern version 1."""
    name: str
    intent: DesignIntent
    measurements: Measurements
    croquis_ref: str | None = None


@app.get("/body-profile")
def get_body_profile(db: sqlite3.Connection = Depends(get_db)) -> dict:
    """PLAN.md Phase 9.1: the current best known measurements for this body.

    404s when nothing's been recorded yet — a real, distinct state from
    "recorded, and happens to be empty," and it lets the frontend tell the
    difference between "measure yourself first" and "here are your numbers."
    """
    profile = get_profile(db)
    if profile is None:
        raise HTTPException(status_code=404, detail="no body profile recorded yet")
    return _measurements_adapter.dump_python(profile, mode="json")


@app.put("/body-profile")
def put_body_profile(
    measurements: Measurements, db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """Record or correct the body model directly, rather than as a
    side effect of revising a pattern."""
    updated_at = save_profile(db, measurements)
    db.commit()
    return {
        "measurements": _measurements_adapter.dump_python(measurements, mode="json"),
        "updated_at": updated_at,
        "measurement_needs": needs_to_dict(measurement_needs(measurements)),
    }


@app.get("/garment-types")
def get_garment_types(db: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    """What can be drafted today, and what each type will need when it can.

    The bodice block with no sleeve is a sleeveless top — that's the honest
    extent of this tool right now. Reported rather than hardcoded in the UI
    so the two can't drift apart.
    """
    profile = get_profile(db)
    out = []
    for garment_type in GarmentType:
        required = required_measurements(garment_type)
        missing = missing_requirements(profile, garment_type) if profile else list(required)
        out.append({
            "garment_type": garment_type.value,
            "draftable": is_draftable(garment_type),
            "blocks": [b.value for b in garment_blocks(garment_type)],
            "missing_blocks": [b.value for b in missing_blocks(garment_type)],
            "requires": list(required),
            "missing": missing,
        })
    return out


@app.get("/body-profile/needs")
def get_body_profile_needs(
    garment_type: GarmentType = Query(GarmentType.BODICE),
    db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """Which measurements the draft is currently estimating, worst guess
    first.

    The alternative to asking is tuning a fallback formula until the output
    looks plausible, which is exactly the "averaging remembered numbers"
    CLAUDE.md warns against. A measurement costs one pass with a tape; a
    guess costs a muslin.
    """
    profile = get_profile(db)
    if profile is None:
        raise HTTPException(status_code=404, detail="no body profile recorded yet")
    return {
        "garment_type": garment_type.value,
        "needs": needs_to_dict(measurement_needs(profile, garment_type)),
    }


@app.post("/patterns")
def create_pattern(
    body: CreatePatternRequest,
    db: sqlite3.Connection = Depends(get_db),
    client: Anthropic = Depends(get_anthropic_client),
) -> dict:
    """PLAN.md Phase 4 + 6 + 9.4 + 10. Four steps, in order:

    1. Deterministic draft — from a marked reference sloper's fitted
       geometry if one exists for this garment type, otherwise from the
       standard formulas.
    2. Design step (Phase 10) — a separate model call reads the croquis,
       if one was uploaded, and proposes named Tier 1 parameters (a
       neckline drop, a cropped hem, and so on). This is what makes two
       different sketches of the same body produce different patterns;
       without it the sketch was stored and never read. Not gated on
       fitting history, so it runs on a pattern's very first version.
    3. Fit adjustment (Phase 9.4) — a second, separate model call applies
       what previous muslin fittings taught us about this body, as
       bounded point deltas, to whatever step 2 produced. Design is about
       the garment; fit correction is about the body, which is why they
       stay two calls rather than one and why fit goes last — a learning
       like "the back pulls across the shoulder blade" applies to the
       final silhouette, not to a block the design step is about to
       reshape.
    4. Write version 1 plus a CROQUIS_DRAFT revision whose reason records
       which path was taken and what, if anything, each model changed —
       their contribution is visible in the timeline rather than baked
       invisibly into the geometry.

    Steps 2 and 3 each validate their own output and fall back to their
    own input on any failure, so a bad design proposal can't discard a
    good fit correction or vice versa, and neither step can make drafting
    unavailable.
    """
    # Only the bodice block is implemented. Drafting a bodice and labelling
    # it "skirt" would produce a pattern that is confidently the wrong
    # garment — refuse instead (CLAUDE.md: never report confident nonsense).
    if not is_draftable(body.intent.garment_type):
        unbuilt = missing_blocks(body.intent.garment_type)
        raise HTTPException(
            status_code=422,
            detail=(
                f"{body.intent.garment_type.value} isn't drafted yet — it needs the "
                f"{', '.join(b.value for b in unbuilt)} block"
                f"{'s' if len(unbuilt) > 1 else ''}, which "
                f"{'are' if len(unbuilt) > 1 else 'is'} not built. This tool currently "
                f"drafts the bodice block (a sleeveless top). When it is, it will need "
                f"these measurements: "
                f"{', '.join(required_measurements(body.intent.garment_type))}."
            ),
        )

    # A measurement a garment can't be drafted without is a blocker, not a
    # quality cost — distinct from the estimated-fallback list reported below.
    missing = missing_requirements(body.measurements, body.intent.garment_type)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a {body.intent.garment_type.value} can't be drafted without: "
                f"{', '.join(missing)}. Measure and add them to your body profile first."
            ),
        )

    sloper_row = db.execute(
        "SELECT pattern_id, version FROM reference_slopers WHERE garment_type = ?",
        (body.intent.garment_type.value,),
    ).fetchone()

    if sloper_row is not None:
        sloper_pattern_id, sloper_version = sloper_row
        sloper_snap = db.execute(
            "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
            (sloper_pattern_id, sloper_version),
        ).fetchone()
        if sloper_snap is None:
            # Same "shouldn't be reachable" case GET /slopers/{type} already
            # guards against (see its docstring) — nothing in this app
            # deletes pattern versions, so this is a clear 500 rather than a
            # confusing downstream crash the day that invariant breaks.
            raise HTTPException(
                status_code=500,
                detail=f"reference sloper for {body.intent.garment_type.value} points at "
                       f"{sloper_pattern_id} v{sloper_version}, which no longer exists",
            )
        sloper_pattern = pattern_from_json(sloper_snap[0])
        pieces = draft_bodice_from_sloper(
            sloper_pattern.pieces, sloper_pattern.measurements, sloper_pattern.intent,
            body.measurements, body.intent,
        )
        draft_note = f"Drafted from the reference sloper ({sloper_pattern_id} v{sloper_version})."
    else:
        pieces = draft_bodice(body.measurements, body.intent)
        draft_note = "Drafted from the standard formulas (no reference sloper marked for this garment type yet)."

    # Phase 10. Gated on the croquis existing, not on fitting history — this
    # is the step that lets a brand-new pattern's sketch actually change the
    # garment. `intent` may come back different from `body.intent`: a
    # released waist dart (waist_suppression) adds ease, and that has to
    # travel with the geometry into both the fit step below and the stored
    # snapshot, or a later re-validation would check the geometry against
    # the wrong ease and fail on a pattern that was correct when drafted.
    design_outcome = design_from_croquis(
        client, pieces, body.measurements, body.intent,
        croquis_path=_croquis_path(body.croquis_ref),
    )
    pieces, intent = design_outcome.pieces, design_outcome.intent
    draft_note = f"{draft_note} {design_outcome.note}"

    # Phase 9.4. Learnings come from muslin fittings only (api.learnings),
    # so with no fitting history this short-circuits without an API call.
    # Runs against the design step's own output, not the bare formula
    # draft — a learning about this body applies to whatever the sketch
    # just became, not to the un-styled block the sketch may have changed.
    garment_type = intent.garment_type.value
    outcome = adjust_draft(
        client,
        pieces,
        body.measurements,
        intent,
        [l.learning for l in learnings_for(db, garment_type)],
        croquis_path=_croquis_path(body.croquis_ref),
    )
    pieces = outcome.pieces
    draft_note = f"{draft_note} {outcome.note}"

    # Record what the draft had to estimate, in the revision reason rather
    # than only in the response — six months on, "which numbers were real
    # when this was drafted?" is exactly the question the history should be
    # able to answer about a pattern that didn't fit.
    needs = measurement_needs(body.measurements, body.intent.garment_type)
    if needs:
        draft_note = (
            f"{draft_note} Estimated (not measured): "
            f"{', '.join(n.field for n in needs)}."
        )

    pattern_id = uuid.uuid4().hex
    pattern = Pattern(
        id=pattern_id, name=body.name, version=1,
        # `intent`, not `body.intent`: the design step may have raised
        # ease_waist_mm to account for a dart it released, and the stored
        # snapshot has to agree with the geometry it's paired with, or a
        # later check_ease() runs against the wrong number.
        measurements=body.measurements, intent=intent, pieces=pieces,
        croquis_ref=body.croquis_ref,
    )

    db.execute(
        "INSERT INTO patterns (id, name, current_version, created_at, garment_type) "
        "VALUES (?, ?, 1, datetime('now'), ?)",
        (pattern_id, body.name, body.intent.garment_type.value),
    )
    db.execute(
        "INSERT INTO pattern_versions (pattern_id, version, snapshot_json) VALUES (?, 1, ?)",
        (pattern_id, pattern.to_json()),
    )
    revision = Revision(
        id=uuid.uuid4().hex, pattern_id=pattern_id, version=1, parent_version=None,
        change_type=ChangeType.OTHER, source=RevisionSource.CROQUIS_DRAFT,
        deltas=[], reason=draft_note,
    )
    db.execute(
        """
        INSERT INTO revisions
            (id, pattern_id, version, parent_version, change_type, source,
             deltas_json, reason, fit_notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, '[]', ?, '', ?)
        """,
        (
            revision.id, revision.pattern_id, revision.version, revision.parent_version,
            revision.change_type.value, revision.source.value, revision.reason, revision.created_at,
        ),
    )
    embed_and_store_revision(db, revision)
    # The measurements submitted here become the body model, every time —
    # this form is where measurements are maintained, so an edit made in it
    # has to stick. Changes are reported back rather than applied silently,
    # since a submission omitting an optional measurement clears one that
    # was really taken.
    profile_updated = update_profile(db, body.measurements)
    db.commit()

    return {
        **pattern.to_dict(),
        "draft_note": draft_note,
        "measurement_needs": needs_to_dict(needs),
        "body_profile_updated": profile_updated,
    }


@app.get("/patterns")
def list_patterns(db: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    rows = db.execute(
        "SELECT id, name, current_version, created_at FROM patterns ORDER BY created_at DESC",
    ).fetchall()
    return [
        {"id": id_, "name": name, "current_version": current_version, "created_at": created_at}
        for id_, name, current_version, created_at in rows
    ]


@app.get("/patterns/{pattern_id}")
def get_pattern(pattern_id: str, db: sqlite3.Connection = Depends(get_db)) -> dict:
    """The pattern's CURRENT version. For a specific older version, see
    GET /patterns/{id}/versions/{version} below."""
    row = db.execute("SELECT current_version FROM patterns WHERE id = ?", (pattern_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="pattern not found")
    return get_pattern_version(pattern_id, row[0], db)


@app.get("/patterns/{pattern_id}/versions/{version}")
def get_pattern_version(pattern_id: str, version: int, db: sqlite3.Connection = Depends(get_db)) -> dict:
    row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, version),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="pattern version not found")
    return pattern_from_json(row[0]).to_dict()


@app.get("/patterns/{pattern_id}/versions/{version}/outline")
def get_pattern_outline(
    pattern_id: str, version: int, db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """Flattened geometry for the Phase 7 pattern viewer: curves already
    walked down to polylines by api.export.geometry's own (tested) math,
    so the frontend only ever draws straight-line SVG — no bezier/dart/
    seam-allowance logic duplicated in JS. Two versions' outputs share the
    same (x, y) millimetre space, so the frontend can overlay them in
    different colours without any extra alignment step."""
    row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, version),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="pattern version not found")

    pattern = pattern_from_json(row[0])
    pieces = []
    for piece in pattern.pieces:
        stitching = piece_outline(piece)
        cut = offset_polyline(stitching, piece.seam_allowance_mm)
        grainline = None
        if piece.grainline is not None:
            pm = piece.point_map()
            g0, g1 = pm[piece.grainline.start], pm[piece.grainline.end]
            grainline = [[g0.x, g0.y], [g1.x, g1.y]]
        pieces.append({
            "id": piece.id,
            "name": piece.name,
            "cut_count": piece.cut_count,
            "cut_mirrored": piece.cut_mirrored,
            "seam_allowance_mm": piece.seam_allowance_mm,
            "stitching": [[x, y] for x, y in stitching],
            "cut": [[x, y] for x, y in cut],
            "darts": [[[ax, ay], [bx, by]] for (ax, ay), (bx, by) in dart_lines(piece)],
            "grainline": grainline,
        })
    return {"pattern_id": pattern_id, "version": version, "pieces": pieces}


@app.get("/patterns/{pattern_id}/versions/{version}/pdf")
def get_pattern_pdf(
    pattern_id: str, version: int, db: sqlite3.Connection = Depends(get_db),
) -> Response:
    row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, version),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="pattern version not found")

    pattern = pattern_from_json(row[0])
    pdf_bytes = render_pattern_pdf(pattern.pieces)
    return Response(content=pdf_bytes, media_type="application/pdf")


@pydantic_dataclass
class ReviseRequest:
    """POST /patterns/{id}/revise's body: the pattern as you want it to
    look after this edit, plus why. `pattern.version`/`pattern.id` are
    ignored in favour of the URL's pattern_id and the next version this
    endpoint computes itself — the caller edits content, not bookkeeping."""
    pattern: Pattern
    reason: str
    fit_notes: str = ""
    source: RevisionSource = RevisionSource.MANUAL


@app.post("/patterns/{pattern_id}/revise")
def revise_pattern(
    pattern_id: str, body: ReviseRequest, db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """PLAN.md Phase 5: diff the submitted pattern against the current
    version, infer what kind of change it is, write a new snapshot and a
    Revision row, bump current_version. Rejects exactly the two cases
    PLAN.md calls out explicitly — an empty diff, and a blank reason —
    because an unexplained or no-op revision is invisible to (or actively
    misleading for) the RAG layer that indexes `reason`/`fit_notes` later.
    """
    row = db.execute(
        "SELECT current_version, garment_type FROM patterns WHERE id = ?", (pattern_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="pattern not found")
    current_version, garment_type = row

    snap_row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, current_version),
    ).fetchone()
    if snap_row is None:
        raise HTTPException(
            status_code=500,
            detail=f"{pattern_id}: current_version {current_version} has no stored snapshot",
        )
    old = pattern_from_json(snap_row[0])

    if not body.reason.strip():
        raise HTTPException(status_code=400, detail="reason is required")

    deltas = diff_patterns(old, body.pattern)
    if not deltas:
        raise HTTPException(status_code=400, detail="no changes to revise")

    next_version = current_version + 1
    new_pattern = dataclasses.replace(body.pattern, id=pattern_id, version=next_version)
    change_type = infer_change_type(deltas)
    revision = Revision(
        id=uuid.uuid4().hex,
        pattern_id=pattern_id,
        version=next_version,
        parent_version=current_version,
        change_type=change_type,
        source=body.source,
        deltas=deltas,
        reason=body.reason,
        fit_notes=body.fit_notes,
    )

    db.execute(
        "INSERT INTO pattern_versions (pattern_id, version, snapshot_json) VALUES (?, ?, ?)",
        (pattern_id, next_version, new_pattern.to_json()),
    )
    db.execute("UPDATE patterns SET current_version = ? WHERE id = ?", (next_version, pattern_id))
    db.execute(
        """
        INSERT INTO revisions
            (id, pattern_id, version, parent_version, change_type, source,
             deltas_json, reason, fit_notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            revision.id, revision.pattern_id, revision.version, revision.parent_version,
            revision.change_type.value, revision.source.value,
            json.dumps(_field_delta_list_adapter.dump_python(deltas, mode="json")),
            revision.reason, revision.fit_notes, revision.created_at,
        ),
    )
    embed_and_store_revision(db, revision)  # same transaction — embedded the moment it exists
    # Phase 9.1: a revision that changes a measurement changes the body, so
    # the body model follows. Only measurements.* paths count — a dart
    # moving in the same edit is a pattern change, not a body change.
    measurements_updated = apply_measurement_deltas(db, deltas)
    # Phase 9.2: a muslin fitting is evidence about this body, so it also
    # becomes a learning for later garments of this type. Manual and
    # AI-adjust revisions are deliberately not evidence — see api.learnings.
    # garment_type is None for patterns created before Phase 9 added the
    # column; those simply don't produce learnings rather than guessing.
    learning = (
        record_learning(db, revision, garment_type) if garment_type is not None else None
    )
    db.commit()

    return {
        **_revision_adapter.dump_python(revision, mode="json"),
        "body_profile_updated": measurements_updated,
        "learning_recorded": (
            {"id": learning.id, "times_confirmed": learning.times_confirmed}
            if learning is not None else None
        ),
    }


@app.get("/patterns/{pattern_id}/revisions")
def get_revisions(pattern_id: str, db: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    """Timeline, newest first. A pattern that exists but hasn't been
    revised yet (nothing beyond its Phase 4 version 1) correctly returns
    an empty list, not a 404 — that's a real, distinct state from the
    pattern not existing at all, which still 404s below."""
    exists = db.execute("SELECT 1 FROM patterns WHERE id = ?", (pattern_id,)).fetchone()
    if exists is None:
        raise HTTPException(status_code=404, detail="pattern not found")

    rows = db.execute(
        "SELECT id, pattern_id, version, parent_version, change_type, source, "
        "deltas_json, reason, fit_notes, created_at FROM revisions "
        "WHERE pattern_id = ? ORDER BY version DESC",
        (pattern_id,),
    ).fetchall()
    return [_revision_adapter.dump_python(revision_from_row(row), mode="json") for row in rows]


@app.get("/patterns/{pattern_id}/diff")
def diff_pattern_versions(
    pattern_id: str,
    from_version: int = Query(..., alias="from"),
    to_version: int = Query(..., alias="to"),
    db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """Aggregate deltas directly between the two actual snapshots, not a
    concatenation of each revision's own stored deltas in between — a
    field nudged twice (15mm -> 20mm -> 25mm across two revisions) should
    show up here as the one change that actually matters end to end
    (15mm -> 25mm), not two redundant entries."""
    from_row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, from_version),
    ).fetchone()
    if from_row is None:
        raise HTTPException(status_code=404, detail=f"{pattern_id}: version {from_version} not found")

    to_row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, to_version),
    ).fetchone()
    if to_row is None:
        raise HTTPException(status_code=404, detail=f"{pattern_id}: version {to_version} not found")

    deltas = diff_patterns(pattern_from_json(from_row[0]), pattern_from_json(to_row[0]))
    return {
        "pattern_id": pattern_id, "from": from_version, "to": to_version,
        "deltas": _field_delta_list_adapter.dump_python(deltas, mode="json"),
    }


@pydantic_dataclass
class AskRequest:
    question: str
    top_k: int = 5
    since: str | None = None  # ISO-8601, inclusive
    until: str | None = None  # ISO-8601, inclusive


@app.post("/patterns/{pattern_id}/ask")
def ask_pattern(
    pattern_id: str, body: AskRequest,
    db: sqlite3.Connection = Depends(get_db), client: Anthropic = Depends(get_anthropic_client),
) -> dict:
    """Phase 6's RAG endpoint: retrieve the pattern's own most relevant
    revisions (local cosine search, no LLM call — see api.rag.embed) and
    have Claude synthesize an answer grounded in their actual deltas.
    Returns the revisions used alongside the answer so it can be checked
    against the real data, not just trusted.

    404s if the pattern doesn't exist at all — a 200 with an explanatory
    "no revision history" answer (see api.rag.answer) is reserved for a
    real pattern that just doesn't have revisions matching the query/date
    range, a meaningfully different state from asking about nothing.
    """
    exists = db.execute("SELECT 1 FROM patterns WHERE id = ?", (pattern_id,)).fetchone()
    if exists is None:
        raise HTTPException(status_code=404, detail="pattern not found")

    return answer_question(
        db, client, pattern_id, body.question,
        top_k=body.top_k, since=body.since, until=body.until,
    )


@app.post("/patterns/{pattern_id}/versions/{version}/mark-sloper")
def mark_sloper(
    pattern_id: str, version: int, db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """Mark this exact version as the reference sloper for its own
    garment_type (see schema.py's `reference_slopers` table). A deliberate,
    separate, human-initiated call — PLAN.md Phase 5 is explicit that this
    must never be inferred from revision count or any other heuristic;
    whether a pattern actually fits is a judgment only a muslin can make.

    One sloper per garment_type: marking a new version here replaces
    whichever version held that slot before, it doesn't add another
    candidate. The garment_type itself isn't a separate parameter — it's
    read off the pattern version being marked, since "make this version
    the sloper" and "for which garment type" are the same decision, not
    two independent ones that could disagree.
    """
    row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, version),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="pattern version not found")

    pattern = pattern_from_json(row[0])
    garment_type = pattern.intent.garment_type.value
    marked_at = datetime.now(timezone.utc).isoformat()

    db.execute(
        """
        INSERT INTO reference_slopers (garment_type, pattern_id, version, marked_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(garment_type) DO UPDATE SET
            pattern_id = excluded.pattern_id,
            version = excluded.version,
            marked_at = excluded.marked_at
        """,
        (garment_type, pattern_id, version, marked_at),
    )
    db.commit()

    return {
        "garment_type": garment_type, "pattern_id": pattern_id, "version": version, "marked_at": marked_at,
    }


@app.get("/slopers/{garment_type}")
def get_sloper(garment_type: GarmentType, db: sqlite3.Connection = Depends(get_db)) -> dict:
    """Phase 4's lookup, run before drafting: is there a reference sloper
    for this garment type, and if so, everything `draft_bodice_from_sloper`
    needs to use it (the sloper's own pieces, measurements, and intent —
    not just a pattern_id/version pointer the caller has to go resolve
    itself with a second round trip).

    404, not a null/empty body, when none is marked — PLAN.md's Phase 4
    note says the fallback is "draft from the formulas as today," and a
    plain status check makes that an ordinary try/except at the call site
    instead of a sentinel-value check buried in a 200 response.

    The 500 case (the sloper's pointer outliving the version it points
    at) shouldn't be reachable — nothing in this sketch deletes pattern
    versions — but a clear 500 beats a confusing downstream KeyError the
    day that invariant breaks instead.
    """
    row = db.execute(
        "SELECT pattern_id, version, marked_at FROM reference_slopers WHERE garment_type = ?",
        (garment_type.value,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no reference sloper marked for {garment_type.value}")
    pattern_id, version, marked_at = row

    snap_row = db.execute(
        "SELECT snapshot_json FROM pattern_versions WHERE pattern_id = ? AND version = ?",
        (pattern_id, version),
    ).fetchone()
    if snap_row is None:
        raise HTTPException(
            status_code=500,
            detail=f"reference sloper for {garment_type.value} points at "
                   f"{pattern_id} v{version}, which no longer exists",
        )

    pattern = pattern_from_json(snap_row[0])
    return {
        "garment_type": garment_type.value, "pattern_id": pattern_id, "version": version,
        "marked_at": marked_at, "pattern": pattern.to_dict(),
    }


@app.post("/croquis")
async def post_croquis(file: UploadFile) -> dict[str, Any]:
    """Phase 9: sketch upload -> a stored image. Nothing else.

    This used to make a vision call that proposed a DesignIntent for a
    human to correct. That's been reversed deliberately (PLAN.md Phase 9,
    "Design intent is author-written"): garment type, fabric, fit,
    closures and style notes are decisions a person makes, and a
    pre-filled field is far less likely to be read critically than an
    empty one — the proposal anchors the answer even when it's wrong.

    The sketch is still read by a model, just later and for a different
    job: it goes into the drafting call (api.drafting.ai_adjust) with the
    author's own intent, where it informs *geometry* — how deep the
    neckline really sits, where a seam falls — rather than deciding what
    the garment is. Storing an image needs no API key, so attaching a
    croquis works offline.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    CROQUIS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".png"
    stored_path = CROQUIS_DIR / f"{uuid.uuid4().hex}{suffix}"
    stored_path.write_bytes(image_bytes)
    croquis_ref = str(stored_path.relative_to(CROQUIS_DIR.parent))

    return {"croquis_ref": croquis_ref}


# The local UI, served by the same app that serves the API — one command
# starts the whole tool, and the page is same-origin with the endpoints it
# calls. Mounted LAST on purpose: a mount at "/" matches anything not
# already claimed, so declaring it earlier would shadow every route above.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
