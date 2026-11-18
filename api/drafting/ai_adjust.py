"""The AI drafting step — the only place a model touches geometry.

The model proposes bounded `{piece, point, dx, dy, reason}` deltas against
an already-valid draft and never emits coordinates. A delta can be
range-checked; a regenerated coordinate set can only be re-validated
wholesale, which is how Phase 2's armhole passed every check while being
thirty times too deep. Deltas also reuse the FieldDelta shape a manual
edit produces, so an AI adjustment needs no new machinery to appear in the
timeline, the diff view or the RAG index.

Everything fails toward the formula draft: no key, no network, bad JSON, a
failed validation. Drafting must not become unavailable because an API
call did.

`draft_bodice()` stays pure; this is the orchestration layer around it.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from api.drafting.validate import check_adjustment_magnitude, validate_pieces
from api.schema import DesignIntent, Measurements, PatternPiece

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"

# PLAN.md 9.5: no single proposed adjustment may move a point further than
# this. Rejected rather than clamped — a model asking for a 300mm move is
# wrong about something structural, and silently shrinking it would hide
# the error while still applying a change nobody reasoned about.
MAX_ADJUSTMENT_MM = 15.0

_MAX_ADJUSTMENTS = 40
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$")

_SYSTEM_PROMPT = """\
You adjust a sewing pattern that has already been drafted by deterministic \
pattern-drafting formulas. The draft you are given is structurally valid.

Your job is NOT to redraft it. It is to apply what previous muslin fittings \
taught us about this specific body, which the formulas cannot know.

Rules:
- Respond with ONLY a JSON array, no prose and no markdown fences.
- Each element is {"piece": str, "point": str, "dx": float, "dy": float, \
"reason": str}. dx/dy are millimetres; +x is right, +y is down.
- Only use piece ids and point ids that appear in the draft below. An \
unrecognised id is discarded.
- No single adjustment may move a point more than %(limit).0fmm. A larger \
move will be rejected and the whole proposal discarded.
- Adjust only what a listed learning actually supports. If the learnings \
don't justify changing anything, return an empty array. An empty array is \
a good answer, not a failure.
- Never invent measurements. Numbers come from the body measurements and \
the deltas you are given, never from prose.
- Each "reason" must name which learning it is acting on.
""" % {"limit": MAX_ADJUSTMENT_MM}


@dataclass
class Adjustment:
    piece: str
    point: str
    dx: float
    dy: float
    reason: str


@dataclass
class AdjustmentOutcome:
    """Detailed enough to put in a revision's reason, so the model's
    contribution is visible rather than baked into the geometry."""
    pieces: list[PatternPiece]
    applied: list[Adjustment]
    rejected: list[str]
    note: str

    @property
    def used_ai(self) -> bool:
        return bool(self.applied)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return _FENCE_RE.sub("", stripped).strip()


def describe_pieces(pieces: Sequence[PatternPiece]) -> str:
    """Point ids and coordinates, plus which darts reference which points
    so a dart can be moved coherently."""
    lines = []
    for piece in pieces:
        lines.append(f'piece "{piece.id}" ({piece.name}):')
        for p in piece.points:
            lines.append(f"  {p.id}: x={p.x:.1f} y={p.y:.1f}")
        for dart in piece.darts:
            lines.append(
                f"  dart {dart.id}: apex={dart.apex} legs={dart.leg_a},{dart.leg_b}"
            )
    return "\n".join(lines)


def build_prompt(
    pieces: Sequence[PatternPiece],
    measurements: Measurements,
    intent: DesignIntent,
    learnings: Sequence[str],
) -> str:
    measurement_lines = "\n".join(
        f"  {f.name}: {getattr(measurements, f.name)}"
        for f in dataclasses.fields(Measurements)
        if getattr(measurements, f.name) is not None
    )
    if learnings:
        learning_lines = "\n".join(f"  - {text}" for text in learnings)
    else:
        learning_lines = "  (none recorded yet for this garment type)"

    return (
        f"Body measurements (mm):\n{measurement_lines}\n\n"
        f"Design intent:\n"
        f"  garment_type: {intent.garment_type.value}\n"
        f"  fabric_class: {intent.fabric_class.value}\n"
        f"  fit: {intent.fit}\n"
        f"  closures: {', '.join(intent.closures) or '(none)'}\n"
        f"  style_notes: {intent.style_notes or '(none)'}\n\n"
        f"What previous fittings taught us about this body:\n{learning_lines}\n\n"
        f"The current draft:\n{describe_pieces(pieces)}\n\n"
        f"Return the JSON array of adjustments."
    )


def parse_adjustments(raw_text: str) -> list[Adjustment]:
    """Raises ValueError on anything malformed; the caller turns that into
    a fallback rather than retrying blindly."""
    parsed = json.loads(_strip_fences(raw_text))
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")
    if len(parsed) > _MAX_ADJUSTMENTS:
        raise ValueError(f"{len(parsed)} adjustments proposed, limit is {_MAX_ADJUSTMENTS}")

    adjustments = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("adjustment entries must be objects")
        try:
            dx, dy = float(item["dx"]), float(item["dy"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"adjustment missing a usable dx/dy: {e}") from e
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise ValueError("adjustment dx/dy must be finite")
        if "piece" not in item or "point" not in item:
            raise ValueError("adjustment missing piece/point")
        adjustments.append(Adjustment(
            piece=str(item["piece"]), point=str(item["point"]),
            dx=dx, dy=dy, reason=str(item.get("reason", "")),
        ))
    return adjustments


def apply_adjustments(
    pieces: Sequence[PatternPiece], adjustments: Sequence[Adjustment],
) -> tuple[list[PatternPiece], list[Adjustment], list[str]]:
    """Returns the new pieces, the adjustments applied, and complaints
    about the ones that weren't.

    An unrecognised id is dropped individually rather than failing the
    batch — one bad point shouldn't discard four good adjustments.
    Structural breakage is caught afterwards by the validator, which does
    reject wholesale.
    """
    by_point: dict[tuple[str, str], Adjustment] = {}
    rejected: list[str] = []
    known = {(piece.id, p.id) for piece in pieces for p in piece.points}

    for adj in adjustments:
        key = (adj.piece, adj.point)
        if key not in known:
            rejected.append(f"unknown point {adj.piece}.{adj.point}")
            continue
        if key in by_point:
            rejected.append(f"duplicate adjustment for {adj.piece}.{adj.point}")
            continue
        by_point[key] = adj

    adjusted = []
    for piece in pieces:
        points = []
        for p in piece.points:
            adj = by_point.get((piece.id, p.id))
            points.append(dataclasses.replace(p, x=p.x + adj.dx, y=p.y + adj.dy) if adj else p)
        adjusted.append(dataclasses.replace(piece, points=points))

    return adjusted, list(by_point.values()), rejected


def image_block(croquis_path: Optional[Path]) -> Optional[dict[str, Any]]:
    """The sketch, if it's still on disk. A missing file is not an error:
    the pattern outlives the upload.

    Public because api.drafting.design_from_croquis builds the same block
    for its own model call — the design step's whole job is reading this
    sketch, so it reuses the one place that already turns a path into
    something the API accepts rather than re-deriving media-type handling.
    """
    if croquis_path is None or not croquis_path.is_file():
        return None
    suffix = croquis_path.suffix.lower()
    media_type = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }.get(suffix)
    if media_type is None:
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(croquis_path.read_bytes()).decode("ascii"),
        },
    }


def adjust_draft(
    client,
    pieces: list[PatternPiece],
    measurements: Measurements,
    intent: DesignIntent,
    learnings: Sequence[str],
    croquis_path: Optional[Path] = None,
) -> AdjustmentOutcome:
    """Apply this body's learnings to a formula draft.

    Always returns usable pieces: every failure path returns the original
    draft with a note saying why, which ends up in the revision reason.
    """
    def fallback(note: str, rejected: Optional[list[str]] = None) -> AdjustmentOutcome:
        return AdjustmentOutcome(
            pieces=list(pieces), applied=[], rejected=rejected or [], note=note,
        )

    if not learnings:
        return fallback(
            "No fitting history for this garment type yet, so the draft is the "
            "formula draft unchanged."
        )

    content: list[dict[str, Any]] = []
    image = image_block(croquis_path)
    if image is not None:
        content.append(image)
    content.append({
        "type": "text",
        "text": build_prompt(pieces, measurements, intent, learnings),
    })

    try:
        response = client.messages.create(
            model=MODEL, max_tokens=2048, system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = response.content[0].text if response.content else ""
    except Exception as e:  # network, auth, rate limit — all the same answer here
        logger.warning("drafting call failed, using the formula draft: %s", e)
        return fallback(f"Drafted from the formulas — the adjustment step failed ({e}).")

    try:
        proposed = parse_adjustments(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        return fallback(f"Drafted from the formulas — the adjustment proposal was unusable ({e}).")

    if not proposed:
        return fallback("The fitting history didn't call for any change to this draft.")

    adjusted, applied, rejected = apply_adjustments(pieces, proposed)
    if not applied:
        return fallback("Drafted from the formulas — no proposed adjustment named a real point.", rejected)

    oversized = check_adjustment_magnitude(pieces, adjusted, MAX_ADJUSTMENT_MM)
    if oversized:
        return fallback(
            "Drafted from the formulas — an adjustment exceeded the "
            f"{MAX_ADJUSTMENT_MM:.0f}mm limit: {'; '.join(oversized)}",
            rejected,
        )

    # measurements/intent included so the ease check runs: an adjustment
    # that quietly changes finished bust or waist has altered the fit the
    # draft was made for, which no purely structural check would catch.
    validation = validate_pieces(adjusted, measurements, intent)
    if not validation.ok:
        # Wholesale rejection, never partial: a set of adjustments is one
        # argument about the geometry, and half of it is not a weaker
        # version of it.
        return fallback(
            "Drafted from the formulas — the adjusted geometry failed validation: "
            f"{'; '.join(validation.failures)}",
            rejected,
        )

    summary = "; ".join(f"{a.piece}.{a.point} {a.reason}".strip() for a in applied)
    return AdjustmentOutcome(
        pieces=adjusted, applied=applied, rejected=rejected,
        note=f"Adjusted from fitting history ({len(applied)} change(s)): {summary}",
    )
