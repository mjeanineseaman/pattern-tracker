"""Runtime geometry checks.

These were test-only assertions until a model was allowed to propose
adjustments to drafted geometry, which it can do in a way that looks
plausible and produces an armhole thirty times too deep.

Sanity checks, not a fit guarantee: passing means the geometry isn't
obviously broken, not that the pattern fits. Pure functions, no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from api.export.geometry import flatten_segment, piece_outline, polygon_centroid
from api.schema import DesignIntent, Measurements, PatternPiece, Point

# Every bound below was measured across bust 800-1300mm on known-good
# drafts, not guessed. The observed range is recorded beside each one so a
# future change can tell a real regression from a re-tuning.

_ARMHOLE_RATIO_MAX = 1.5        # front vs back armhole length

# Bulge is a fraction of the chord, not millimetres: the same correct curve
# deviates 37-134mm across that size range while holding at 0.24-0.37 of
# chord. An absolute bound tuned on one body rejects correct drafts for
# every other one.
_ARMHOLE_BULGE_MIN_RATIO = 0.08
_ARMHOLE_BULGE_MAX_RATIO = 0.60
_ARMHOLE_PINCH_RATIO = 1.8      # arc vs chord; observed 1.14-1.38

# Direction, not size, is what separates a correct armhole from Phase 2's
# original bug: that one bulged outward by 0.18-0.32 of chord, overlapping
# the correct draft's own magnitude exactly. Observed concavity +0.12-0.34.
_ARMHOLE_MIN_CONCAVITY = 0.02

_DART_MAX_APEX_ANGLE_DEG = 60.0   # observed 2.6-13.5; the ceiling catches a fan
_DART_MIN_INTAKE_MM = 0.5

# Observed: shoulders match exactly, side seams within 0.60mm, bust ease
# exact, waist ease within 0.22mm.
_SEAM_MATCH_TOLERANCE_MM = 2.0
_EASE_TOLERANCE_MM = 2.0


@dataclass
class ValidationResult:
    """`failures` are human-readable because they end up in a revision's
    reason, where the point is telling later what was refused and why.

    `notes` records a check that did not run, and why. A check that quietly
    skips itself is worse than one that fails: the ease identity is the
    strongest thing here, and a design that moves the edge it measures
    would otherwise disable it invisibly.
    """
    ok: bool
    failures: list[str]
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _edge(piece: PatternPiece, edge_id: str):
    for e in piece.edges:
        if e.id == edge_id:
            return e
    return None


def _edge_polyline(piece: PatternPiece, edge_id: str) -> list[tuple[float, float]]:
    edge = _edge(piece, edge_id)
    if edge is None:
        return []
    pm = piece.point_map()
    poly: list[tuple[float, float]] = []
    for seg in edge.segments:
        poly.extend(flatten_segment(pm, seg))
    return poly


def _polyline_length(poly: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(poly, poly[1:]))


def _max_perpendicular_deviation(poly: list[tuple[float, float]]) -> float:
    """How far a curve bulges off its own start->end chord."""
    if len(poly) < 3:
        return 0.0
    (x0, y0), (x1, y1) = poly[0], poly[-1]
    dx, dy = x1 - x0, y1 - y0
    chord = math.hypot(dx, dy)
    if chord == 0:
        return 0.0
    return max(abs((x - x0) * dy - (y - y0) * dx) / chord for x, y in poly)


def armhole_concavity(piece: PatternPiece) -> float | None:
    """How far the armhole scoops toward the piece's interior, as a
    fraction of its chord. Positive is concave (correct). Measured against
    the outline centroid rather than a fixed axis, so front and back work
    the same without either knowing which way it faces.
    """
    poly = _edge_polyline(piece, "armhole")
    if len(poly) < 3:
        return None
    chord = math.dist(poly[0], poly[-1])
    if chord == 0:
        return None
    try:
        centroid = polygon_centroid(piece_outline(piece))
    except (KeyError, ZeroDivisionError, IndexError):
        return None

    curve_mid = poly[len(poly) // 2]
    chord_mid = ((poly[0][0] + poly[-1][0]) / 2, (poly[0][1] + poly[-1][1]) / 2)
    return (math.dist(chord_mid, centroid) - math.dist(curve_mid, centroid)) / chord


def check_armholes(pieces: list[PatternPiece]) -> list[str]:
    """Comparable front to back, scooped inward, not pinched back on
    itself. Each guards a bug that actually happened."""
    failures: list[str] = []
    lengths: dict[str, float] = {}

    for piece in pieces:
        poly = _edge_polyline(piece, "armhole")
        if len(poly) < 2:
            continue
        lengths[piece.id] = _polyline_length(poly)
        chord = math.dist(poly[0], poly[-1])
        if chord == 0:
            failures.append(f"{piece.id}: armhole starts and ends at the same place")
            continue

        bulge_ratio = _max_perpendicular_deviation(poly) / chord
        if not _ARMHOLE_BULGE_MIN_RATIO < bulge_ratio < _ARMHOLE_BULGE_MAX_RATIO:
            failures.append(
                f"{piece.id}: armhole deviates {bulge_ratio:.2f}x its own chord "
                f"(expected {_ARMHOLE_BULGE_MIN_RATIO}-{_ARMHOLE_BULGE_MAX_RATIO}) — "
                f"a straight line or a wild bulge, not an armhole"
            )

        concavity = armhole_concavity(piece)
        if concavity is not None and concavity < _ARMHOLE_MIN_CONCAVITY:
            failures.append(
                f"{piece.id}: armhole curves outward ({concavity:+.2f}) instead of scooping "
                f"into the piece — this is the shape Phase 2 originally got backwards"
            )

        if lengths[piece.id] > chord * _ARMHOLE_PINCH_RATIO:
            failures.append(
                f"{piece.id}: armhole arc is {lengths[piece.id] / chord:.2f}x its own chord "
                f"(limit {_ARMHOLE_PINCH_RATIO}) — the curve is pinched or self-crossing"
            )

    if len(lengths) == 2:
        longer, shorter = max(lengths.values()), min(lengths.values())
        if shorter > 0 and longer / shorter > _ARMHOLE_RATIO_MAX:
            failures.append(
                f"front and back armholes are {longer / shorter:.2f}x apart in length "
                f"(limit {_ARMHOLE_RATIO_MAX}) — one of them is badly mis-drafted"
            )
    return failures


def check_necklines_are_monotonic(pieces: list[PatternPiece]) -> list[str]:
    """A neckline that backtracks in x has doubled over on itself. Checked
    on the flattened curve, since a bezier can backtrack while its
    endpoints look fine."""
    failures = []
    for piece in pieces:
        poly = _edge_polyline(piece, "neckline")
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        forward = xs[-1] >= xs[0]
        for a, b in zip(xs, xs[1:]):
            if (forward and b < a - 0.5) or (not forward and b > a + 0.5):
                failures.append(
                    f"{piece.id}: neckline reverses direction along x — the curve doubles back"
                )
                break
    return failures


def check_darts(pieces: list[PatternPiece]) -> list[str]:
    """A dart is a narrow wedge: two legs meeting at an apex across a
    small angle.

    Checked as an apex angle, not "the legs straddle the apex in x" — a
    back shoulder dart legitimately has both legs on one side, because the
    shoulder seam it sits on runs diagonally.
    """
    failures = []
    for piece in pieces:
        pm = piece.point_map()
        for dart in piece.darts:
            try:
                apex, leg_a, leg_b = pm[dart.apex], pm[dart.leg_a], pm[dart.leg_b]
            except KeyError:
                failures.append(f"{piece.id}: dart {dart.id} references a point that doesn't exist")
                continue

            if math.dist((leg_a.x, leg_a.y), (leg_b.x, leg_b.y)) < _DART_MIN_INTAKE_MM:
                failures.append(f"{piece.id}: dart {dart.id} has zero intake — its legs coincide")
                continue

            va = (leg_a.x - apex.x, leg_a.y - apex.y)
            vb = (leg_b.x - apex.x, leg_b.y - apex.y)
            len_a, len_b = math.hypot(*va), math.hypot(*vb)
            if len_a < _DART_MIN_INTAKE_MM or len_b < _DART_MIN_INTAKE_MM:
                failures.append(
                    f"{piece.id}: dart {dart.id} has a leg of near-zero length — "
                    f"its apex sits on top of a leg point"
                )
                continue

            cos = (va[0] * vb[0] + va[1] * vb[1]) / (len_a * len_b)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if angle > _DART_MAX_APEX_ANGLE_DEG:
                failures.append(
                    f"{piece.id}: dart {dart.id} opens {angle:.0f}° at its apex "
                    f"(limit {_DART_MAX_APEX_ANGLE_DEG}°) — that's a fan, not a dart"
                )
    return failures


def sewn_edge_length(piece: PatternPiece, edge_id: str) -> Optional[float]:
    """An edge's length as sewn, with any dart along it closed.

    A dart's legs meet when stitched, so the span between them adds nothing
    to the finished seam — the back shoulder is 143.8mm flat and 127.0mm
    sewn. Comparing flat lengths would make every darted seam look
    mismatched against an undarted partner.

    Dart spans are found by position along the edge, which also drops the
    raised intermediate dart point.
    """
    edge = _edge(piece, edge_id)
    if edge is None or not edge.segments:
        return None
    pm = piece.point_map()
    points = [edge.segments[0].start] + [s.end for s in edge.segments]

    skip: set[int] = set()
    for dart in piece.darts:
        if dart.leg_a in points and dart.leg_b in points:
            i, j = sorted((points.index(dart.leg_a), points.index(dart.leg_b)))
            skip |= set(range(i, j))

    total = 0.0
    for k in range(len(points) - 1):
        if k in skip:
            continue
        try:
            a, b = pm[points[k]], pm[points[k + 1]]
        except KeyError:
            return None
        total += math.dist((a.x, a.y), (b.x, b.y))
    return total


def check_mated_seams(pieces: list[PatternPiece]) -> list[str]:
    """Seams sewn to each other must be the same length — "walking the
    seams".

    Catches a design change shortening one side of a pair: a neckline
    widened on the front but not the back leaves shoulders that can't be
    joined, and no other check notices. Pairs come from each edge's
    `joins_to`, so a sleeve or skirt block is checked by the same code.
    """
    by_id = {p.id: p for p in pieces}
    failures: list[str] = []
    seen: set[tuple[str, str]] = set()

    for piece in pieces:
        for edge in piece.edges:
            if not edge.joins_to:
                continue
            target = edge.joins_to
            other_piece_id, other_edge_id = target.split(".", 1)

            key = tuple(sorted([f"{piece.id}.{edge.id}", target]))
            if key in seen:
                continue  # a pair declares itself from both sides; check it once
            seen.add(key)

            other = by_id.get(other_piece_id)
            if other is None:
                continue  # the partner piece isn't in this draft; not this check's business
            mine, theirs = sewn_edge_length(piece, edge.id), sewn_edge_length(other, other_edge_id)
            if mine is None or theirs is None:
                failures.append(
                    f"{piece.id}.{edge.id} joins {target}, which could not be measured"
                )
                continue
            if abs(mine - theirs) > _SEAM_MATCH_TOLERANCE_MM:
                failures.append(
                    f"{piece.id}.{edge.id} ({mine:.1f}mm) and {target} ({theirs:.1f}mm) "
                    f"are sewn together but differ by {abs(mine - theirs):.1f}mm "
                    f"(tolerance {_SEAM_MATCH_TOLERANCE_MM}mm)"
                )
    return failures


def hem_lift_mm(
    pieces: list[PatternPiece], measurements: Measurements,
) -> Optional[float]:
    """How far the bottom edge sits above the waistline, or None if it's on it.

    The waist ease identity assumes the bottom edge still *is* the
    waistline. A cropped hem (`design.apply_hem_change`) sits above it,
    over the ribs, where a fitted block is wider — so the identity would
    measure rib girth, compare it against the waist measurement, and reject
    a correct cropped top for being cropped.

    The geometry answers this without being told: `cb_waist` is drafted
    exactly `back_length` below the nape at y=0, so a lifted hem shows up
    as a mismatch there. No body measurement is taken at rib level, so
    there is nothing to check against instead and the identity is skipped
    — reported in `ValidationResult.notes`, never silently.
    """
    back = next((p for p in pieces if p.id == "bodice_back"), None)
    if back is None or measurements.back_length is None:
        return None
    cb_waist = back.point_map().get("cb_waist")
    if cb_waist is None:
        return None
    lift = measurements.back_length - cb_waist.y
    return lift if lift > _EASE_TOLERANCE_MM else None


def check_ease(
    pieces: list[PatternPiece], measurements: Measurements, intent: DesignIntent,
) -> list[str]:
    """The finished garment still has the ease it was drafted for.

    Patternmaking's final step after any alteration, and what proves dart
    rotation conserved suppression.

    The waist identity is NOT waist + ease_waist. This block draws its
    rectangle at (bust + ease_bust)/2 down to the waistline while
    suppression comes off RAW bust, so bust ease carries down and the
    finished waist is waist + ease_waist + ease_bust. Asserting the naive
    identity would reject every correct draft.
    """
    by_id = {p.id: p for p in pieces}
    front, back = by_id.get("bodice_front"), by_id.get("bodice_back")
    if front is None or back is None:
        return []  # not a bodice pair; nothing this check knows how to measure

    failures: list[str] = []
    fm, bm = front.point_map(), back.point_map()

    # Bust: each piece is a half, measured from its fold (x=0) out to the
    # underarm, which sits on the bustline by construction.
    try:
        bust_finished = 2 * (fm["underarm_front"].x + bm["underarm_back"].x)
    except KeyError:
        bust_finished = None
    if bust_finished is not None:
        bust_expected = measurements.bust + intent.ease_bust_mm
        if abs(bust_finished - bust_expected) > _EASE_TOLERANCE_MM:
            failures.append(
                f"finished bust {bust_finished:.1f}mm but the draft was made for "
                f"{bust_expected:.1f}mm (bust {measurements.bust:.1f} + ease "
                f"{intent.ease_bust_mm:.1f}) — off by {bust_finished - bust_expected:+.1f}mm"
            )

    waist_front, waist_back = sewn_edge_length(front, "waist"), sewn_edge_length(back, "waist")
    # A cropped hem sits above the waistline, not on it; see `hem_lift_mm`.
    skip_waist = hem_lift_mm(pieces, measurements) is not None
    if not skip_waist and waist_front is not None and waist_back is not None:
        waist_finished = 2 * (waist_front + waist_back)
        waist_expected = measurements.waist + intent.ease_waist_mm + intent.ease_bust_mm
        if abs(waist_finished - waist_expected) > _EASE_TOLERANCE_MM:
            failures.append(
                f"finished waist {waist_finished:.1f}mm but the draft was made for "
                f"{waist_expected:.1f}mm (waist {measurements.waist:.1f} + waist ease "
                f"{intent.ease_waist_mm:.1f} + bust ease {intent.ease_bust_mm:.1f}) — "
                f"off by {waist_finished - waist_expected:+.1f}mm"
            )
    return failures


def check_points_are_finite(pieces: list[PatternPiece]) -> list[str]:
    """NaN and inf propagate silently through geometry and only surface as
    an unrenderable PDF much later. A model returning "null" or a wild
    float for a coordinate lands here."""
    failures = []
    for piece in pieces:
        for p in piece.points:
            if not (math.isfinite(p.x) and math.isfinite(p.y)):
                failures.append(f"{piece.id}: point {p.id} has a non-finite coordinate")
    return failures


def validate_pieces(
    pieces: list[PatternPiece],
    measurements: Optional[Measurements] = None,
    intent: Optional[DesignIntent] = None,
) -> ValidationResult:
    """Every check, run together.

    `measurements`/`intent` are optional: without them everything but the
    ease check still runs, since only that one needs to know what the
    draft was made for.
    """
    failures = (
        check_points_are_finite(pieces)
        + check_armholes(pieces)
        + check_necklines_are_monotonic(pieces)
        + check_darts(pieces)
        + check_mated_seams(pieces)
    )
    notes: list[str] = []
    if measurements is not None and intent is not None:
        failures += check_ease(pieces, measurements, intent)
        lift = hem_lift_mm(pieces, measurements)
        if lift is not None:
            notes.append(
                f"waist ease not checked: the hem is {lift:.0f}mm above the "
                f"waist, and no body measurement is taken at that level"
            )
    return ValidationResult(ok=not failures, failures=failures, notes=notes)


def check_adjustment_magnitude(
    before: list[PatternPiece], after: list[PatternPiece], limit_mm: float,
) -> list[str]:
    """No single point may move more than `limit_mm`.

    Rejects rather than clamps: a model asking for 300mm is wrong about
    something structural, and shrinking it would hide that while still
    applying a change nobody reasoned about.
    """
    failures = []
    before_points: dict[tuple[str, str], Point] = {
        (piece.id, p.id): p for piece in before for p in piece.points
    }
    for piece in after:
        for p in piece.points:
            old = before_points.get((piece.id, p.id))
            if old is None:
                continue
            moved = math.dist((old.x, old.y), (p.x, p.y))
            if moved > limit_mm:
                failures.append(
                    f"{piece.id}.{p.id} moved {moved:.1f}mm (limit {limit_mm:.0f}mm)"
                )
    return failures
