"""Runtime geometry checks (PLAN.md Phase 9.5).

These relationships were only ever test assertions. Phase 9 lets a model
propose adjustments to drafted geometry, so they have to run against real
output at request time too — a model is perfectly capable of returning a
plausible-looking JSON adjustment that produces an armhole thirty times
too deep, which is exactly the Phase 2 failure CLAUDE.md records.

What this is and isn't: these are *sanity* checks, not a fit guarantee.
Passing means the geometry isn't obviously broken. It does not mean the
pattern fits — only a muslin says that, and nothing here changes
CLAUDE.md's rule that a draft is reported as generated, never correct.

Pure functions, no I/O, same as the rest of api/drafting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from api.export.geometry import flatten_segment, piece_outline, polygon_centroid
from api.schema import DesignIntent, Measurements, PatternPiece, Point

# Tuned from the real construction's own behaviour, as recorded in
# tests/test_bodice.py. Each one guards a bug that actually happened.
_ARMHOLE_RATIO_MAX = 1.5        # front vs back armhole length

# Bulge is bounded as a FRACTION OF THE CHORD, not in millimetres. Measured
# across bust 800-1300mm and both rich and minimal measurement sets, the
# real draft holds steady at 0.24-0.37 while the same curve's absolute
# deviation ranges 37-134mm — a 3.7x spread. An absolute bound tuned on one
# body silently rejects correct drafts for every other body.
_ARMHOLE_BULGE_MIN_RATIO = 0.08
_ARMHOLE_BULGE_MAX_RATIO = 0.60
_ARMHOLE_PINCH_RATIO = 1.8      # arc vs its own chord; real draft runs 1.14-1.38

# The armhole must scoop INTO the piece. This is the check that actually
# guards Phase 2's original bug: that version bulged outward by 0.18-0.32x
# of chord, which overlaps the correct draft's own 0.24-0.37x magnitude
# exactly. Size alone cannot tell the two apart — only direction can.
# Measured concavity on real drafts runs +0.12 to +0.34.
_ARMHOLE_MIN_CONCAVITY = 0.02

# Real construction produces 2.6-13.5 degrees. The ceiling catches a dart
# that has become a fan, not normal variation.
_DART_MAX_APEX_ANGLE_DEG = 60.0
_DART_MIN_INTAKE_MM = 0.5

# Measured across bust 800-1300mm on correct drafts: shoulders match
# exactly, side seams within 0.60mm, bust ease exact, waist ease within
# 0.22mm. These leave real headroom while still catching the millimetres
# a broken design change would introduce.
_SEAM_MATCH_TOLERANCE_MM = 2.0
_EASE_TOLERANCE_MM = 2.0


@dataclass
class ValidationResult:
    """Why a draft was accepted or rejected. `failures` are human-readable
    because they end up in a revision's reason, where the point is being
    able to tell later what the model proposed and why it was refused."""
    ok: bool
    failures: list[str]

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
    fraction of its own chord. Positive is concave (correct); negative
    means it bulges outward, which is the Phase 2 bug.

    Measured against the outline's centroid rather than a fixed axis, so
    it works the same for the front and back piece without either needing
    to know which way it faces.
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
    """Front and back armholes comparable in length, scooped inward by a
    sane amount, and not pinched back on themselves. Each of these guards
    a bug that actually happened."""
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
    """A dart is a narrow wedge: two legs of similar length meeting at an
    apex, enclosing a small angle.

    Checked as an angle at the apex rather than "the legs straddle the
    apex in x" — measured against the real draft, a back shoulder dart
    legitimately has both legs on the same side in x, because the shoulder
    seam it sits on runs diagonally. The real construction produces apex
    angles between 2.6 and 13.5 degrees; the ceiling below is deliberately
    far above that, since this is catching a dart that has become a fan,
    not policing normal variation.
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
    """An edge's length *as sewn*, with any dart along it closed up.

    A dart's two legs meet when it's stitched, so the span between them
    contributes nothing to the finished seam — the back shoulder measures
    143.8mm flat on paper and 127.0mm once its dart is closed. Comparing
    flat lengths would make every darted seam look mismatched against its
    undarted partner.

    Dart spans are found by position along the edge, not by id prefix: the
    run of segments between a dart's two legs is excluded wholesale, which
    also drops any intermediate dart point (the raised `_d1`) that would
    otherwise make the span longer than the straight leg-to-leg distance.
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
    """Seams that get sewn to each other must be the same length.

    This is "walking the seams" — the most basic correctness check in
    patternmaking, and the one that catches a design change shortening one
    side of a pair. A neckline widened on the front but not the back
    leaves shoulder seams that cannot be sewn together, and no other check
    here would notice.

    Pairs are read from each edge's `joins_to` rather than hardcoded, so a
    sleeve or skirt block gets checked the same way without touching this
    function. Measured across bust 800-1300mm, correct drafts match
    exactly at the shoulder and within 0.6mm at the side seam.
    """
    by_id = {p.id: p for p in pieces}
    failures: list[str] = []
    seen: set[tuple[str, str]] = set()

    for piece in pieces:
        for edge in piece.edges:
            if not edge.joins_to:
                continue
            target = edge.joins_to
            if "." not in target:
                failures.append(f"{piece.id}.{edge.id}: malformed joins_to {target!r}")
                continue
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


def check_ease(
    pieces: list[PatternPiece], measurements: Measurements, intent: DesignIntent,
) -> list[str]:
    """The finished garment still has the ease it was drafted for.

    Patternmaking's own final step after any alteration: re-measure bust
    and waist to confirm you haven't silently added or lost ease. Dart
    manipulation is the obvious risk — rotating intake to a new position
    should conserve suppression exactly, and this is what proves it did.

    The waist identity is NOT waist + ease_waist. This block draws its
    rectangle at (bust + ease_bust)/2 from bustline to waistline, while
    the dart calculation removes suppression based on RAW bust — so the
    bust ease carries down to the waist, and the finished waist is
    waist + ease_waist + ease_bust. That's a documented property of the
    construction (see api/drafting/bodice.py), not slack in this check:
    asserting the naive identity would reject every correct draft.
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
    if waist_front is not None and waist_back is not None:
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
    """Every structural check, run together. Used both by the tests and by
    the drafting path before an AI-adjusted draft is accepted.

    `measurements`/`intent` are optional because not every caller has
    them: without them the geometry is still checked for shape and for
    mated-seam agreement, and only the ease check — which needs to know
    what the draft was made *for* — is skipped.
    """
    failures = (
        check_points_are_finite(pieces)
        + check_armholes(pieces)
        + check_necklines_are_monotonic(pieces)
        + check_darts(pieces)
        + check_mated_seams(pieces)
    )
    if measurements is not None and intent is not None:
        failures += check_ease(pieces, measurements, intent)
    return ValidationResult(ok=not failures, failures=failures)


def check_adjustment_magnitude(
    before: list[PatternPiece], after: list[PatternPiece], limit_mm: float,
) -> list[str]:
    """No single point may move more than `limit_mm`.

    Deliberately rejects rather than clamps. A model asking to move a
    point 300mm is wrong about something structural, and quietly shrinking
    that to the limit would hide the error while still applying a change
    nobody reasoned about.
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
