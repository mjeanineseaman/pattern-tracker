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

from api.export.geometry import flatten_segment
from api.schema import PatternPiece, Point

# Tuned from the real construction's own behaviour, as recorded in
# tests/test_bodice.py. Each one guards a bug that actually happened.
_ARMHOLE_RATIO_MAX = 1.5        # front vs back armhole length
_ARMHOLE_BULGE_MIN_MM = 5.0     # a straight line is not an armhole; real draft ~37mm
_ARMHOLE_BULGE_MAX_MM = 60.0    # the original bug peaked at 81.7mm
_ARMHOLE_PINCH_RATIO = 1.8      # arc vs its own chord; real draft runs 1.14-1.20

# Real construction produces 2.6-13.5 degrees. The ceiling catches a dart
# that has become a fan, not normal variation.
_DART_MAX_APEX_ANGLE_DEG = 60.0
_DART_MIN_INTAKE_MM = 0.5


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


def check_armholes(pieces: list[PatternPiece]) -> list[str]:
    """Front and back armholes comparable, curved but not wildly, and not
    pinched back on themselves. All three of these caught real bugs."""
    failures: list[str] = []
    lengths: dict[str, float] = {}

    for piece in pieces:
        poly = _edge_polyline(piece, "armhole")
        if len(poly) < 2:
            continue
        lengths[piece.id] = _polyline_length(poly)

        bulge = _max_perpendicular_deviation(poly)
        if not _ARMHOLE_BULGE_MIN_MM < bulge < _ARMHOLE_BULGE_MAX_MM:
            failures.append(
                f"{piece.id}: armhole curve deviates {bulge:.1f}mm from its chord "
                f"(expected between {_ARMHOLE_BULGE_MIN_MM} and {_ARMHOLE_BULGE_MAX_MM}) — "
                f"a straight line or a wild bulge, not an armhole"
            )

        chord = math.dist(poly[0], poly[-1])
        if chord > 0 and lengths[piece.id] > chord * _ARMHOLE_PINCH_RATIO:
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


def validate_pieces(pieces: list[PatternPiece]) -> ValidationResult:
    """Every structural check, run together. Used both by the tests and by
    the drafting path before an AI-adjusted draft is accepted."""
    failures = (
        check_points_are_finite(pieces)
        + check_armholes(pieces)
        + check_necklines_are_monotonic(pieces)
        + check_darts(pieces)
    )
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
