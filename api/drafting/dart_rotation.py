"""Dart rotation — relocating the front bust dart without changing how
much suppression there is (PLAN.md Phase 10, Tier 2).

A dart's intake is an *angle* subtended at the point it points toward
(`bust_point`), not a fixed mm width. Patternmaking exploits that: the
whole dart can be swung to a new position on the outline, as long as you
rotate the exact fabric arc it's attached to by that same angle — closing
the dart where it was and reopening an equal one wherever you cut. The mm
width changes with the radius from `bust_point`; the angle, and so the
amount of fabric actually removed from the garment, does not.

Centre front is fixed — on the fold, by CLAUDE.md's own invariant — so the
arc that rotates is always the one on the *other* side of the dart: the
side seam, armhole, and anything beyond, rigidly, as one body (every edge
in between is a fixed-length line or curve). That constrains which
positions this can reach without side effects. `"side"` and `"french"`
both cut into the side seam below `underarm_front`, so only the dart's own
two points and `side_waist_front` ever move — `underarm_front`, the
armhole, and the shoulder never do, which is what keeps bust and waist
circumference unchanged (checked below, not assumed).

`"shoulder"` is not implemented. Reaching it means rotating the arc all
the way up through the armhole to the existing shoulder dart's far leg,
which drags `underarm_front` with it — measured at ~13mm on a 34in bust,
well past any reasonable tolerance, because every point between the two
darts is rigidly connected. There's no local version of that move on this
block's current topology; it would need either a different pivot
convention (letting centre front move slightly off the fold) or a
different bust check (one that measures a true sewn cross-section instead
of reading `underarm_front.x` directly). Flagged for a person, like
bodice.py's own dart-apex-standoff question, not decided here.

Front piece only — the back's own darts aren't touched.
"""
from __future__ import annotations

import copy
import math

from api.drafting.bodice import DART_APEX_STANDOFF_CM, dart_point_out, unit_vector
from api.drafting.design import drop_dart
from api.schema import Dart, PatternPiece, Point, Segment, SegmentKind

_PIECE_ID = "bodice_front"
_STANDOFF_MM = DART_APEX_STANDOFF_CM * 10.0

# The model's fixed set (PLAN.md): it names a position, never a
# coordinate. "shoulder" is a real member of that set but not yet
# reachable here — see the module docstring — so it's listed separately
# rather than silently folded into an "unknown position" error.
POSITIONS: tuple[str, ...] = ("waist", "side", "french")
_NOT_YET_IMPLEMENTED = ("shoulder",)

# side_dart_fraction/french_dart_fraction place the new dart by interpolating
# between underarm_front and side_waist_front -- a linear interpolation,
# which only lands ON that segment (not its extension) when bust_point
# actually sits between the two in height. On a body whose armscye-depth
# estimate is large relative to its own measured bust_height, bust_point
# can end up ABOVE underarm_front instead, and the fraction goes negative
# -- extrapolating past underarm_front rather than landing on the seam.
# The side-seam-length-is-conserved proof in the module docstring depends
# on T sitting between the two endpoints; outside that, it silently
# doesn't hold (found by testing the body-size range validate.py's own
# tolerances were calibrated against, not a realistic single body -- see
# tests/test_dart_rotation.py). A small margin, not a bare 0/1, keeps the
# new dart's near leg off of underarm_front itself.
_MIN_SEAM_MARGIN = 0.05


def _rotate(x: float, y: float, pivot: Point, angle: float) -> tuple[float, float]:
    dx, dy = x - pivot.x, y - pivot.y
    c, s = math.cos(angle), math.sin(angle)
    return (pivot.x + dx * c - dy * s, pivot.y + dx * s + dy * c)


def _signed_angle(pivot: Point, frm: Point, to: Point) -> float:
    """Radians to rotate the direction pivot->frm onto pivot->to."""
    v1 = (frm.x - pivot.x, frm.y - pivot.y)
    v2 = (to.x - pivot.x, to.y - pivot.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(cross, dot)


def side_dart_fraction(front: PatternPiece) -> float:
    """Where the side seam passes `bust_point`'s own height — the
    conventional entry for a dart pointed straight at the bust from the
    side. Derived from this draft's own geometry rather than a fixed
    fraction, so a longer or shorter torso moves the entry point with it
    instead of landing above the armhole or below the waist on a
    different body."""
    pm = front.point_map()
    underarm, side_waist = pm["underarm_front"], pm["side_waist_front"]
    bust_point = pm["bust_point"]
    return (bust_point.y - underarm.y) / (side_waist.y - underarm.y)


def french_dart_fraction(front: PatternPiece) -> float:
    """Flagged drafting-formula choice, same spirit as bodice.py's own
    dart-apex-standoff: halfway between the side dart's entry and the
    waist, which gives a longer, more diagonal dart without a second
    independent constant. There's no single textbook height for this;
    worth a second look once it's actually been seen on paper."""
    return (side_dart_fraction(front) + 1.0) / 2.0


def _target_on_side_seam(front: PatternPiece, t: float) -> tuple[float, float]:
    pm = front.point_map()
    underarm, side_waist = pm["underarm_front"], pm["side_waist_front"]
    return (
        underarm.x + t * (side_waist.x - underarm.x),
        underarm.y + t * (side_waist.y - underarm.y),
    )


def rotate_bust_dart_to(pieces: list[PatternPiece], position: str) -> list[PatternPiece]:
    """Relocate the front bust dart (currently at the waist) to a new
    named position, conserving its intake angle around `bust_point`.

    `"waist"` is the block's own position and a no-op. `"side"` and
    `"french"` cut a new dart into the side seam, each at its own height
    (see `side_dart_fraction`/`french_dart_fraction`), close the old one,
    and rotate only the small arc strictly between them — `side_waist_front`
    and the two new dart points — never `underarm_front` or anything
    beyond it. Raises on `"shoulder"` (not yet implemented, see the module
    docstring) and on any other name.

    Doesn't validate its own output — same division of labour as
    api.drafting.design's individual `apply_*` functions: this is the
    geometry; a caller runs `validate_pieces()` over the result.
    """
    if position == "waist":
        return copy.deepcopy(pieces)
    if position in _NOT_YET_IMPLEMENTED:
        raise ValueError(
            "shoulder dart position isn't implemented yet: merging the bust "
            "dart into the existing shoulder dart rotates underarm_front by "
            "roughly 13mm on a 34in bust, well outside tolerance, because "
            "every point between the two darts (the side seam, armhole, and "
            "shoulder tip) is rigidly connected and has to rotate with it. "
            "See PLAN.md Tier 2."
        )
    if position not in POSITIONS:
        raise ValueError(f"unknown bust_dart_position {position!r}, expected one of {POSITIONS}")

    pieces = copy.deepcopy(pieces)
    front = next(p for p in pieces if p.id == _PIECE_ID)
    pm = front.point_map()
    bust_point = pm["bust_point"]

    old_dart = next((d for d in front.darts if d.id == "bust_dart"), None)
    if old_dart is None:
        raise ValueError(
            "no bust_dart on bodice_front to rotate -- it's already been "
            "dropped (fully released by waist_suppression, or cropped past "
            "its apex)"
        )
    leg_a, leg_b = pm[old_dart.leg_a], pm[old_dart.leg_b]

    # The one angle that does both jobs at once: rotating leg_b onto
    # leg_a's own angular position around bust_point closes the old dart
    # (that's what "onto leg_a" means) and, applied to whatever's cut at
    # the new location below, reopens an equal gap there. Same intake,
    # different address -- see the module docstring.
    theta = _signed_angle(bust_point, leg_b, leg_a)

    t = side_dart_fraction(front) if position == "side" else french_dart_fraction(front)
    if not _MIN_SEAM_MARGIN < t < 1.0 - _MIN_SEAM_MARGIN:
        raise ValueError(
            f"a {position} dart can't be placed on this body: bust_point sits "
            f"outside the underarm-to-side_waist run on the side seam "
            f"(fraction {t:.2f}, needs to be well within 0..1) -- the "
            "underlying measurements are likely inconsistent (e.g. a large "
            "armscye_depth relative to a small bust_height)"
        )
    target_x, target_y = _target_on_side_seam(front, t)

    side_waist = pm["side_waist_front"]
    rotated_side_waist = _rotate(side_waist.x, side_waist.y, bust_point, theta)
    rotated_far = _rotate(target_x, target_y, bust_point, theta)
    side_waist.x, side_waist.y = rotated_side_waist

    prefix = f"front_{position}_dart"
    ux, uy = unit_vector(target_x - bust_point.x, target_y - bust_point.y)
    apex = Point(f"{prefix}_apex", bust_point.x + ux * _STANDOFF_MM, bust_point.y + uy * _STANDOFF_MM)
    near_leg = Point(f"{prefix}_leg_a", target_x, target_y)       # underarm side: fixed
    far_leg = Point(f"{prefix}_leg_b", *rotated_far)              # side_waist side: rotated
    d1 = Point(f"{prefix}_d1", *dart_point_out(
        (near_leg.x, near_leg.y), (far_leg.x, far_leg.y), (apex.x, apex.y),
    ))

    drop_dart(front, old_dart)   # closes the old dart, heals the waist edge

    side_seam = next(e for e in front.edges if e.id == "side_seam")
    side_seam.segments = [
        Segment("underarm_front", near_leg.id, SegmentKind.LINE),
        Segment(near_leg.id, d1.id, SegmentKind.LINE),
        Segment(d1.id, far_leg.id, SegmentKind.LINE),
        Segment(far_leg.id, "side_waist_front", SegmentKind.LINE),
    ]
    front.points += [apex, near_leg, far_leg, d1]
    front.darts.append(Dart(
        id=f"{position}_dart", apex=apex.id, leg_a=near_leg.id, leg_b=far_leg.id,
    ))

    return pieces
