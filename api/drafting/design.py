"""Parametric design changes to a drafted block (PLAN.md Phase 10, Tier 1).

`bodice.py` produces one block from a body. This module turns that block
into a *design* — a scoop neck, a cropped boxy top — by moving named points
a measured amount. It is the step that was missing: without it the croquis
could not reach the geometry, and two different sketches produced
byte-identical patterns.

Pure throughout, same rule as `bodice.py`: no I/O, no model call, no
database. A model chooses the numbers; this module is the only thing that
touches geometry, and it can be read and tested without one.

Everything is in millimetres (CLAUDE.md). y grows *downward*: `cb_neck`
sits at y=0 and `cb_waist` at y≈356, so dropping a neckline adds to y and
raising an armhole subtracts from it.

**The one idea that makes this tractable.** Every function here goes
through `translate_points`, which moves points *and carries each cubic's
control point along with the endpoint it belongs to* — c1 with the
segment's start, c2 with its end. That single rule is what preserves the
fold-line tangent invariant (CLAUDE.md: a curve meeting the fold with
anything but a horizontal tangent mirrors into a cusp that is invisible on
the half-piece). The front neckline's c2 sits at the same y as `cf_neck`;
move both by the same dy and it still does. Hand-deriving new control
points per parameter would have had to rediscover that each time.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

from api.drafting.bodice import dart_point_out
from api.schema import DesignIntent, PatternPiece, SegmentKind

# A dart needs to stay a dart. Cropping that lands the hem at or above a
# waist dart's apex turns the wedge inside out, so there is a body-dependent
# limit well short of the parameter range below; `apply_hem_change` raises
# rather than emitting a piece the validator would have to catch.
_MIN_DART_LENGTH_MM = 20.0

# Below this much intake there is nothing to fold. A dart released to less
# than it is not a shallower dart, it is a crease, and the right answer is
# to drop it — see `_drop_dart`.
_MIN_DART_INTAKE_MM = 2.0

# Widening a neckline slides the neck point toward the next point along the
# shoulder; it must not reach it, or the seam reverses on itself. Smaller
# than the dart margin because the segment it travels is short to begin
# with — on a 34in bust it is 71mm.
_MIN_SHOULDER_RUN_MM = 5.0

# Point ids this module has to know by name. The block's ids are permanent
# (CLAUDE.md), so naming them is safe; deriving them from edge roles instead
# would be indirection for its own sake.
_FOLD_NECK = {"bodice_front": "cf_neck", "bodice_back": "cb_neck"}
_NECK_SIDE = {"bodice_front": "front_neck_side", "bodice_back": "back_neck_side"}
_UNDERARM = {"bodice_front": "underarm_front", "bodice_back": "underarm_back"}
_UNDERARM_WAIST = {
    "bodice_front": "underarm_waist_front", "bodice_back": "underarm_waist_back",
}
_FOLD_WAIST = {"bodice_front": "cf_waist", "bodice_back": "cb_waist"}
_SIDE_WAIST = {"bodice_front": "side_waist_front", "bodice_back": "side_waist_back"}
_SHOULDER_TIP = {"bodice_front": "front_shoulder_tip", "bodice_back": "back_shoulder_tip"}

# Where the shoulder seam runs on each piece, as (from, toward). Widening a
# neckline slides the neck point along this line. The front's neck point is
# a dart leg, so the line it travels is the *shoulder* proper — the dart's
# far leg to the shoulder tip — not the horizontal gap across the dart.
_SHOULDER_LINE = {
    "bodice_front": ("front_shoulder_dart_leg_t", "front_shoulder_tip"),
    "bodice_back": ("back_neck_side", "back_shoulder_dart_leg_p"),
}

# Which points move together when the neckline widens, per piece. On the
# back that is just the neck point. On the front it is *both* legs of the
# shoulder dart: `front_neck_side` is the near leg, and the far leg
# `front_shoulder_dart_leg_t` is what the shoulder seam actually starts
# from once the dart is sewn shut. Moving both by the same vector keeps the
# dart's intake exact and shortens the sewn shoulder by the same amount the
# back's shortens by — which is the whole reason width is one parameter.
_WIDEN_WITH_NECK_SIDE = {
    "bodice_front": ("front_neck_side", "front_shoulder_dart_leg_t"),
    "bodice_back": ("back_neck_side",),
}

# Parameter bounds, from PLAN.md Phase 10 Tier 1. These are style limits —
# a 200mm neckline drop is not a sleeveless top — and are separate from the
# geometric limits above, which depend on the body.
RANGES: dict[str, tuple[float, float]] = {
    "neckline_drop_front_mm": (0.0, 140.0),
    "neckline_drop_back_mm": (0.0, 120.0),
    "neckline_width_mm": (0.0, 60.0),
    "armhole_raise_mm": (0.0, 30.0),
    "hem_change_mm": (-150.0, 0.0),
    "waist_suppression": (0.0, 1.0),
}


@dataclass(frozen=True)
class DesignParameters:
    """What a croquis asks for, as numbers.

    Every field is `Optional` and defaults to `None`, meaning "leave the
    block alone". That is deliberate: a model that cannot tell from a
    sketch how wide the neckline is should say so rather than guess, and
    `None` is the only answer that doesn't quietly invent a design
    decision. `waist_suppression` defaults to 1.0 when asked for because
    1.0 *is* the block — a dropped field and a field set to the block
    default have to mean the same thing.
    """
    neckline_drop_front_mm: Optional[float] = None
    neckline_drop_back_mm: Optional[float] = None
    neckline_width_mm: Optional[float] = None
    armhole_raise_mm: Optional[float] = None
    hem_change_mm: Optional[float] = None
    waist_suppression: Optional[float] = None

    def out_of_range(self) -> list[str]:
        problems = []
        for field, (low, high) in RANGES.items():
            value = getattr(self, field)
            if value is None:
                continue
            if not math.isfinite(value):
                problems.append(f"{field} is not a finite number")
            elif not low <= value <= high:
                problems.append(f"{field}={value:g} is outside {low:g}..{high:g}")
        return problems

    @property
    def is_empty(self) -> bool:
        """No parameter set, or every one set to its block default — either
        way the block itself is the answer and there is nothing to apply."""
        return all(
            value is None or (field == "waist_suppression" and value == 1.0)
            for field, value in dataclasses.asdict(self).items()
        )

    def summary(self) -> str:
        """Human-readable, for the revision reason — the record of why a
        version looks the way it does has to say what was asked for."""
        parts = []
        if self.neckline_drop_front_mm:
            parts.append(f"front neckline {self.neckline_drop_front_mm:.0f}mm lower")
        if self.neckline_drop_back_mm:
            parts.append(f"back neckline {self.neckline_drop_back_mm:.0f}mm lower")
        if self.neckline_width_mm:
            parts.append(f"neckline {self.neckline_width_mm:.0f}mm wider")
        if self.armhole_raise_mm:
            parts.append(f"armhole {self.armhole_raise_mm:.0f}mm higher")
        if self.hem_change_mm:
            parts.append(f"cropped {abs(self.hem_change_mm):.0f}mm")
        if self.waist_suppression is not None and self.waist_suppression != 1.0:
            parts.append(f"waist shaping at {self.waist_suppression:.0%}")
        return ", ".join(parts) if parts else "block as drafted"


# --------------------------------------------------------------------------
# The shared mechanism
# --------------------------------------------------------------------------
Moves = dict[str, dict[str, tuple[float, float]]]   # piece id -> point id -> (dx, dy)


def translate_points(pieces: list[PatternPiece], moves: Moves) -> list[PatternPiece]:
    """Move named points, carrying curve controls with their own endpoint.

    A cubic's c1 belongs to the segment's start and c2 to its end, so
    displacing each control by its endpoint's displacement keeps the
    tangent *direction* at both ends while letting the curve stretch. That
    is what holds the fold-line tangent: on the front neckline c2 sits
    level with `cf_neck`, and after a drop of dy both have moved dy, so it
    still does. The same rule keeps the front and back armholes meeting in
    one direction at the underarm when it rises.

    Returns a new list; the input is untouched.
    """
    result = copy.deepcopy(pieces)
    for piece in result:
        piece_moves = moves.get(piece.id)
        if not piece_moves:
            continue
        unknown = set(piece_moves) - {p.id for p in piece.points}
        if unknown:
            raise KeyError(f"{piece.id} has no point(s) {sorted(unknown)}")
        for point in piece.points:
            move = piece_moves.get(point.id)
            if move:
                point.x += move[0]
                point.y += move[1]
        for edge in piece.edges:
            for segment in edge.segments:
                if segment.kind is not SegmentKind.CUBIC:
                    continue
                for attr, endpoint in (("c1", segment.start), ("c2", segment.end)):
                    move = piece_moves.get(endpoint)
                    control = getattr(segment, attr)
                    if move and control is not None:
                        setattr(
                            segment, attr,
                            (control[0] + move[0], control[1] + move[1]),
                        )
    return result


def _point(piece: PatternPiece, point_id: str):
    return piece.point_map()[point_id]


def _unit(frm, toward) -> tuple[float, float]:
    dx, dy = toward.x - frm.x, toward.y - frm.y
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("cannot take a direction between coincident points")
    return dx / length, dy / length


def _waist_darts(piece: PatternPiece):
    """The darts whose legs sit on the waistline.

    Decided by where the legs actually are rather than by name: each piece
    also carries a shoulder dart, and the two cannot be told apart by id
    (`bust_dart` is a waist dart; `back_shoulder_dart` is not), so a name
    test would have silently cropped a shoulder dart.
    """
    on_waist = {
        point_id
        for edge in piece.edges if edge.id == "waist"
        for segment in edge.segments
        for point_id in (segment.start, segment.end)
    }
    return [d for d in piece.darts if d.leg_a in on_waist and d.leg_b in on_waist]


def _dart_d1_id(apex_id: str) -> str:
    """`back_waist_dart_apex` -> `back_waist_dart_d1`. Derived from the apex
    rather than the dart id because the front waist dart is `bust_dart`
    while its points are named `front_waist_dart_*`."""
    return apex_id.removesuffix("_apex") + "_d1"


def _drop_dart(piece: PatternPiece, dart) -> None:
    """Remove a dart the design has closed, and heal the seam it sat on.

    Two parameters arrive here. `waist_suppression=0` releases a dart's
    intake to nothing, and cropping far enough up takes the hem above a
    dart's apex. Either way the dart is gone, and a dart with coincident
    legs is not a smaller dart — it is three identical points on a seam,
    which divides by zero on the way to rebuilding D1 and leaves zero
    length segments for the exporter to stitch.

    The legs and D1 leave the piece; the apex stays, as an interior
    reference like `bust_point` and the other guide points that sit on no
    edge. The seam is re-emitted as straight lines between whatever points
    survive, which is exact because every waist segment is already a line.
    """
    gone = {dart.leg_a, dart.leg_b, _dart_d1_id(dart.apex)}
    piece.darts = [d for d in piece.darts if d is not dart]
    piece.points = [p for p in piece.points if p.id not in gone]
    piece.notches = [n for n in piece.notches if n.point not in gone]
    for edge in piece.edges:
        chain = [edge.segments[0].start] + [s.end for s in edge.segments]
        if not any(point_id in gone for point_id in chain):
            continue
        if any(s.kind is not SegmentKind.LINE for s in edge.segments):
            raise ValueError(
                f"cannot drop {dart.id}: it sits on the curved edge "
                f"{piece.id}.{edge.id}, whose shape is not line segments"
            )
        survivors = [point_id for point_id in chain if point_id not in gone]
        edge.segments = [
            type(edge.segments[0])(start, end, SegmentKind.LINE)
            for start, end in zip(survivors, survivors[1:])
        ]


def _refresh_dart_points(pieces: list[PatternPiece]) -> list[PatternPiece]:
    """Rebuild every dart's D1 from its legs and apex.

    D1 is derived geometry — the legs' midpoint pushed away from the apex —
    so any function that moves a leg has to recompute it or leave the seam
    edge short where the dart folds shut. Darts with no D1 point (the front
    shoulder dart) are skipped. Mutates and returns the list it is given;
    callers here have already copied.
    """
    for piece in pieces:
        points = piece.point_map()
        for dart in piece.darts:
            d1 = points.get(_dart_d1_id(dart.apex))
            if d1 is None:
                continue
            leg_a, leg_b, apex = points[dart.leg_a], points[dart.leg_b], points[dart.apex]
            d1.x, d1.y = dart_point_out(
                (leg_a.x, leg_a.y), (leg_b.x, leg_b.y), (apex.x, apex.y),
            )
    return pieces


# --------------------------------------------------------------------------
# One function per parameter
# --------------------------------------------------------------------------
def apply_neckline_drop_front(pieces: list[PatternPiece], drop_mm: float) -> list[PatternPiece]:
    """Lower the front neckline at centre front. Drop alone makes a scoop
    or a V; it is the one neckline parameter that may differ front to back."""
    return _apply_neckline_drop(pieces, "bodice_front", drop_mm)


def apply_neckline_drop_back(pieces: list[PatternPiece], drop_mm: float) -> list[PatternPiece]:
    return _apply_neckline_drop(pieces, "bodice_back", drop_mm)


def _apply_neckline_drop(
    pieces: list[PatternPiece], piece_id: str, drop_mm: float,
) -> list[PatternPiece]:
    if drop_mm == 0:
        return copy.deepcopy(pieces)
    piece = next(p for p in pieces if p.id == piece_id)
    neck = _point(piece, _FOLD_NECK[piece_id])
    waist = _point(piece, _FOLD_WAIST[piece_id])
    # The neckline cannot be dropped past the waist: the centre-front edge
    # would reverse and the piece would fold through itself.
    headroom = waist.y - neck.y - _MIN_DART_LENGTH_MM
    if drop_mm > headroom:
        raise ValueError(
            f"a {drop_mm:.0f}mm drop leaves no {piece_id} left: "
            f"the neckline is only {waist.y - neck.y:.0f}mm above the waist"
        )
    return translate_points(pieces, {piece_id: {_FOLD_NECK[piece_id]: (0.0, drop_mm)}})


def apply_neckline_width(pieces: list[PatternPiece], width_mm: float) -> list[PatternPiece]:
    """Widen the neckline on both pieces at once, by sliding each neck point
    along its own shoulder seam.

    One parameter, not two, and not negotiable: the neck point is where the
    shoulder seam *starts*, so moving the front's without the back's leaves
    two shoulder seams of different lengths that cannot be sewn together.
    PLAN.md records an earlier version of the spec that had them
    independent; `check_mated_seams` in the validator now catches it.

    On the front, both legs of the shoulder dart travel together, so the
    dart's intake is preserved exactly and its apex stays put — the wedge
    tilts a little, which is the cost of expressing "slide the sewn neck
    point along the shoulder" on an unsewn half.
    """
    if width_mm == 0:
        return copy.deepcopy(pieces)
    moves: Moves = {}
    for piece_id, ids in _WIDEN_WITH_NECK_SIDE.items():
        piece = next(p for p in pieces if p.id == piece_id)
        frm_id, toward_id = _SHOULDER_LINE[piece_id]
        frm, toward = _point(piece, frm_id), _point(piece, toward_id)
        span = math.hypot(toward.x - frm.x, toward.y - frm.y)
        if width_mm >= span - _MIN_SHOULDER_RUN_MM:
            raise ValueError(
                f"a {width_mm:.0f}mm widening eats the {piece_id} shoulder: "
                f"only {span:.0f}mm from {frm_id} to {toward_id}"
            )
        ux, uy = _unit(frm, toward)
        moves[piece_id] = {pid: (ux * width_mm, uy * width_mm) for pid in ids}
    return _refresh_dart_points(translate_points(pieces, moves))


def apply_armhole_raise(pieces: list[PatternPiece], raise_mm: float) -> list[PatternPiece]:
    """Lift the underarm, cutting the armhole higher and shorter.

    Both underarm points move by the same amount, which keeps the side
    seams equal — they both grow by `raise_mm` — and keeps the front and
    back armholes meeting at the same place. The underarm's x is untouched,
    so bust ease is unchanged; what changes is how high the armhole is cut.

    This is the parameter that interacts with the validator: the armhole's
    concavity and bulge-ratio checks measure a curve against its chord, and
    raising the underarm shortens that chord.
    """
    if raise_mm == 0:
        return copy.deepcopy(pieces)
    moves: Moves = {}
    for piece_id, underarm_id in _UNDERARM.items():
        piece = next(p for p in pieces if p.id == piece_id)
        underarm = _point(piece, underarm_id)
        tip = _point(piece, _SHOULDER_TIP[piece_id])
        depth = underarm.y - tip.y
        if raise_mm >= depth - _MIN_DART_LENGTH_MM:
            raise ValueError(
                f"a {raise_mm:.0f}mm raise closes the {piece_id} armhole: "
                f"it is only {depth:.0f}mm deep"
            )
        # The underarm's waist-level guide point shares its x and marks the
        # waistline, not the underarm level, so it does not move.
        moves[piece_id] = {underarm_id: (0.0, -raise_mm)}
    return translate_points(pieces, moves)


def apply_hem_change(pieces: list[PatternPiece], change_mm: float) -> list[PatternPiece]:
    """Crop the hem. Negative only, for now.

    Cropping is a truncation, not a translation, and that distinction is
    the whole content of this function. The waist dart and the side seam
    both taper toward the body's narrowest point, so a hem 80mm above the
    waist crosses them where they are *wider* than they are at the waist.
    Sliding the waistline up bodily would carry the waist's own intake to a
    level that does not want it, and the finished top would be tight across
    the ribs. Each tapering point is instead re-found where its own line
    crosses the new hem level.

    Lengthening is not the mirror image of this and is not implemented:
    the hem width of a longer top comes from mid-hip circumference, with
    the side seam lifted and the hem curved (PLAN.md Tier 1, constraint 2).
    That needs `hip` and `waist_to_hip`, neither of which is measured yet.
    """
    if change_mm == 0:
        return copy.deepcopy(pieces)
    if change_mm > 0:
        raise ValueError(
            "lengthening needs hip and waist_to_hip measurements "
            "(PLAN.md Phase 10 Tier 1); only cropping is supported"
        )
    crop = -change_mm
    result = copy.deepcopy(pieces)
    for piece in result:
        points = piece.point_map()

        # A hem cropped above a dart's apex does not shorten that dart, it
        # passes it: the shaping it provided is for a level the garment no
        # longer reaches. Short of the apex by less than a sewable dart's
        # length is the same case.
        for dart in _waist_darts(piece):
            remaining = points[dart.leg_a].y - crop - points[dart.apex].y
            if remaining < _MIN_DART_LENGTH_MM:
                _drop_dart(piece, dart)
        points = piece.point_map()

        tapers: list[tuple[str, str]] = [
            (_SIDE_WAIST[piece.id], _UNDERARM[piece.id]),
        ]
        for dart in _waist_darts(piece):
            tapers += [(dart.leg_a, dart.apex), (dart.leg_b, dart.apex)]

        for moving_id, toward_id in tapers:
            moving, toward = points[moving_id], points[toward_id]
            run = moving.y - toward.y
            if crop >= run - _MIN_DART_LENGTH_MM:
                raise ValueError(
                    f"cropping {crop:.0f}mm reaches {toward_id} on "
                    f"{piece.id}: {moving_id} is only {run:.0f}mm below it"
                )
            # Walk up this point's own line to the new hem level, so the
            # taper decides the new x.
            t = (moving.y - crop - toward.y) / run
            moving.x = toward.x + t * (moving.x - toward.x)
            moving.y = moving.y - crop

        # Centre front and centre back run straight down the fold, so they
        # just rise; the waist-level guide points rise with them.
        for point_id in (_FOLD_WAIST[piece.id], _UNDERARM_WAIST[piece.id]):
            points[point_id].y -= crop

    return _refresh_dart_points(result)


def apply_waist_suppression(
    pieces: list[PatternPiece], suppression: float,
) -> list[PatternPiece]:
    """Scale the waist darts' intake. 1.0 is the block, 0.0 is boxy.

    Releasing a dart does not move fabric somewhere else — it adds girth,
    so the finished waist grows by exactly what the darts give up. That is
    the design intent of a boxy top and it is also an ease change, which
    the validator checks against `DesignIntent`; `added_waist_ease_mm`
    below reports how much, and `apply_design` folds it into the intent so
    the two stay consistent.

    The side seam's own share of the suppression is left alone: it is a
    seam, not a dart, and releasing it would change where the side seam
    hangs rather than how much shaping the body gets.
    """
    if suppression == 1.0:
        return copy.deepcopy(pieces)
    if not 0.0 <= suppression <= 1.0:
        raise ValueError(f"waist_suppression must be 0..1, got {suppression!r}")
    result = copy.deepcopy(pieces)
    for piece in result:
        points = piece.point_map()
        for dart in _waist_darts(piece):
            leg_a, leg_b = points[dart.leg_a], points[dart.leg_b]
            intake = math.hypot(leg_a.x - leg_b.x, leg_a.y - leg_b.y)
            if intake * suppression < _MIN_DART_INTAKE_MM:
                _drop_dart(piece, dart)
                continue
            mid_x, mid_y = (leg_a.x + leg_b.x) / 2, (leg_a.y + leg_b.y) / 2
            for leg in (leg_a, leg_b):
                leg.x = mid_x + (leg.x - mid_x) * suppression
                leg.y = mid_y + (leg.y - mid_y) * suppression
    return _refresh_dart_points(result)


def added_waist_ease_mm(pieces: list[PatternPiece], suppression: float) -> float:
    """How much girth releasing the waist darts to `suppression` adds.

    Both pieces are cut on the fold, so each dart's released intake counts
    twice round the body.
    """
    released = 0.0
    for piece in pieces:
        points = piece.point_map()
        for dart in _waist_darts(piece):
            leg_a, leg_b = points[dart.leg_a], points[dart.leg_b]
            intake = math.hypot(leg_a.x - leg_b.x, leg_a.y - leg_b.y)
            released += intake * (1.0 - suppression)
    return released * 2


# --------------------------------------------------------------------------
# Applying a whole design
# --------------------------------------------------------------------------
def apply_design(
    pieces: list[PatternPiece], params: DesignParameters, intent: DesignIntent,
) -> tuple[list[PatternPiece], DesignIntent]:
    """Apply every set parameter, in the order they depend on each other.

    Returns the modified pieces *and* a possibly-modified intent. The
    intent comes back because `waist_suppression` is both a shape change
    and an ease change: releasing a dart adds girth, and the validator
    checks finished girth against the intent's ease. Leaving the intent
    alone would make the validator reject a boxy top for being boxy.

    Width runs before drop (one moves the shoulder end of the neckline, the
    other the fold end, so the order is cosmetic but fixed for
    reproducibility), suppression before cropping (cropping walks up the
    dart legs, so it needs their final positions), and the armhole
    anywhere — it shares no points with the rest.
    """
    problems = params.out_of_range()
    if problems:
        raise ValueError("; ".join(problems))

    result = copy.deepcopy(pieces)
    if params.neckline_width_mm:
        result = apply_neckline_width(result, params.neckline_width_mm)
    if params.neckline_drop_front_mm:
        result = apply_neckline_drop_front(result, params.neckline_drop_front_mm)
    if params.neckline_drop_back_mm:
        result = apply_neckline_drop_back(result, params.neckline_drop_back_mm)
    if params.armhole_raise_mm:
        result = apply_armhole_raise(result, params.armhole_raise_mm)

    if params.waist_suppression is not None and params.waist_suppression != 1.0:
        added = added_waist_ease_mm(result, params.waist_suppression)
        result = apply_waist_suppression(result, params.waist_suppression)
        intent = dataclasses.replace(
            intent, ease_waist_mm=intent.ease_waist_mm + added,
        )
    if params.hem_change_mm:
        result = apply_hem_change(result, params.hem_change_mm)

    return result, intent
