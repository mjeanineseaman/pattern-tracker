"""Tests for the runtime geometry validator (PLAN.md Phase 9.5).

Two things need proving: a real draft passes (or the validator is useless
as a gate, because everything gets rejected), and each check actually
fires on the specific breakage it exists for (or it's decoration).
"""
import dataclasses
import math

from api.drafting.bodice import draft_bodice
from api.drafting.validate import (
    _max_perpendicular_deviation, check_adjustment_magnitude, check_armholes,
    check_darts, check_ease, check_mated_seams, check_necklines_are_monotonic,
    check_points_are_finite, sewn_edge_length, validate_pieces,
)
from api.schema import DesignIntent, FabricClass, GarmentType, Measurements, Point

_IN_TO_MM = 25.4


def _measurements(**overrides) -> Measurements:
    defaults = dict(
        bust=34 * _IN_TO_MM, waist=31.5 * _IN_TO_MM, hip=38 * _IN_TO_MM,
        shoulder_width=15 * _IN_TO_MM, back_length=14 * _IN_TO_MM,
        bust_height=11 * _IN_TO_MM, bust_span=7.5 * _IN_TO_MM,
        armscye_depth=7.5 * _IN_TO_MM, nape_to_bust=11 * _IN_TO_MM,
        neck_circumference=15 * _IN_TO_MM, shoulder_length=5 * _IN_TO_MM,
        back_width=13.0 * _IN_TO_MM,
    )
    defaults.update(overrides)
    return Measurements(**defaults)


def _pieces():
    return draft_bodice(_measurements(), DesignIntent(GarmentType.BODICE, FabricClass.WOVEN))


def _move_point(pieces, piece_id, point_id, dx=0.0, dy=0.0):
    """Return a copy of `pieces` with one point shifted."""
    out = []
    for piece in pieces:
        if piece.id != piece_id:
            out.append(piece)
            continue
        points = [
            dataclasses.replace(p, x=p.x + dx, y=p.y + dy) if p.id == point_id else p
            for p in piece.points
        ]
        out.append(dataclasses.replace(piece, points=points))
    return out


# --------------------------------------------------------------------------
# The gate has to let real work through
# --------------------------------------------------------------------------
def test_a_real_draft_passes_every_check():
    result = validate_pieces(_pieces())
    assert result.ok, result.failures
    assert bool(result) is True


def test_a_sloper_adapted_draft_also_passes():
    """draft_bodice_from_sloper's output goes through the same gate, so a
    correction carried from a fitted block must not trip it."""
    from api.drafting.bodice import draft_bodice_from_sloper

    intent = DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)
    sloper_m = _measurements()
    sloper = _move_point(draft_bodice(sloper_m, intent), "bodice_back", "cb_neck", dy=3.0)

    adapted = draft_bodice_from_sloper(sloper, sloper_m, intent, _measurements(bust=36 * _IN_TO_MM), intent)

    assert validate_pieces(adapted).ok


# --------------------------------------------------------------------------
# Each check fires on its own breakage
# --------------------------------------------------------------------------
def test_non_finite_coordinates_are_caught():
    broken = _move_point(_pieces(), "bodice_front", "bust_point", dx=math.inf)

    failures = check_points_are_finite(broken)

    assert any("non-finite" in f for f in failures)


def test_a_straight_armhole_trips_the_lower_bulge_bound():
    """Checked on the deviation maths directly: a chain of collinear points
    has nowhere to bulge, which is the "that's not an armhole" end of the
    range. Done this way because the real armhole is multi-segment, so
    merely dropping its bezier control points still leaves the remaining
    corner points deviating off the chord."""
    assert _max_perpendicular_deviation([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]) == 0.0


def test_a_grossly_deepened_armhole_is_caught():
    """The Phase 2 bug this whole module exists for: an armhole far too
    deep still produces a closed, renderable piece."""
    pieces = _pieces()
    broken = _move_point(pieces, "bodice_front", "underarm_front", dy=400.0)

    assert check_armholes(broken)


def test_an_outward_bulging_armhole_is_caught():
    """THE bug. Phase 2's original armhole bulged outward by 0.18-0.32x of
    chord, which overlaps the correct draft's own 0.24-0.37x magnitude —
    so a size-only check cannot tell them apart. Mirrors the curve's
    control points across its chord to flip the direction while leaving
    the magnitude untouched, and the check must still fire."""
    pieces = _pieces()
    piece = next(p for p in pieces if p.id == "bodice_front")
    armhole = next(e for e in piece.edges if e.id == "armhole")
    pm = piece.point_map()
    start, end = pm[armhole.segments[0].start], pm[armhole.segments[-1].end]

    def mirror(pt):
        """Reflect a control point across the chord from start to end."""
        ax, ay = start.x, start.y
        dx, dy = end.x - ax, end.y - ay
        norm = dx * dx + dy * dy
        t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / norm
        px, py = ax + t * dx, ay + t * dy
        return (2 * px - pt[0], 2 * py - pt[1])

    flipped = [
        dataclasses.replace(s, c1=mirror(s.c1), c2=mirror(s.c2))
        if s.c1 is not None and s.c2 is not None else s
        for s in armhole.segments
    ]
    broken = [dataclasses.replace(
        piece, edges=[dataclasses.replace(armhole, segments=flipped) if e.id == "armhole" else e
                      for e in piece.edges],
    )]

    assert any("curves outward" in f for f in check_armholes(broken))


def test_armhole_checks_hold_across_body_sizes():
    """The bulge bound is a fraction of the chord, not millimetres: the
    same curve's absolute deviation ranges 37-134mm across these bodies
    while its chord ratio barely moves. An absolute bound tuned on one
    body rejects correct drafts for every other one."""
    intent = DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)
    for bust in (800.0, 920.0, 1100.0, 1300.0):
        measurements = Measurements(
            bust=bust, waist=bust - 180, hip=bust + 60, shoulder_width=380,
            back_length=400, bust_height=240, bust_span=180,
        )
        pieces = draft_bodice(measurements, intent)
        assert check_armholes(pieces) == [], f"bust={bust} should draft a valid armhole"


def test_a_dart_opened_into_a_fan_is_caught():
    """Real darts open 2.6-13.5 degrees at the apex. Swinging a leg right
    around turns the wedge into a fan."""
    pieces = _pieces()
    piece = next(p for p in pieces if p.id == "bodice_back")
    dart = piece.darts[0]
    pm = piece.point_map()
    apex, leg_a = pm[dart.apex], pm[dart.leg_a]
    # rotate leg_a 90 degrees about the apex
    vx, vy = leg_a.x - apex.x, leg_a.y - apex.y
    broken = _move_point(
        pieces, "bodice_back", dart.leg_a,
        dx=(apex.x - vy) - leg_a.x, dy=(apex.y + vx) - leg_a.y,
    )

    assert any("fan, not a dart" in f for f in check_darts(broken))


def test_a_real_drafts_darts_all_pass():
    """Guards the thresholds themselves: a back shoulder dart legitimately
    has both legs on one side of its apex in x, which an earlier version of
    this check wrongly rejected."""
    assert check_darts(_pieces()) == []


def test_a_zero_intake_dart_is_caught():
    pieces = _pieces()
    piece = next(p for p in pieces if p.id == "bodice_back")
    dart = piece.darts[0]
    pm = piece.point_map()
    leg_a, leg_b = pm[dart.leg_a], pm[dart.leg_b]
    broken = _move_point(
        pieces, "bodice_back", dart.leg_a, dx=leg_b.x - leg_a.x, dy=leg_b.y - leg_a.y,
    )

    assert any("zero intake" in f for f in check_darts(broken))


def test_an_apex_sitting_on_its_own_leg_is_caught():
    pieces = _pieces()
    piece = next(p for p in pieces if p.id == "bodice_back")
    dart = piece.darts[0]
    pm = piece.point_map()
    apex, leg_a = pm[dart.apex], pm[dart.leg_a]
    broken = _move_point(
        pieces, "bodice_back", dart.apex, dx=leg_a.x - apex.x, dy=leg_a.y - apex.y,
    )

    assert any("near-zero length" in f for f in check_darts(broken))


def test_a_dart_pointing_at_a_missing_point_is_caught():
    pieces = _pieces()
    piece = next(p for p in pieces if p.id == "bodice_back")
    stripped = dataclasses.replace(
        piece, points=[p for p in piece.points if p.id != piece.darts[0].apex],
    )

    assert any("doesn't exist" in f for f in check_darts([stripped]))


def test_a_backtracking_neckline_is_caught():
    """CLAUDE.md records this exact class of bug: a neckline that looks
    fine as a half-piece but doubles back on itself."""
    pieces = _pieces()
    piece = next(p for p in pieces if p.id == "bodice_front")
    neckline = next(e for e in piece.edges if e.id == "neckline")
    pm = piece.point_map()
    start, end = pm[neckline.segments[0].start], pm[neckline.segments[-1].end]
    # drag the control points far past the end point so the curve reverses
    segs = [
        dataclasses.replace(s, c1=(end.x + 200.0, start.y), c2=(start.x - 200.0, end.y))
        if s.kind.value == "cubic" else s
        for s in neckline.segments
    ]
    broken_edge = dataclasses.replace(neckline, segments=segs)
    broken = [dataclasses.replace(
        piece, edges=[broken_edge if e.id == "neckline" else e for e in piece.edges],
    )]

    assert any("doubles back" in f for f in check_necklines_are_monotonic(broken))


# --------------------------------------------------------------------------
# The magnitude bound on AI-proposed adjustments
# --------------------------------------------------------------------------
def test_a_small_adjustment_is_within_the_bound():
    before = _pieces()
    after = _move_point(before, "bodice_back", "cb_neck", dy=4.0)

    assert check_adjustment_magnitude(before, after, limit_mm=15.0) == []


def test_an_oversized_adjustment_is_reported_not_clamped():
    """Rejecting rather than clamping is deliberate: a model asking for a
    300mm move is wrong about something, and shrinking it to the limit
    would hide that while still applying a change nobody reasoned about."""
    before = _pieces()
    after = _move_point(before, "bodice_back", "cb_neck", dy=300.0)

    failures = check_adjustment_magnitude(before, after, limit_mm=15.0)

    assert len(failures) == 1
    assert "cb_neck moved 300.0mm" in failures[0]
    # and the geometry itself was left exactly as proposed, not quietly modified
    moved = next(p for p in after[1].points if p.id == "cb_neck")
    original = next(p for p in before[1].points if p.id == "cb_neck")
    assert moved.y == original.y + 300.0


def test_points_absent_from_the_original_are_not_magnitude_checked():
    """A genuinely new point has no "before" to measure against; the
    structural checks still cover it."""
    before = _pieces()
    piece = before[0]
    after = [dataclasses.replace(piece, points=[*piece.points, Point("brand_new", 0.0, 0.0)]), *before[1:]]

    assert check_adjustment_magnitude(before, after, limit_mm=15.0) == []


# --------------------------------------------------------------------------
# Walking the seams — mated seams must be the same sewn length
# --------------------------------------------------------------------------
def test_a_real_draft_has_matching_mated_seams():
    assert check_mated_seams(_pieces()) == []


def test_sewn_length_closes_darts_along_the_edge():
    """The back shoulder carries a dart, the front doesn't. Flat on paper
    the back is longer; sewn, they're the same — which is the whole point
    of measuring sewn length rather than polyline length."""
    pieces = _pieces()
    back = next(p for p in pieces if p.id == "bodice_back")
    front = next(p for p in pieces if p.id == "bodice_front")

    flat = sum(
        math.dist(
            (back.point_map()[s.start].x, back.point_map()[s.start].y),
            (back.point_map()[s.end].x, back.point_map()[s.end].y),
        )
        for s in next(e for e in back.edges if e.id == "shoulder").segments
    )
    sewn = sewn_edge_length(back, "shoulder")

    assert sewn < flat  # the dart span was removed
    assert abs(sewn - sewn_edge_length(front, "shoulder")) < 0.01


def test_widening_one_neckline_without_the_other_is_caught():
    """The exact failure a parametric neckline change introduces: moving
    front_neck_side outward shortens the front shoulder, and the two
    seams can no longer be sewn together."""
    broken = _move_point(_pieces(), "bodice_front", "front_shoulder_dart_leg_t", dx=12.0)

    failures = check_mated_seams(broken)

    assert any("shoulder" in f and "sewn together" in f for f in failures)


def test_a_mismatched_side_seam_is_caught():
    broken = _move_point(_pieces(), "bodice_front", "side_waist_front", dy=20.0)

    assert any("side_seam" in f for f in check_mated_seams(broken))


def test_a_missing_partner_piece_is_not_a_failure():
    """Validating one piece alone (a sleeve drafted on its own, say)
    shouldn't complain that its partner isn't present."""
    front_only = [p for p in _pieces() if p.id == "bodice_front"]

    assert check_mated_seams(front_only) == []


# --------------------------------------------------------------------------
# Ease conservation
# --------------------------------------------------------------------------
def _intent():
    return DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)


def test_a_real_draft_conserves_the_ease_it_was_drafted_for():
    assert check_ease(_pieces(), _measurements(), _intent()) == []


def test_ease_holds_across_body_sizes():
    intent = _intent()
    for bust in (800.0, 950.0, 1100.0, 1300.0):
        m = Measurements(bust=bust, waist=bust - 63.5, back_length=355.6,
                         bust_height=279.4, bust_span=190.5)
        assert check_ease(draft_bodice(m, intent), m, intent) == [], f"bust={bust}"


def test_losing_width_at_the_bust_is_caught():
    """Pulling the underarm in narrows the finished garment — the draft no
    longer has the ease it was made for, and no structural check notices."""
    broken = _move_point(_pieces(), "bodice_front", "underarm_front", dx=-15.0)

    assert any("finished bust" in f for f in check_ease(broken, _measurements(), _intent()))


def test_changing_waist_suppression_is_caught():
    """Dart manipulation must conserve suppression. Widening a waist dart
    without compensating elsewhere removes real width from the garment."""
    pieces = _pieces()
    back = next(p for p in pieces if p.id == "bodice_back")
    dart = next(d for d in back.darts if d.id == "back_waist_dart")
    broken = _move_point(pieces, "bodice_back", dart.leg_a, dx=-18.0)

    assert any("finished waist" in f for f in check_ease(broken, _measurements(), _intent()))


def test_ease_is_skipped_when_the_draft_context_is_unknown():
    """validate_pieces without measurements still checks geometry — it
    just can't know what ease the draft was aiming for."""
    result = validate_pieces(_pieces())

    assert result.ok


def test_validate_pieces_runs_the_ease_check_when_given_context():
    broken = _move_point(_pieces(), "bodice_front", "underarm_front", dx=-15.0)

    result = validate_pieces(broken, _measurements(), _intent())

    assert not result.ok
    assert any("finished bust" in f for f in result.failures)


def test_validate_pieces_collects_failures_from_every_check():
    broken = _move_point(_pieces(), "bodice_front", "bust_point", dx=math.nan)

    result = validate_pieces(broken)

    assert not result.ok
    assert bool(result) is False
    assert result.failures
