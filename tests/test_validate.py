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
    check_darts, check_necklines_are_monotonic, check_points_are_finite, validate_pieces,
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


def test_validate_pieces_collects_failures_from_every_check():
    broken = _move_point(_pieces(), "bodice_front", "bust_point", dx=math.nan)

    result = validate_pieces(broken)

    assert not result.ok
    assert bool(result) is False
    assert result.failures
