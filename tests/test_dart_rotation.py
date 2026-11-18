"""Tests for dart rotation (PLAN.md Phase 10, Tier 2): relocating the
front bust dart without changing how much suppression there is.

Three things need proving, same priority order as test_design.py: the
invariant the whole technique rests on (bust and waist circumference
unchanged -- not approximately, exactly, modulo the few tenths of a
millimetre already present in the base draft's own leg asymmetry), that
`underarm_front` genuinely never moves for the two implemented positions
(the thing that makes them safe where "shoulder" isn't), and every
failure path raises clearly rather than producing a silently broken
piece.
"""
import math

import pytest

from api.drafting.bodice import draft_bodice
from api.drafting.dart_rotation import (
    POSITIONS, french_dart_fraction, rotate_bust_dart_to, side_dart_fraction,
)
from api.drafting.design import apply_hem_change, apply_waist_suppression
from api.drafting.validate import check_mated_seams, sewn_edge_length, validate_pieces
from api.export.geometry import mirrored_outline
from api.schema import DesignIntent, FabricClass, GarmentType, Measurements

_IN_TO_MM = 25.4
_TOLERANCE_MM = 0.5  # the base draft's own leg_a/leg_b asymmetry is ~0.16mm


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


def _intent() -> DesignIntent:
    return DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)


def _pieces() -> list:
    return draft_bodice(_measurements(), _intent())


def _piece(pieces, piece_id):
    return next(p for p in pieces if p.id == piece_id)


def _bust_finished(pieces) -> float:
    front, back = _piece(pieces, "bodice_front"), _piece(pieces, "bodice_back")
    return 2 * (front.point_map()["underarm_front"].x + back.point_map()["underarm_back"].x)


def _assert_sewable(pieces):
    for piece in pieces:
        mirrored_outline(piece)
    assert check_mated_seams(pieces) == []


# --------------------------------------------------------------------------
# The invariant the whole technique rests on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("position", ["side", "french"])
def test_rotating_conserves_bust_and_waist_within_tolerance(position):
    before = _pieces()
    after = rotate_bust_dart_to(before, position)

    assert _bust_finished(after) == pytest.approx(_bust_finished(before), abs=_TOLERANCE_MM)

    before_front, after_front = _piece(before, "bodice_front"), _piece(after, "bodice_front")
    before_waist = sewn_edge_length(before_front, "waist") + sewn_edge_length(
        _piece(before, "bodice_back"), "waist")
    after_waist = sewn_edge_length(after_front, "waist") + sewn_edge_length(
        _piece(after, "bodice_back"), "waist")
    assert after_waist == pytest.approx(before_waist, abs=_TOLERANCE_MM)


@pytest.mark.parametrize("position", ["side", "french"])
def test_rotating_passes_the_full_validator(position):
    pieces = rotate_bust_dart_to(_pieces(), position)
    result = validate_pieces(pieces, _measurements(), _intent())
    assert result.ok, result.failures
    _assert_sewable(pieces)


@pytest.mark.parametrize("position", ["side", "french"])
def test_rotating_holds_across_body_sizes(position):
    """side_dart_fraction/french_dart_fraction are derived from this
    draft's own geometry, not a fixed constant, so a differently
    proportioned body is exactly the thing that could break them.

    Scaled proportionally with bust, the way a real size chart would --
    NOT validate.py's own ease-conservation range, which holds
    bust_height fixed while bust runs from 31 to 51in specifically to
    stress-test arithmetic, producing bodies with no plausible proportions
    (see test_a_disproportionate_body_raises_rather_than_producing_a_bad_seam
    below for exactly that case, and why it has to raise)."""
    for bust_in in (28, 30, 32, 34, 38, 42, 46, 50, 56):
        scale = bust_in / 34.0
        m = _measurements(**{
            field: getattr(_measurements(), field) * scale
            for field in ("bust", "waist", "back_length", "bust_height", "bust_span",
                          "armscye_depth", "nape_to_bust", "neck_circumference",
                          "shoulder_length", "back_width")
        })
        pieces = rotate_bust_dart_to(draft_bodice(m, _intent()), position)
        result = validate_pieces(pieces, m, _intent())
        assert result.ok, f"bust={bust_in}in: {result.failures}"
        _assert_sewable(pieces)


@pytest.mark.parametrize("position", ["side", "french"])
def test_a_disproportionate_body_raises_rather_than_producing_a_bad_seam(position):
    """Found by testing validate.py's own ease-conservation body range: a
    fixed bust_height against a growing bust walks bust_point up above
    underarm_front, and the side/french target -- a linear interpolation
    between underarm_front and side_waist_front -- extrapolates past
    underarm_front instead of landing on the seam. The conservation proof
    this whole technique rests on needs the target to sit BETWEEN the two
    (module docstring); outside that it silently doesn't hold, so this has
    to raise rather than hand back a mismatched side seam.

    bust=1100 specifically: "side" is already invalid at bust=800
    (fraction -0.28) and "french"'s own fraction (biased toward the
    waist, see french_dart_fraction) doesn't go invalid until further
    along -- bust=1100 is where both are, so one body covers both
    parametrized cases."""
    m = Measurements(bust=1100.0, waist=1036.5, back_length=355.6,
                      bust_height=279.4, bust_span=190.5)
    with pytest.raises(ValueError, match="can't be placed on this body"):
        rotate_bust_dart_to(draft_bodice(m, _intent()), position)


# --------------------------------------------------------------------------
# underarm_front never moves for an implemented position -- the property
# that makes side/french safe and shoulder isn't
# --------------------------------------------------------------------------
@pytest.mark.parametrize("position", ["side", "french"])
def test_underarm_front_never_moves(position):
    before, after = _pieces(), None
    after = rotate_bust_dart_to(before, position)
    before_underarm = _piece(before, "bodice_front").point_map()["underarm_front"]
    after_underarm = _piece(after, "bodice_front").point_map()["underarm_front"]
    assert (after_underarm.x, after_underarm.y) == (before_underarm.x, before_underarm.y)


@pytest.mark.parametrize("position", ["side", "french"])
def test_the_armhole_and_shoulder_are_untouched(position):
    """Only the side seam's own dart points and side_waist_front should
    differ -- everything from underarm_front onward (armhole, shoulder)
    is on the fixed side of the cut."""
    before, after = _piece(_pieces(), "bodice_front"), None
    after = _piece(rotate_bust_dart_to(_pieces(), position), "bodice_front")
    for point_id in (
        "underarm_front", "front_armhole_b2", "front_armhole_a2",
        "front_shoulder_tip", "front_shoulder_dart_leg_t", "front_neck_side",
    ):
        b, a = before.point_map()[point_id], after.point_map()[point_id]
        assert (a.x, a.y) == (b.x, b.y), f"{point_id} moved"


# --------------------------------------------------------------------------
# What actually changed
# --------------------------------------------------------------------------
def test_waist_is_a_no_op():
    before, after = _pieces(), rotate_bust_dart_to(_pieces(), "waist")
    before_front, after_front = _piece(before, "bodice_front"), _piece(after, "bodice_front")
    assert {d.id for d in before_front.darts} == {d.id for d in after_front.darts}
    for point_id in ("cf_waist", "side_waist_front", "underarm_front"):
        b, a = before_front.point_map()[point_id], after_front.point_map()[point_id]
        assert (a.x, a.y) == (b.x, b.y)


@pytest.mark.parametrize("position,dart_id", [("side", "side_dart"), ("french", "french_dart")])
def test_the_bust_dart_is_replaced_by_the_named_dart(position, dart_id):
    front = _piece(rotate_bust_dart_to(_pieces(), position), "bodice_front")
    assert {d.id for d in front.darts} == {"front_shoulder_dart", dart_id}


def test_the_new_dart_sits_on_the_side_seam():
    front = _piece(rotate_bust_dart_to(_pieces(), "side"), "bodice_front")
    side_seam = next(e for e in front.edges if e.id == "side_seam")
    chain = [side_seam.segments[0].start] + [s.end for s in side_seam.segments]
    assert chain[0] == "underarm_front" and chain[-1] == "side_waist_front"
    assert len(side_seam.segments) == 4  # endpoint -> leg_a -> d1 -> leg_b -> endpoint


def test_the_waist_edge_is_healed_to_a_plain_line():
    front = _piece(rotate_bust_dart_to(_pieces(), "side"), "bodice_front")
    waist = next(e for e in front.edges if e.id == "waist")
    chain = [waist.segments[0].start] + [s.end for s in waist.segments]
    assert chain == ["cf_waist", "side_waist_front"]


def test_french_sits_lower_on_the_side_seam_than_side():
    front = _piece(_pieces(), "bodice_front")
    assert french_dart_fraction(front) > side_dart_fraction(front)


def test_rotating_does_not_mutate_its_input():
    pieces = _pieces()
    before_front = _piece(pieces, "bodice_front")
    before_dart_ids = {d.id for d in before_front.darts}
    before_underarm = (before_front.point_map()["underarm_front"].x,
                        before_front.point_map()["underarm_front"].y)

    rotate_bust_dart_to(pieces, "side")

    after_front = _piece(pieces, "bodice_front")
    assert {d.id for d in after_front.darts} == before_dart_ids
    after_underarm = (after_front.point_map()["underarm_front"].x,
                       after_front.point_map()["underarm_front"].y)
    assert after_underarm == before_underarm


# --------------------------------------------------------------------------
# Every failure path raises clearly rather than producing broken geometry
# --------------------------------------------------------------------------
def test_shoulder_is_not_yet_implemented():
    with pytest.raises(ValueError, match="isn't implemented yet"):
        rotate_bust_dart_to(_pieces(), "shoulder")


def test_an_unknown_position_raises():
    with pytest.raises(ValueError, match="unknown bust_dart_position"):
        rotate_bust_dart_to(_pieces(), "princess")


@pytest.mark.parametrize("position", ["side", "french"])
def test_rotating_a_dart_thats_already_been_released_raises(position):
    """Tier 1's waist_suppression=0.0 can drop bust_dart entirely before
    Tier 2 ever runs; rotating a dart that no longer exists has to say so,
    not pick some other dart or crash on a missing key."""
    suppressed = apply_waist_suppression(_pieces(), 0.0)
    with pytest.raises(ValueError, match="no bust_dart"):
        rotate_bust_dart_to(suppressed, position)


@pytest.mark.parametrize("position", ["side", "french"])
def test_rotating_a_dart_thats_already_been_cropped_away_raises(position):
    cropped = apply_hem_change(_pieces(), -90.0)  # crops past the bust dart's apex
    with pytest.raises(ValueError, match="no bust_dart"):
        rotate_bust_dart_to(cropped, position)
