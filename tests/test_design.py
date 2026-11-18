"""Tests for parametric design changes to a drafted block (PLAN.md Phase 10,
Tier 1).

Three things need proving, in order: each function preserves the fold-line
tangent (CLAUDE.md's own recorded bug, so this is non-negotiable), the
degenerate end of every range either produces a sewable piece or raises
rather than silently inverting a dart, and `apply_design` reports the
ease change a released dart actually makes rather than leaving the
validator checking the block's original intent against a boxier garment.
"""
import math

import pytest

from api.drafting.bodice import draft_bodice
from api.drafting.design import (
    RANGES, DesignParameters, added_waist_ease_mm, apply_armhole_raise,
    apply_design, apply_hem_change, apply_neckline_drop_back,
    apply_neckline_drop_front, apply_neckline_width, apply_waist_suppression,
    translate_points,
)
from api.drafting.validate import check_mated_seams, hem_lift_mm, validate_pieces
from api.export.geometry import mirrored_outline
from api.schema import DesignIntent, FabricClass, GarmentType, Measurements

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


def _intent() -> DesignIntent:
    return DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)


def _pieces() -> list:
    return draft_bodice(_measurements(), _intent())


def _piece(pieces, piece_id):
    return next(p for p in pieces if p.id == piece_id)


def _fold_tangent_dy(piece, fold_point_id: str) -> float:
    """The neckline cubic's control point nearest the fold, minus the fold
    point's own y. Zero means the tangent at the fold is horizontal."""
    pm = piece.point_map()
    seg = next(e for e in piece.edges if e.id == "neckline").segments[0]
    control = seg.c2 if seg.end == fold_point_id else seg.c1
    return control[1] - pm[fold_point_id].y


def _assert_sewable(pieces):
    """The checks every one of these functions has to pass, every time:
    both halves still mirror cleanly, and the two shoulder/side seams
    they're sewn to still match in length."""
    for piece in pieces:
        mirrored_outline(piece)   # raises if the fold-point count is wrong
    assert check_mated_seams(pieces) == []


# --------------------------------------------------------------------------
# The fold-line tangent invariant
# --------------------------------------------------------------------------
def test_base_draft_already_has_horizontal_tangents_at_the_fold():
    front, back = _piece(_pieces(), "bodice_front"), _piece(_pieces(), "bodice_back")
    assert _fold_tangent_dy(front, "cf_neck") == pytest.approx(0.0, abs=1e-6)
    assert _fold_tangent_dy(back, "cb_neck") == pytest.approx(0.0, abs=1e-6)


def test_neckline_drop_front_preserves_the_tangent_and_mirrors_cleanly():
    pieces = apply_neckline_drop_front(_pieces(), 80.0)
    front = _piece(pieces, "bodice_front")
    assert _fold_tangent_dy(front, "cf_neck") == pytest.approx(0.0, abs=1e-6)
    _assert_sewable(pieces)


def test_neckline_drop_back_preserves_the_tangent_and_mirrors_cleanly():
    pieces = apply_neckline_drop_back(_pieces(), 60.0)
    back = _piece(pieces, "bodice_back")
    assert _fold_tangent_dy(back, "cb_neck") == pytest.approx(0.0, abs=1e-6)
    _assert_sewable(pieces)


def test_neckline_width_preserves_both_tangents_and_mirrors_cleanly():
    pieces = apply_neckline_width(_pieces(), 40.0)
    front, back = _piece(pieces, "bodice_front"), _piece(pieces, "bodice_back")
    assert _fold_tangent_dy(front, "cf_neck") == pytest.approx(0.0, abs=1e-6)
    assert _fold_tangent_dy(back, "cb_neck") == pytest.approx(0.0, abs=1e-6)
    _assert_sewable(pieces)


# --------------------------------------------------------------------------
# Neckline width is one parameter, not two
# --------------------------------------------------------------------------
def test_widening_the_neckline_keeps_the_shoulder_seams_equal():
    """The whole reason width isn't independent per piece: moving one neck
    point without the other breaks "front and back shoulder seams match"."""
    before = _pieces()
    after = apply_neckline_width(before, 35.0)
    assert check_mated_seams(after) == []
    # and it's not a no-op either -- the shoulder actually got shorter
    from api.drafting.validate import sewn_edge_length
    assert sewn_edge_length(_piece(after, "bodice_front"), "shoulder") < \
        sewn_edge_length(_piece(before, "bodice_front"), "shoulder")


def test_widening_preserves_the_front_shoulder_darts_intake():
    """Both legs of the front shoulder dart travel together, so the wedge
    it cuts stays the same size -- only its position on the shoulder moves."""
    before, after = _pieces(), apply_neckline_width(_pieces(), 35.0)
    before_front, after_front = _piece(before, "bodice_front"), _piece(after, "bodice_front")
    dart_b = next(d for d in before_front.darts if d.id == "front_shoulder_dart")
    dart_a = next(d for d in after_front.darts if d.id == "front_shoulder_dart")

    def intake(piece, dart):
        pm = piece.point_map()
        a, b = pm[dart.leg_a], pm[dart.leg_b]
        return math.hypot(a.x - b.x, a.y - b.y)

    assert intake(after_front, dart_a) == pytest.approx(intake(before_front, dart_b), abs=1e-6)


def test_widening_rejects_an_amount_that_would_close_the_shoulder():
    with pytest.raises(ValueError, match="eats"):
        apply_neckline_width(_pieces(), 70.0)   # back shoulder is ~71mm


def test_a_zero_width_change_is_a_no_op():
    before, after = _pieces(), apply_neckline_width(_pieces(), 0.0)
    before_front, after_front = _piece(before, "bodice_front"), _piece(after, "bodice_front")
    assert before_front.point_map()["front_neck_side"].x == after_front.point_map()["front_neck_side"].x


# --------------------------------------------------------------------------
# Neckline drop: direction and limits
# --------------------------------------------------------------------------
def test_dropping_the_front_neckline_moves_only_the_front_fold_point_down():
    before, after = _pieces(), apply_neckline_drop_front(_pieces(), 50.0)
    before_cf = _piece(before, "bodice_front").point_map()["cf_neck"]
    after_cf = _piece(after, "bodice_front").point_map()["cf_neck"]
    after_cb = _piece(after, "bodice_back").point_map()["cb_neck"]
    before_cb = _piece(before, "bodice_back").point_map()["cb_neck"]
    assert after_cf.y == pytest.approx(before_cf.y + 50.0)
    assert after_cb.y == before_cb.y   # back untouched


def test_dropping_the_neckline_past_the_waist_is_rejected():
    with pytest.raises(ValueError, match="no bodice_front left"):
        apply_neckline_drop_front(_pieces(), 400.0)


def test_dropping_at_the_top_of_the_allowed_range_still_validates():
    pieces = apply_neckline_drop_front(_pieces(), RANGES["neckline_drop_front_mm"][1])
    assert validate_pieces(pieces, _measurements(), _intent()).ok


# --------------------------------------------------------------------------
# Armhole raise
# --------------------------------------------------------------------------
def test_raising_the_armhole_keeps_the_side_seams_equal_and_lengthens_them():
    before, after = _pieces(), apply_armhole_raise(_pieces(), 25.0)
    from api.drafting.validate import sewn_edge_length
    before_side = sewn_edge_length(_piece(before, "bodice_front"), "side_seam")
    after_side = sewn_edge_length(_piece(after, "bodice_front"), "side_seam")
    assert after_side == pytest.approx(before_side + 25.0, abs=0.5)
    _assert_sewable(after)


def test_raising_the_armhole_does_not_change_bust_width():
    before, after = _pieces(), apply_armhole_raise(_pieces(), 25.0)
    before_underarm = _piece(before, "bodice_front").point_map()["underarm_front"]
    after_underarm = _piece(after, "bodice_front").point_map()["underarm_front"]
    assert after_underarm.x == before_underarm.x


def test_raising_past_the_armhole_depth_is_rejected():
    with pytest.raises(ValueError, match="closes"):
        apply_armhole_raise(_pieces(), 190.0)


def test_armhole_raise_at_the_top_of_the_range_still_validates():
    pieces = apply_armhole_raise(_pieces(), RANGES["armhole_raise_mm"][1])
    assert validate_pieces(pieces, _measurements(), _intent()).ok


# --------------------------------------------------------------------------
# Hem change: crop only
# --------------------------------------------------------------------------
def test_a_positive_hem_change_is_rejected_lengthening_is_not_supported():
    with pytest.raises(ValueError, match="lengthening"):
        apply_hem_change(_pieces(), 50.0)


def test_cropping_shortens_the_side_seam_and_lifts_the_hem_off_the_waist():
    before = _pieces()
    after = apply_hem_change(before, -60.0)
    from api.drafting.validate import sewn_edge_length
    before_side = sewn_edge_length(_piece(before, "bodice_front"), "side_seam")
    after_side = sewn_edge_length(_piece(after, "bodice_front"), "side_seam")
    assert after_side == pytest.approx(before_side - 60.0, abs=0.5)
    assert hem_lift_mm(after, _measurements()) == pytest.approx(60.0, abs=0.5)
    _assert_sewable(after)


def test_a_moderate_crop_leaves_the_waist_darts_in_place():
    """Short of the dart apex, the dart just tapers less -- it isn't dropped."""
    after = apply_hem_change(_pieces(), -30.0)
    assert len(_piece(after, "bodice_front").darts) == 2
    assert len(_piece(after, "bodice_back").darts) == 2


def test_cropping_past_a_darts_apex_drops_that_dart_rather_than_inverting_it():
    after = apply_hem_change(_pieces(), -90.0)
    front = _piece(after, "bodice_front")
    assert "bust_dart" not in {d.id for d in front.darts}
    # its points are gone too, not just the dart record
    assert "front_waist_dart_leg_a" not in front.point_map()
    # and the waist edge still closes, as plain lines, with no gap
    waist = next(e for e in front.edges if e.id == "waist")
    chain = [waist.segments[0].start] + [s.end for s in waist.segments]
    assert chain[0] == "cf_waist" and chain[-1] == "side_waist_front"
    mirrored_outline(front)


def test_cropping_past_the_armhole_depth_is_rejected():
    with pytest.raises(ValueError):
        apply_hem_change(_pieces(), RANGES["hem_change_mm"][0])   # -150, the floor


def test_a_zero_hem_change_is_a_no_op():
    before, after = _pieces(), apply_hem_change(_pieces(), 0.0)
    assert _piece(before, "bodice_front").point_map()["cf_waist"].y == \
        _piece(after, "bodice_front").point_map()["cf_waist"].y


# --------------------------------------------------------------------------
# Waist suppression
# --------------------------------------------------------------------------
def test_full_suppression_is_the_block_default_and_a_no_op():
    before, after = _pieces(), apply_waist_suppression(_pieces(), 1.0)
    assert len(_piece(after, "bodice_front").darts) == len(_piece(before, "bodice_front").darts)
    before_front = _piece(before, "bodice_front").point_map()["front_waist_dart_leg_a"]
    after_front = _piece(after, "bodice_front").point_map()["front_waist_dart_leg_a"]
    assert after_front.x == pytest.approx(before_front.x)


def test_zero_suppression_drops_the_waist_darts_but_not_the_shoulder_darts():
    after = apply_waist_suppression(_pieces(), 0.0)
    front, back = _piece(after, "bodice_front"), _piece(after, "bodice_back")
    assert {d.id for d in front.darts} == {"front_shoulder_dart"}
    assert {d.id for d in back.darts} == {"back_shoulder_dart"}
    _assert_sewable(after)


def test_half_suppression_halves_the_waist_darts_intake():
    before, after = _pieces(), apply_waist_suppression(_pieces(), 0.5)
    from api.drafting.design import _waist_darts

    def intake(piece):
        pm = piece.point_map()
        dart = _waist_darts(piece)[0]
        a, b = pm[dart.leg_a], pm[dart.leg_b]
        return math.hypot(a.x - b.x, a.y - b.y)

    before_front, after_front = _piece(before, "bodice_front"), _piece(after, "bodice_front")
    assert intake(after_front) == pytest.approx(intake(before_front) * 0.5, abs=1e-6)


def test_suppression_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="0..1"):
        apply_waist_suppression(_pieces(), 1.5)


def test_added_waist_ease_is_zero_at_full_suppression_and_positive_below_it():
    pieces = _pieces()
    assert added_waist_ease_mm(pieces, 1.0) == pytest.approx(0.0, abs=1e-6)
    assert added_waist_ease_mm(pieces, 0.0) > 0
    assert added_waist_ease_mm(pieces, 0.0) > added_waist_ease_mm(pieces, 0.5)


# --------------------------------------------------------------------------
# translate_points, the shared mechanism
# --------------------------------------------------------------------------
def test_translate_points_moves_a_cubics_control_with_its_own_endpoint():
    pieces = _pieces()
    front = _piece(pieces, "bodice_front")
    seg = next(e for e in front.edges if e.id == "neckline").segments[0]
    before_c2 = seg.c2
    moved = translate_points(pieces, {"bodice_front": {"cf_neck": (3.0, 7.0)}})
    moved_seg = next(e for e in _piece(moved, "bodice_front").edges if e.id == "neckline").segments[0]
    assert moved_seg.c2 == pytest.approx((before_c2[0] + 3.0, before_c2[1] + 7.0))


def test_translate_points_does_not_mutate_its_input():
    pieces = _pieces()
    before = _piece(pieces, "bodice_front").point_map()["cf_neck"].y
    translate_points(pieces, {"bodice_front": {"cf_neck": (0.0, 99.0)}})
    assert _piece(pieces, "bodice_front").point_map()["cf_neck"].y == before


def test_translate_points_rejects_an_unknown_point_id():
    with pytest.raises(KeyError):
        translate_points(_pieces(), {"bodice_front": {"not_a_real_point": (1.0, 1.0)}})


# --------------------------------------------------------------------------
# DesignParameters
# --------------------------------------------------------------------------
def test_an_all_none_parameter_set_is_empty():
    assert DesignParameters().is_empty


def test_waist_suppression_at_the_block_default_still_counts_as_empty():
    assert DesignParameters(waist_suppression=1.0).is_empty


def test_any_set_field_makes_it_not_empty():
    assert not DesignParameters(neckline_drop_front_mm=10.0).is_empty


def test_out_of_range_reports_each_offending_field():
    problems = DesignParameters(
        neckline_width_mm=200.0, hem_change_mm=50.0,
    ).out_of_range()
    assert len(problems) == 2


def test_out_of_range_catches_a_non_finite_value():
    assert DesignParameters(armhole_raise_mm=float("inf")).out_of_range()


def test_summary_names_what_changed():
    text = DesignParameters(neckline_drop_front_mm=80.0, hem_change_mm=-40.0).summary()
    assert "front neckline" in text
    assert "cropped" in text


def test_summary_of_an_empty_set_says_so():
    assert DesignParameters().summary() == "block as drafted"


# --------------------------------------------------------------------------
# apply_design: the whole thing together
# --------------------------------------------------------------------------
def test_apply_design_with_no_parameters_returns_the_block_unchanged():
    before = _pieces()
    after, intent = apply_design(before, DesignParameters(), _intent())
    assert intent == _intent()
    assert before[0].point_map()["cf_waist" if before[0].id == "bodice_front" else "cb_waist"].y == \
        after[0].point_map()["cf_waist" if after[0].id == "bodice_front" else "cb_waist"].y


def test_apply_design_rejects_out_of_range_parameters_before_touching_geometry():
    with pytest.raises(ValueError):
        apply_design(_pieces(), DesignParameters(neckline_width_mm=999.0), _intent())


def test_apply_design_combination_validates_and_notes_the_skipped_waist_check():
    pieces, intent = apply_design(
        _pieces(),
        DesignParameters(
            neckline_drop_front_mm=90.0, neckline_drop_back_mm=20.0,
            neckline_width_mm=30.0, armhole_raise_mm=15.0,
            hem_change_mm=-60.0, waist_suppression=0.4,
        ),
        _intent(),
    )
    result = validate_pieces(pieces, _measurements(), intent)
    assert result.ok, result.failures
    assert any("waist ease not checked" in n for n in result.notes)
    _assert_sewable(pieces)


def test_apply_design_folds_released_waist_suppression_into_the_intents_ease():
    base_intent = _intent()
    _, intent = apply_design(_pieces(), DesignParameters(waist_suppression=0.0), base_intent)
    assert intent.ease_waist_mm > base_intent.ease_waist_mm


def test_apply_design_leaves_the_intent_alone_when_suppression_is_untouched():
    base_intent = _intent()
    _, intent = apply_design(_pieces(), DesignParameters(neckline_width_mm=20.0), base_intent)
    assert intent.ease_waist_mm == base_intent.ease_waist_mm


def test_apply_design_result_validates_across_the_whole_range_boundary():
    """Every Tier 1 parameter at its most extreme single value, one at a
    time -- the range this ships with, not just a convenient midpoint.

    `hem_change_mm`'s floor (-150mm) is the one case excluded: on this
    body it crops past the armhole's own depth, which is a real geometric
    impossibility this function is right to refuse (see
    `test_cropping_past_the_armhole_depth_is_rejected`) rather than a
    range Tier 1 promises always succeeds.
    """
    for field, (low, high) in RANGES.items():
        for value in (low, high):
            if field == "hem_change_mm" and value == low:
                continue
            pieces, intent = apply_design(_pieces(), DesignParameters(**{field: value}), _intent())
            result = validate_pieces(pieces, _measurements(), intent)
            assert result.ok, f"{field}={value}: {result.failures}"
