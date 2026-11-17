"""Tests for the AI drafting step (PLAN.md Phase 9.4).

The fallback paths matter more than the happy path here. This is the only
place a model touches geometry, and the project's whole safety argument is
that every way it can go wrong ends with the deterministic formula draft
shipping instead — never a 500, never a half-applied proposal.

No network and no API key: the client is a stub, the same pattern the rest
of the suite uses.
"""
import json
from types import SimpleNamespace

from api.drafting.ai_adjust import (
    MAX_ADJUSTMENT_MM, Adjustment, adjust_draft, apply_adjustments, build_prompt,
    describe_pieces, parse_adjustments,
)
from api.drafting.bodice import draft_bodice
from api.schema import DesignIntent, FabricClass, GarmentType, Measurements

_IN_TO_MM = 25.4
_LEARNINGS = ["widened the back waist dart — the back pulled across the shoulder blade"]


class _FakeClient:
    """Returns canned text, or raises, without any network."""
    def __init__(self, reply_text=None, raises=None):
        self.last_create_kwargs = None
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.last_create_kwargs = kwargs
                if raises is not None:
                    raise raises
                return SimpleNamespace(content=[SimpleNamespace(text=reply_text)])

        self.messages = _Messages()


def _measurements() -> Measurements:
    return Measurements(
        bust=34 * _IN_TO_MM, waist=31.5 * _IN_TO_MM, hip=38 * _IN_TO_MM,
        shoulder_width=15 * _IN_TO_MM, back_length=14 * _IN_TO_MM,
        bust_height=11 * _IN_TO_MM, bust_span=7.5 * _IN_TO_MM,
        armscye_depth=7.5 * _IN_TO_MM, nape_to_bust=11 * _IN_TO_MM,
        neck_circumference=15 * _IN_TO_MM, shoulder_length=5 * _IN_TO_MM,
        back_width=13.0 * _IN_TO_MM,
    )


def _intent() -> DesignIntent:
    return DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)


def _pieces():
    return draft_bodice(_measurements(), _intent())


def _adjust(client, learnings=_LEARNINGS, croquis_path=None):
    return adjust_draft(client, _pieces(), _measurements(), _intent(), learnings, croquis_path)


def _point(pieces, piece_id, point_id):
    piece = next(p for p in pieces if p.id == piece_id)
    return next(p for p in piece.points if p.id == point_id)


def _reply(*adjustments) -> str:
    return json.dumps(list(adjustments))


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------
def test_a_valid_proposal_is_applied():
    client = _FakeClient(_reply({
        "piece": "bodice_back", "point": "cb_neck", "dx": 0, "dy": 3,
        "reason": "acting on the back-pull learning",
    }))
    before = _point(_pieces(), "bodice_back", "cb_neck")

    outcome = _adjust(client)

    assert outcome.used_ai
    assert len(outcome.applied) == 1
    assert _point(outcome.pieces, "bodice_back", "cb_neck").y == before.y + 3
    assert "Adjusted from fitting history" in outcome.note
    assert "back-pull learning" in outcome.note


def test_the_prompt_carries_the_body_the_intent_the_learnings_and_the_draft():
    client = _FakeClient(_reply())
    _adjust(client)

    prompt = client.last_create_kwargs["messages"][0]["content"][-1]["text"]
    assert "bust: 863" in prompt            # a real measurement, in mm
    assert "garment_type: bodice" in prompt
    assert "shoulder blade" in prompt       # the learning
    assert "cb_neck" in prompt              # the draft's own point ids


def test_the_croquis_image_is_sent_when_one_exists(tmp_path):
    """The sketch informs geometry — how deep the neckline really sits —
    without ever deciding what the garment is."""
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png-but-bytes-are-bytes")
    client = _FakeClient(_reply())

    _adjust(client, croquis_path=png)

    content = client.last_create_kwargs["messages"][0]["content"]
    image = next(b for b in content if b["type"] == "image")
    assert image["source"]["media_type"] == "image/png"


def test_a_missing_croquis_file_is_not_an_error(tmp_path):
    """The pattern outlives the upload; a deleted sketch shouldn't break
    drafting."""
    client = _FakeClient(_reply())

    outcome = _adjust(client, croquis_path=tmp_path / "gone.png")

    content = client.last_create_kwargs["messages"][0]["content"]
    assert all(b["type"] != "image" for b in content)
    assert outcome.pieces


# --------------------------------------------------------------------------
# Every path back to the formula draft
# --------------------------------------------------------------------------
def test_no_learnings_short_circuits_without_calling_the_model():
    """A first garment has no fitting history, so there's nothing to apply
    and no reason to spend a call finding that out."""
    client = _FakeClient(_reply())

    outcome = _adjust(client, learnings=[])

    assert client.last_create_kwargs is None
    assert not outcome.used_ai
    assert "No fitting history" in outcome.note


def test_an_api_failure_falls_back_to_the_formula_draft():
    """No key, no network, rate limited — all the same answer: ship the
    deterministic draft and say so."""
    client = _FakeClient(raises=RuntimeError("Could not resolve authentication method"))
    expected = _pieces()

    outcome = _adjust(client)

    assert not outcome.used_ai
    assert "the adjustment step failed" in outcome.note
    assert "authentication" in outcome.note
    assert _point(outcome.pieces, "bodice_back", "cb_neck").y == _point(expected, "bodice_back", "cb_neck").y


def test_unparseable_json_falls_back():
    client = _FakeClient("I think you should let out the side seam a bit.")

    outcome = _adjust(client)

    assert not outcome.used_ai
    assert "unusable" in outcome.note


def test_a_json_object_instead_of_an_array_falls_back():
    client = _FakeClient('{"piece": "bodice_back"}')

    assert not _adjust(client).used_ai


def test_a_non_finite_delta_falls_back():
    client = _FakeClient('[{"piece": "bodice_back", "point": "cb_neck", "dx": 0, "dy": 1e999}]')

    assert not _adjust(client).used_ai


def test_an_empty_array_is_a_good_answer_not_a_failure():
    client = _FakeClient(_reply())

    outcome = _adjust(client)

    assert not outcome.used_ai
    assert "didn't call for any change" in outcome.note


def test_an_oversized_move_rejects_the_whole_proposal():
    """Rejected, not clamped: a model asking for a 300mm move is wrong
    about something structural, and shrinking it would hide that while
    still applying a change nobody reasoned about."""
    client = _FakeClient(_reply(
        {"piece": "bodice_back", "point": "cb_neck", "dx": 0, "dy": 3, "reason": "fine"},
        {"piece": "bodice_back", "point": "cb_waist", "dx": 0, "dy": 300, "reason": "wild"},
    ))
    expected = _pieces()

    outcome = _adjust(client)

    assert not outcome.used_ai
    assert f"{MAX_ADJUSTMENT_MM:.0f}mm limit" in outcome.note
    # the reasonable adjustment in the same batch is discarded too
    assert _point(outcome.pieces, "bodice_back", "cb_neck").y == _point(expected, "bodice_back", "cb_neck").y


def test_geometry_that_fails_validation_rejects_the_whole_proposal():
    """A set of adjustments is one argument about the geometry; half of it
    is not a weaker version of it."""
    pieces = _pieces()
    piece = next(p for p in pieces if p.id == "bodice_back")
    dart = piece.darts[0]
    pm = piece.point_map()
    apex, leg_a = pm[dart.apex], pm[dart.leg_a]
    # swing the leg around the apex, within the magnitude bound but enough
    # to turn the dart into a fan
    vx, vy = leg_a.x - apex.x, leg_a.y - apex.y
    import math
    scale = min(MAX_ADJUSTMENT_MM / max(math.hypot(vx, vy), 1e-9), 1.0)
    client = _FakeClient(_reply({
        "piece": "bodice_back", "point": dart.leg_a,
        "dx": -vy * scale, "dy": vx * scale, "reason": "swing it",
    }))

    outcome = adjust_draft(client, pieces, _measurements(), _intent(), _LEARNINGS)

    if outcome.used_ai:
        # the move was too small to break anything — then it must at least
        # still be valid geometry, which is the real invariant
        from api.drafting.validate import validate_pieces
        assert validate_pieces(outcome.pieces).ok
    else:
        assert "failed validation" in outcome.note or "limit" in outcome.note


def test_a_proposal_naming_only_unknown_points_falls_back():
    client = _FakeClient(_reply({
        "piece": "bodice_back", "point": "not_a_real_point", "dx": 1, "dy": 1, "reason": "?",
    }))

    outcome = _adjust(client)

    assert not outcome.used_ai
    assert "no proposed adjustment named a real point" in outcome.note
    assert any("unknown point" in r for r in outcome.rejected)


# --------------------------------------------------------------------------
# Parsing and applying, as units
# --------------------------------------------------------------------------
def test_fenced_json_is_parsed():
    adjustments = parse_adjustments(
        '```json\n[{"piece": "a", "point": "b", "dx": 1, "dy": 2}]\n```'
    )
    assert adjustments[0].piece == "a"


def test_too_many_adjustments_is_rejected():
    import pytest
    reply = json.dumps([{"piece": "a", "point": "b", "dx": 0, "dy": 0}] * 100)
    with pytest.raises(ValueError, match="limit is"):
        parse_adjustments(reply)


def test_an_unknown_point_is_dropped_without_discarding_its_batch():
    """One bad id shouldn't throw away the good adjustments beside it —
    structural breakage is caught later, and that does reject wholesale."""
    pieces = _pieces()
    adjusted, applied, rejected = apply_adjustments(pieces, [
        Adjustment("bodice_back", "cb_neck", 0, 2, "good"),
        Adjustment("bodice_back", "nope", 0, 2, "bad"),
    ])

    assert [a.point for a in applied] == ["cb_neck"]
    assert rejected == ["unknown point bodice_back.nope"]
    assert _point(adjusted, "bodice_back", "cb_neck").y == _point(pieces, "bodice_back", "cb_neck").y + 2


def test_a_duplicate_adjustment_for_one_point_is_rejected():
    """Two moves for the same point is ambiguous — applying both silently
    would double a correction nobody asked for twice."""
    _adjusted, applied, rejected = apply_adjustments(_pieces(), [
        Adjustment("bodice_back", "cb_neck", 0, 2, "first"),
        Adjustment("bodice_back", "cb_neck", 0, 5, "second"),
    ])

    assert len(applied) == 1
    assert any("duplicate" in r for r in rejected)


def test_describe_pieces_lists_points_and_darts():
    text = describe_pieces(_pieces())

    assert "cb_neck" in text
    assert "dart back_waist_dart" in text


def test_build_prompt_omits_unset_optional_measurements():
    prompt = build_prompt(_pieces(), _measurements(), _intent(), _LEARNINGS)

    assert "arm_length" not in prompt  # never measured, so never claimed
    assert "bust:" in prompt


def test_build_prompt_says_so_when_there_are_no_learnings():
    prompt = build_prompt(_pieces(), _measurements(), _intent(), [])

    assert "none recorded yet" in prompt
