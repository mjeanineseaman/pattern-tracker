"""Tests for the design step (PLAN.md Phase 10): the model call that reads
a croquis and proposes Tier 1 design parameters and a Tier 2 dart
position, together, from one reply.

Same emphasis as test_ai_adjust.py and for the same reason: this is the
other place a model touches geometry, so every way it can go wrong has to
end with the undesigned draft shipping — never a 500, never a half-applied
design.
"""
import json
from types import SimpleNamespace

from api.drafting.bodice import draft_bodice
from api.drafting.design_from_croquis import (
    build_prompt, design_from_croquis, parse_bust_dart_position,
    parse_design_parameters,
)
from api.schema import DesignIntent, FabricClass, GarmentType, Measurements

_IN_TO_MM = 25.4


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
        bust=34 * _IN_TO_MM, waist=31.5 * _IN_TO_MM, back_length=14 * _IN_TO_MM,
        bust_height=11 * _IN_TO_MM, bust_span=7.5 * _IN_TO_MM,
        armscye_depth=7.5 * _IN_TO_MM, nape_to_bust=11 * _IN_TO_MM,
        neck_circumference=15 * _IN_TO_MM, shoulder_length=5 * _IN_TO_MM,
        back_width=13.0 * _IN_TO_MM,
    )


def _intent() -> DesignIntent:
    return DesignIntent(GarmentType.BODICE, FabricClass.WOVEN)


def _pieces():
    return draft_bodice(_measurements(), _intent())


def _design(client, croquis_path):
    return design_from_croquis(client, _pieces(), _measurements(), _intent(), croquis_path)


def _reply(**params) -> str:
    return json.dumps(params)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------
def test_a_valid_proposal_is_applied(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png-but-bytes-are-bytes")
    client = _FakeClient(_reply(neckline_drop_front_mm=80, hem_change_mm=-40))

    outcome = _design(client, png)

    assert outcome.used_ai
    assert "Designed from the croquis" in outcome.note
    front = next(p for p in outcome.pieces if p.id == "bodice_front")
    before_front = next(p for p in _pieces() if p.id == "bodice_front")
    assert front.point_map()["cf_neck"].y == \
        before_front.point_map()["cf_neck"].y + 80


def test_waist_suppression_in_the_reply_raises_the_returned_intents_ease(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(waist_suppression=0.0))

    outcome = _design(client, png)

    assert outcome.intent.ease_waist_mm > _intent().ease_waist_mm


def test_the_image_is_sent_with_the_prompt(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply())

    _design(client, png)

    content = client.last_create_kwargs["messages"][0]["content"]
    image = next(b for b in content if b["type"] == "image")
    assert image["source"]["media_type"] == "image/png"


def test_the_prompt_carries_the_intent():
    prompt = build_prompt(_intent())
    assert "garment_type: bodice" in prompt


# --------------------------------------------------------------------------
# Every path back to the formula draft
# --------------------------------------------------------------------------
def test_no_croquis_short_circuits_without_calling_the_model():
    client = _FakeClient(_reply())

    outcome = _design(client, None)

    assert client.last_create_kwargs is None
    assert not outcome.used_ai
    assert "nothing to design from" in outcome.note


def test_a_missing_croquis_file_also_short_circuits(tmp_path):
    client = _FakeClient(_reply())

    outcome = _design(client, tmp_path / "gone.png")

    assert client.last_create_kwargs is None
    assert not outcome.used_ai


def test_an_api_failure_falls_back_to_the_formula_draft(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(raises=RuntimeError("Could not resolve authentication method"))
    expected = _pieces()

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "the design step failed" in outcome.note
    assert "authentication" in outcome.note
    front = next(p for p in outcome.pieces if p.id == "bodice_front")
    expected_front = next(p for p in expected if p.id == "bodice_front")
    assert front.point_map()["cf_neck"].y == expected_front.point_map()["cf_neck"].y


def test_unparseable_json_falls_back(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient("looks like a scoop neck, maybe 8cm deep?")

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "unusable" in outcome.note


def test_a_json_array_instead_of_an_object_falls_back(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient('[{"neckline_drop_front_mm": 80}]')

    assert not _design(client, png).used_ai


def test_a_non_numeric_value_falls_back(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient('{"neckline_drop_front_mm": "deep"}')

    assert not _design(client, png).used_ai


def test_an_all_null_reply_is_a_good_answer_not_a_failure(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply())

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "didn't call for any change" in outcome.note


def test_an_out_of_range_value_rejects_the_whole_proposal(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(neckline_width_mm=999))
    expected = _pieces()

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "rejected" in outcome.note
    front = next(p for p in outcome.pieces if p.id == "bodice_front")
    expected_front = next(p for p in expected if p.id == "bodice_front")
    assert front.point_map()["front_neck_side"].x == expected_front.point_map()["front_neck_side"].x


def test_a_value_thats_in_range_but_geometrically_impossible_rejects(tmp_path):
    """999mm is caught by out_of_range; a value still inside the documented
    range can still be geometrically impossible for this body (a crop deep
    enough to eat the armhole) -- both go through the same fallback."""
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(hem_change_mm=-150))

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "rejected" in outcome.note


# --------------------------------------------------------------------------
# Tier 2: the dart position, read from the same reply
# --------------------------------------------------------------------------
def test_a_dart_position_alone_is_applied(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(bust_dart_position="side"))

    outcome = _design(client, png)

    assert outcome.used_ai
    assert "bust dart moved to side" in outcome.note
    front = next(p for p in outcome.pieces if p.id == "bodice_front")
    assert {d.id for d in front.darts} == {"front_shoulder_dart", "side_dart"}


def test_tier_1_and_tier_2_compose_in_one_proposal(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(neckline_drop_front_mm=60, bust_dart_position="french"))

    outcome = _design(client, png)

    assert outcome.used_ai
    assert "front neckline 60mm lower" in outcome.note
    assert "bust dart moved to french" in outcome.note
    front = next(p for p in outcome.pieces if p.id == "bodice_front")
    assert "french_dart" in {d.id for d in front.darts}
    before_front = next(p for p in _pieces() if p.id == "bodice_front")
    assert front.point_map()["cf_neck"].y == before_front.point_map()["cf_neck"].y + 60


def test_a_position_of_waist_is_a_good_answer_not_a_failure(tmp_path):
    """"waist" names the block's own position explicitly, same meaning as
    null -- it shouldn't read as "nothing happened" being a problem."""
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(bust_dart_position="waist"))

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "didn't call for any change" in outcome.note


def test_shoulder_is_rejected_with_its_own_specific_reason(tmp_path):
    """Not offered to the model (see dart_rotation's own module docstring
    for why), but if one arrives anyway it has to reach
    rotate_bust_dart_to()'s specific, documented error -- not a generic
    "unusable JSON" one, and not silently ignored."""
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(bust_dart_position="shoulder"))
    expected = _pieces()

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "isn't implemented yet" in outcome.note
    front = next(p for p in outcome.pieces if p.id == "bodice_front")
    expected_front = next(p for p in expected if p.id == "bodice_front")
    assert {d.id for d in front.darts} == {d.id for d in expected_front.darts}


def test_an_unknown_dart_position_rejects_the_whole_proposal(tmp_path):
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(bust_dart_position="princess"))

    outcome = _design(client, png)

    assert not outcome.used_ai
    assert "unusable" in outcome.note


def test_a_dart_position_that_fails_validation_rejects_the_whole_proposal(tmp_path):
    """A neckline widened enough to eat the shoulder seam AND a dart move
    in the same reply: Tier 1's own failure has to fall the whole
    proposal back, dart position included, not apply one and drop the
    other."""
    png = tmp_path / "sketch.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _FakeClient(_reply(neckline_width_mm=999, bust_dart_position="side"))
    expected = _pieces()

    outcome = _design(client, png)

    assert not outcome.used_ai
    front = next(p for p in outcome.pieces if p.id == "bodice_front")
    expected_front = next(p for p in expected if p.id == "bodice_front")
    assert {d.id for d in front.darts} == {d.id for d in expected_front.darts}


# --------------------------------------------------------------------------
# Parsing, as a unit
# --------------------------------------------------------------------------
def test_fenced_json_is_parsed():
    params = parse_design_parameters(
        '```json\n{"neckline_drop_front_mm": 50}\n```'
    )
    assert params.neckline_drop_front_mm == 50


def test_missing_keys_default_to_none():
    params = parse_design_parameters('{"neckline_drop_front_mm": 50}')
    assert params.neckline_width_mm is None
    assert params.waist_suppression is None


def test_an_explicit_null_is_none():
    params = parse_design_parameters('{"neckline_drop_front_mm": null}')
    assert params.neckline_drop_front_mm is None


def test_parse_bust_dart_position_reads_a_valid_name():
    assert parse_bust_dart_position('{"bust_dart_position": "french"}') == "french"


def test_parse_bust_dart_position_accepts_shoulder_even_though_its_not_offered():
    """Not in the prompt, but still a real name from PLAN.md's fixed set
    -- parsing accepts it so design_from_croquis() can hand it to
    rotate_bust_dart_to() for that function's own specific rejection."""
    assert parse_bust_dart_position('{"bust_dart_position": "shoulder"}') == "shoulder"


def test_parse_bust_dart_position_defaults_to_none():
    assert parse_bust_dart_position('{"neckline_drop_front_mm": 50}') is None


def test_parse_bust_dart_position_rejects_an_unknown_name():
    import pytest
    with pytest.raises(ValueError, match="bust_dart_position"):
        parse_bust_dart_position('{"bust_dart_position": "princess"}')


def test_parse_bust_dart_position_rejects_a_non_string():
    import pytest
    with pytest.raises(ValueError, match="bust_dart_position"):
        parse_bust_dart_position('{"bust_dart_position": 1}')
