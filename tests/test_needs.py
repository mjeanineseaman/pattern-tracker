"""Tests for measurement needs — which values the draft had to guess.

The rule being enforced: use every measurement that is available, and ask
for the missing ones rather than quietly tuning a fallback formula until
the output looks plausible.
"""
from api.drafting.needs import (
    Block, garment_blocks, is_draftable, measurement_needs, missing_blocks,
    missing_requirements, needs_to_dict, required_measurements,
)
from api.schema import GarmentType, Measurements

_REQUIRED = dict(
    bust=920, waist=740, back_length=400, bust_height=240, bust_span=180,
)


def _measurements(**overrides) -> Measurements:
    return Measurements(**{**_REQUIRED, **overrides})


def test_a_bare_measurement_set_reports_every_estimated_field():
    needs = measurement_needs(_measurements())

    assert {n.field for n in needs} == {
        "shoulder_length", "armscye_depth", "back_width",
        "nape_to_bust", "neck_circumference",
    }


def test_a_supplied_measurement_is_no_longer_asked_for():
    needs = measurement_needs(_measurements(armscye_depth=190))

    assert "armscye_depth" not in {n.field for n in needs}


def test_a_complete_set_asks_for_nothing():
    complete = _measurements(
        shoulder_length=127, armscye_depth=190, back_width=330,
        nape_to_bust=280, neck_circumference=380,
    )

    assert measurement_needs(complete) == []


def test_the_fixed_guess_is_asked_for_first():
    """shoulder_length falls back to a flat 12.5cm regardless of body size,
    so it's equally wrong for every body except the one it was picked for.
    A fallback that at least scales with the bust is a better guess."""
    needs = measurement_needs(_measurements())

    assert needs[0].field == "shoulder_length"
    assert needs[0].scales_with_body is False
    assert all(n.scales_with_body for n in needs[1:])


def test_every_need_says_what_it_affects_and_how_to_measure_it():
    """A list of field names is useless to someone holding a tape measure."""
    for need in measurement_needs(_measurements()):
        assert need.affects
        assert need.how_to_measure
        assert need.estimated_as


def test_only_measurements_the_draft_actually_uses_are_requested():
    """Asking for a wrist measurement to draft a bodice would train the
    sewist to ignore the whole list."""
    fields = {n.field for n in measurement_needs(_measurements())}

    assert "wrist_circumference" not in fields
    assert "crotch_depth" not in fields
    assert "thigh_circumference" not in fields


def test_an_unsupported_garment_type_asks_for_nothing_rather_than_guessing():
    """Only the bodice is drafted today. Listing bodice measurements for a
    trouser would be confidently wrong."""
    assert measurement_needs(_measurements(), GarmentType.TROUSER) == []


def test_needs_serialise_for_the_api():
    payload = needs_to_dict(measurement_needs(_measurements()))

    assert payload[0]["field"] == "shoulder_length"
    assert "how_to_measure" in payload[0]


# --------------------------------------------------------------------------
# Hard requirements per garment type — a blocker, not a quality cost
# --------------------------------------------------------------------------
def test_a_bodice_needs_nothing_beyond_the_core_measurements():
    """A bodice block with no sleeve is a sleeveless top: it needs neither
    hip nor shoulder_width."""
    assert missing_requirements(_measurements(), GarmentType.BODICE) == []


def test_a_skirt_needs_a_hip_measurement():
    """hip isn't required by the schema any more, so a skirt has to check
    for it rather than assume a placeholder is there."""
    assert missing_requirements(_measurements(), GarmentType.SKIRT) == ["hip"]
    assert missing_requirements(_measurements(hip=980), GarmentType.SKIRT) == []


def test_a_trouser_needs_hip_and_crotch_depth():
    missing = missing_requirements(_measurements(), GarmentType.TROUSER)

    assert set(missing) == {"hip", "crotch_depth"}


def test_only_the_bodice_is_draftable_today():
    """Drafting a bodice and labelling it a skirt would be confidently the
    wrong garment."""
    assert is_draftable(GarmentType.BODICE)
    for other in (GarmentType.SHIRT, GarmentType.SKIRT,
                  GarmentType.TROUSER, GarmentType.DRESS):
        assert not is_draftable(other), other


def test_every_undraftable_type_still_declares_what_it_will_need():
    """The dependency is known now even though the drafting isn't written,
    which is what lets the tool ask for a measurement before it's needed."""
    for garment_type in (GarmentType.SKIRT, GarmentType.TROUSER,
                         GarmentType.SHIRT, GarmentType.DRESS):
        assert required_measurements(garment_type), garment_type


# --------------------------------------------------------------------------
# Garments are composed of blocks
# --------------------------------------------------------------------------
def test_a_shirt_is_a_bodice_plus_a_sleeve():
    assert garment_blocks(GarmentType.SHIRT) == (Block.BODICE, Block.SLEEVE)


def test_a_dress_is_a_bodice_plus_a_skirt():
    assert garment_blocks(GarmentType.DRESS) == (Block.BODICE, Block.SKIRT)


def test_a_shirt_requires_both_blocks_measurements():
    required = required_measurements(GarmentType.SHIRT)

    assert set(required) >= set(required_measurements(GarmentType.BODICE))
    for sleeve_field in ("shoulder_width", "arm_length", "upper_arm", "armscye_depth"):
        assert sleeve_field in required


def test_a_shared_measurement_is_required_once_not_twice():
    """A dress is a bodice plus a skirt and both want waist."""
    required = required_measurements(GarmentType.DRESS)

    assert len(required) == len(set(required))
    assert required.count("waist") == 1


def test_a_shirt_is_blocked_specifically_on_the_sleeve_block():
    """Half of a shirt — the bodice — is already drafted, so reporting a
    flat "not supported" would hide which piece is actually missing."""
    assert missing_blocks(GarmentType.SHIRT) == [Block.SLEEVE]


def test_a_shirt_needs_the_sleeve_measurements_that_arent_taken_yet():
    missing = missing_requirements(_measurements(), GarmentType.SHIRT)

    assert "shoulder_width" in missing
    assert "arm_length" in missing
    assert "bust" not in missing  # the bodice half is already measured


def test_garment_requirements_follow_their_blocks():
    """Requirements are derived from blocks, not hand-copied, so a change to
    what a bodice needs reaches every garment built on one."""
    for garment_type in (GarmentType.SHIRT, GarmentType.DRESS):
        for field in required_measurements(GarmentType.BODICE):
            assert field in required_measurements(garment_type), (garment_type, field)


def test_a_requirement_is_not_the_same_as_an_estimated_fallback():
    """Conflating them would either block drafting over a nice-to-have or
    let a skirt through with no hip."""
    bodice_estimates = {n.field for n in measurement_needs(_measurements())}
    bodice_requirements = set(required_measurements(GarmentType.BODICE))

    assert bodice_estimates & bodice_requirements == set()
