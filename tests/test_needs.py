"""Tests for measurement needs — which values the draft had to guess.

The rule being enforced: use every measurement that is available, and ask
for the missing ones rather than quietly tuning a fallback formula until
the output looks plausible.
"""
from api.drafting.needs import measurement_needs, needs_to_dict
from api.schema import GarmentType, Measurements

_REQUIRED = dict(
    bust=920, waist=740, hip=980, shoulder_width=380,
    back_length=400, bust_height=240, bust_span=180,
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
