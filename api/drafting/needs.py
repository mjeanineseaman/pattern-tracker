"""Which measurements the draft is estimating rather than using.

`Measurements` carries a handful of optional fields. When one is unset the
bodice draft falls back to deriving it — usually from bust, in one case
from a flat constant. Those fallbacks are reasonable defaults, but they
are guesses about a body, and a guess about a body produces a garment that
needs more fitting than it should have.

The project's standing rule is that drafting formulas need a human rather
than a model averaging remembered numbers (CLAUDE.md, "things that need a
human"). This module applies the same logic one level up: rather than
quietly tuning a fallback formula until the output looks plausible, use
every measurement that *is* available and tell the sewist which missing
one would most improve the next draft. Asking for a number costs one pass
with a tape measure; guessing it costs a muslin.

Pure functions, no I/O. Nothing here changes the draft — it only reports
what the draft had to invent.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from api.schema import GarmentType, Measurements


@dataclass(frozen=True)
class MeasurementNeed:
    """One measurement the draft wants but doesn't have."""
    field: str
    affects: str          # what in the pattern this measurement actually decides
    estimated_as: str     # how it's being filled in meanwhile
    scales_with_body: bool
    how_to_measure: str

    @property
    def priority(self) -> int:
        """A fallback that doesn't scale with the body is the worst kind:
        it's equally wrong for every body except the one it was picked for.
        Those come first."""
        return 0 if not self.scales_with_body else 1


# Keyed to what `_shared_frame` in api.drafting.bodice actually does when a
# field is None. Kept beside that logic in spirit; if a fallback there
# changes, the description here has to change with it.
_BODICE_NEEDS: tuple[MeasurementNeed, ...] = (
    MeasurementNeed(
        field="shoulder_length",
        affects="where the shoulder tip lands, which sets the top of the armhole curve",
        estimated_as="a fixed 12.5cm, the same for every body",
        scales_with_body=False,
        how_to_measure="from the side of the neck to the bony point at the end of the shoulder",
    ),
    MeasurementNeed(
        field="armscye_depth",
        affects="how deep the armhole is cut",
        estimated_as="a quarter of the bust plus 5cm",
        scales_with_body=True,
        how_to_measure="from the nape of the neck straight down to the underarm level",
    ),
    MeasurementNeed(
        field="back_width",
        affects="how wide the back is across the shoulder blades, and the back armhole",
        estimated_as="a quarter of the bust plus 3cm",
        scales_with_body=True,
        how_to_measure="straight across the back at shoulder-blade level, armhole to armhole",
    ),
    MeasurementNeed(
        field="nape_to_bust",
        affects="where the bustline sits, and so the bust dart",
        estimated_as="68% of the nape-to-waist length",
        scales_with_body=True,
        how_to_measure="from the nape of the neck straight down to the bust line",
    ),
    MeasurementNeed(
        field="neck_circumference",
        affects="the neckline width at the back",
        estimated_as="a sixth of the bust plus 6cm",
        scales_with_body=True,
        how_to_measure="around the base of the neck, where a collar would sit",
    ),
)

_NEEDS_BY_GARMENT: dict[GarmentType, tuple[MeasurementNeed, ...]] = {
    GarmentType.BODICE: _BODICE_NEEDS,
}


def measurement_needs(
    measurements: Measurements, garment_type: GarmentType = GarmentType.BODICE,
) -> list[MeasurementNeed]:
    """Which measurements this draft had to estimate, worst guess first.

    Only reports fields the draft genuinely uses — asking for a wrist
    measurement to draft a bodice would train the sewist to ignore the
    whole list.
    """
    candidates = _NEEDS_BY_GARMENT.get(garment_type, ())
    missing = [
        need for need in candidates if getattr(measurements, need.field, None) is None
    ]
    return sorted(missing, key=lambda n: (n.priority, n.field))


def needs_to_dict(needs: list[MeasurementNeed]) -> list[dict]:
    return [dataclasses.asdict(n) for n in needs]
