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
from enum import Enum

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


class Block(str, Enum):
    """A drafted pattern block. Garments are made of these.

    Separating blocks from garments is what lets a shirt be defined as "a
    bodice plus a sleeve" rather than as a hand-copied list of
    measurements that can drift out of step with the bodice's own.
    """
    BODICE = "bodice"
    SLEEVE = "sleeve"
    SKIRT = "skirt"
    TROUSER = "trouser"


# What each block needs *measured* before it can be drafted at all —
# distinct from the estimated-fallback list above, which is about quality.
# These are hard prerequisites: without them there is nothing to draft from.
_BLOCK_MEASUREMENTS: dict[Block, tuple[str, ...]] = {
    Block.BODICE: ("bust", "waist", "back_length", "bust_span"),
    Block.SLEEVE: ("shoulder_width", "arm_length", "upper_arm", "armscye_depth"),
    Block.SKIRT: ("waist", "hip"),
    Block.TROUSER: ("waist", "hip", "crotch_depth"),
}

# Which blocks each garment is built from. A shirt is a bodice and a sleeve;
# a dress is a bodice and a skirt. Requirements are derived from this rather
# than listed per garment, so a change to what a bodice needs reaches every
# garment built on one automatically.
_GARMENT_BLOCKS: dict[GarmentType, tuple[Block, ...]] = {
    GarmentType.BODICE: (Block.BODICE,),
    GarmentType.SHIRT: (Block.BODICE, Block.SLEEVE),
    GarmentType.SKIRT: (Block.SKIRT,),
    GarmentType.TROUSER: (Block.TROUSER,),
    GarmentType.DRESS: (Block.BODICE, Block.SKIRT),
}

# Blocks api/drafting can actually produce. Everything else is declared but
# unbuilt — knowing a dependency before it's needed is what lets the tool
# ask for a measurement ahead of time rather than at the moment it blocks.
IMPLEMENTED_BLOCKS: frozenset[Block] = frozenset({Block.BODICE})


def garment_blocks(garment_type: GarmentType) -> tuple[Block, ...]:
    return _GARMENT_BLOCKS.get(garment_type, ())


def missing_blocks(garment_type: GarmentType) -> list[Block]:
    """Which of this garment's blocks aren't drafted yet. A shirt needs a
    sleeve block that doesn't exist, so it reports that specifically rather
    than a flat "not supported"."""
    return [b for b in garment_blocks(garment_type) if b not in IMPLEMENTED_BLOCKS]


def is_draftable(garment_type: GarmentType) -> bool:
    """Draftable only if every block it's made of is implemented. A bodice
    block with no sleeve is a sleeveless top — the honest extent of what
    this tool makes today."""
    blocks = garment_blocks(garment_type)
    return bool(blocks) and not missing_blocks(garment_type)


def required_measurements(garment_type: GarmentType) -> tuple[str, ...]:
    """The union of every constituent block's requirements, in a stable
    order with duplicates collapsed — a shirt needs the bodice's waist once,
    not twice."""
    seen: dict[str, None] = {}
    for block in garment_blocks(garment_type):
        for field in _BLOCK_MEASUREMENTS[block]:
            seen.setdefault(field, None)
    return tuple(seen)


def missing_requirements(
    measurements: Measurements, garment_type: GarmentType,
) -> list[str]:
    """Measurements this garment type can't be drafted without.

    Separate from `measurement_needs`: a missing requirement is a blocker,
    a missing need is a quality cost. Conflating them would either block
    drafting over a nice-to-have or let a skirt be drafted with no hip.
    """
    return [
        field for field in required_measurements(garment_type)
        if getattr(measurements, field, None) is None
    ]


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
