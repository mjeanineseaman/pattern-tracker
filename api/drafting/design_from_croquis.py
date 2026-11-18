"""The design step — the model call that reads a croquis and proposes
Tier 1 design parameters (PLAN.md Phase 10).

This is the fix for the bug the rest of Phase 10 exists to close: the
croquis was stored, referenced, and never read. `adjust_draft()`
(api.drafting.ai_adjust) only runs on fitting history, so a brand-new
pattern with zero history never called a model at all, and when it did,
that call was explicitly forbidden from redrafting. This step is separate,
runs first, and is gated on the croquis existing rather than on history —
so it runs on pattern one.

Same bound discipline as the fit step, for the same reason (CLAUDE.md: an
armhole thirty times too deep passed every check it was given once a model
could emit coordinates). The model never sees or emits a coordinate here
either — it picks named parameters from api.drafting.design.RANGES, and
api.drafting.design.apply_design() is the only thing that turns a
parameter into geometry. A parameter can be range-checked before it ever
touches a point; a regenerated coordinate set can only be re-validated
wholesale.

Every failure path returns the undesigned draft unchanged, same as
adjust_draft(): no croquis, no key, no network, unusable JSON, an
out-of-range or geometrically-impossible parameter, a result that fails
the validator. Design must not become unavailable because an API call, or
a sketch, did.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from api.drafting.ai_adjust import image_block
from api.drafting.design import RANGES, DesignParameters, apply_design
from api.drafting.validate import validate_pieces
from api.schema import DesignIntent, Measurements, PatternPiece

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$")

_PARAMETER_LINES = "\n".join(
    f"  {field} ({low:g} to {high:g})" for field, (low, high) in RANGES.items()
)

_SYSTEM_PROMPT = f"""\
You look at a croquis sketch and translate its silhouette into a small set \
of named design parameters. A deterministic drafting formula has already \
built a basic sleeveless-top block; your job is to say how this sketch's \
garment differs from that block, not to draft anything yourself.

Today this tool only drafts a bodice block with no sleeve — a sleeveless \
top. If the sketch shows sleeves, a collar, a placket, or anything else \
this block can't represent, ignore that part and describe only what the \
parameters below can.

Respond with ONLY one JSON object, no prose and no markdown fences, with \
exactly these keys. Use null for any you can't read from the sketch —
null means "leave the block's own default," which is a better answer \
than a guess:
{_PARAMETER_LINES}

Notes on specific parameters:
- neckline_drop_front_mm / neckline_drop_back_mm: how much lower than the \
block's own neckline the sketch sits, at centre front and centre back. \
They may differ — that's what makes a scoop or a V rather than a scoop \
on both sides.
- neckline_width_mm: one number for both front and back, because the \
neckline's width is where the shoulder seam starts — widening it on only \
one piece would make the front and back shoulder seams different \
lengths, which cannot be sewn. Read it from whichever side shows it more \
clearly.
- armhole_raise_mm: how much higher and smaller the sketch's armhole is \
cut than the block's.
- hem_change_mm: 0 to -150, a crop only. This tool cannot lengthen a hem \
yet (it needs hip measurements it doesn't take). If the sketch's hem is \
LONGER than the block's, return 0 for this field, never a positive \
number.
- waist_suppression: 1.0 is the block's own fitted waist dart, 0.0 is a \
boxy top with no waist shaping. Read it as roughly how fitted the \
sketch's waist looks, not as a measurement.

Never invent a number you can't actually see in the sketch — null is \
always better than a guess.
"""


@dataclass
class DesignOutcome:
    """Detailed enough to go straight into a revision's reason, same as
    api.drafting.ai_adjust.AdjustmentOutcome, so the design step's
    contribution to a pattern is visible in the timeline.

    `params` is carried on every fallback path too, including a rejected
    one, purely as a diagnostic — it's what was parsed, not what was
    applied. `used_ai` has to be its own field rather than derived from
    `params.is_empty` for exactly that reason: a 999mm proposal is
    non-empty and still rejected.
    """
    pieces: list[PatternPiece]
    intent: DesignIntent
    params: Optional[DesignParameters]
    note: str
    used_ai: bool = False


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return _FENCE_RE.sub("", stripped).strip()


def build_prompt(intent: DesignIntent) -> str:
    return (
        f"Design intent:\n"
        f"  garment_type: {intent.garment_type.value}\n"
        f"  fabric_class: {intent.fabric_class.value}\n"
        f"  fit: {intent.fit}\n"
        f"  style_notes: {intent.style_notes or '(none)'}\n\n"
        f"Return the JSON object of design parameters for the attached sketch."
    )


def parse_design_parameters(raw_text: str) -> DesignParameters:
    """Raises ValueError on anything malformed; the caller turns that into
    a fallback rather than retrying blindly."""
    parsed = json.loads(_strip_fences(raw_text))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")

    values: dict[str, Optional[float]] = {}
    for field in RANGES:
        raw = parsed.get(field)
        if raw is None:
            values[field] = None
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{field} must be a number or null: {e}") from e
        values[field] = value
    return DesignParameters(**values)


def design_from_croquis(
    client,
    pieces: list[PatternPiece],
    measurements: Measurements,
    intent: DesignIntent,
    croquis_path: Optional[Path] = None,
) -> DesignOutcome:
    """Read a croquis and apply the Tier 1 parameters it calls for.

    Always returns usable pieces and an intent to match them: every
    failure path returns the pieces and intent it was given, unchanged,
    with a note saying why. Called before `adjust_draft()`
    (api.drafting.ai_adjust) — design reshapes the garment; the fit step
    then corrects that result for this specific body, so it has to see
    whatever this step actually did, not the formula block.
    """
    def fallback(note: str, params: Optional[DesignParameters] = None) -> DesignOutcome:
        return DesignOutcome(pieces=list(pieces), intent=intent, params=params, note=note)

    image = image_block(croquis_path)
    if image is None:
        return fallback(
            "No croquis on file for this pattern, so there's nothing to design "
            "from — the draft is the formula block unchanged."
        )

    try:
        response = client.messages.create(
            model=MODEL, max_tokens=512, system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [image, {
                "type": "text", "text": build_prompt(intent),
            }]}],
        )
        raw_text = response.content[0].text if response.content else ""
    except Exception as e:  # network, auth, rate limit — all the same answer here
        logger.warning("design step failed, using the formula draft: %s", e)
        return fallback(f"Drafted from the formulas — the design step failed ({e}).")

    try:
        params = parse_design_parameters(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        return fallback(f"Drafted from the formulas — the design proposal was unusable ({e}).")

    if params.is_empty:
        return fallback(
            "The croquis didn't call for any change from the formula block.", params,
        )

    try:
        designed_pieces, designed_intent = apply_design(pieces, params, intent)
    except ValueError as e:
        # Covers both an out-of-range parameter and a value that's in
        # range but geometrically impossible on this body (e.g. a
        # neckline widened past the shoulder) — design.py raises the same
        # way for both, and either is one bad argument about the garment,
        # not a partial one to salvage.
        return fallback(f"Drafted from the formulas — the design proposal was rejected ({e}).", params)

    validation = validate_pieces(designed_pieces, measurements, designed_intent)
    if not validation.ok:
        return fallback(
            "Drafted from the formulas — the designed geometry failed validation: "
            f"{'; '.join(validation.failures)}",
            params,
        )

    return DesignOutcome(
        pieces=designed_pieces, intent=designed_intent, params=params,
        note=f"Designed from the croquis: {params.summary()}.", used_ai=True,
    )
