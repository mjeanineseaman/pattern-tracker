# Build plan

Each phase is sized to roughly one Claude Code session and ends with something
runnable. Work them in order — later phases assume earlier ones are stable.
Phase 2 is the risky one; everything else is ordinary CRUD and plumbing.

Status: **Phases 0-3, 5 and 7 done; 4 and 6 partially done.** Phase 5 (revise,
revisions, diff, mark-sloper) is fully built and tested. Phase 6's
reference-sloper half (`reference_slopers` table, the `/slopers/{type}`
lookup, `draft_bodice_from_sloper()`) and its RAG layer (local hashed
bag-of-words embeddings, retrieve, answer, `POST /patterns/{id}/ask`) are
built and tested. Phase 4/6's missing join point is now built too —
`POST /patterns` checks `GET /slopers/{garment_type}` and drafts from the
sloper when one's marked, from standard formulas otherwise — and Phase 7's
frontend exercises it for real, so `draft_bodice_from_sloper()` now does run
end to end. Phase 4's note on defaulting a new pattern's measurements from
the most recent existing one is also done now — the Phase 7 frontend fetches
the most recent entry from `GET /patterns` (already sorted newest-first) and
reads its `.measurements` straight from `GET /patterns/{id}`, a plain local
lookup against data already on this machine, deliberately **not** routed
through the RAG/`ask` layer — that layer exists for synthesizing an answer
over revision *history*, not for sourcing a number as plain as "what did I
measure last time," which is just sitting in the most recent row. Still
missing: Phase 6's **body profile** (a persistent record distinct from a
single pattern's `Measurements` snapshot — see Phase 6 below for why that's
a different thing from this defaulting). Phase 4's `POST /croquis` is still
only tested against a fake Anthropic client, not the real API. Print scale
(Phase 3) and fit (Phase 2) still need physical verification — a ruler and a
muslin, not something to automate.

**2026-10-17 — what's actually left, in priority order:**
1. **Phase 6's body profile** — its own table/schema, seeded from the full
   measurement set, distinct from a per-version `Measurements` snapshot. Now
   also the first step of Phase 9, where it becomes the "body model."
2. Validate `POST /croquis` and `POST /patterns/{id}/ask` against the real
   Anthropic API with a real key — everything so far is tested against a
   stub client, and both 500 right now with the `.env` key left empty.
   Phase 9 depends on this working, so it comes before it.
3. **Phase 9 (AI-mediated drafting and the learning loop)** — newly specified,
   not started. This is the phase that makes the flow match the intended one:
   the AI drafts against the block math, the body model and past garments'
   learnings, and muslin corrections feed back. Highest-risk phase in the
   project — it puts a model in the geometry path for the first time. Read its
   "Limitations and risks" section before starting.
4. **Phase 8 (polish)** — not started: backup automation, JSON export with
   full history, revision tags for muslin rounds.

**Note on the current flow vs. the intended one (2026-10-17):** today the AI
does *not* draft. `POST /patterns` is pure math, the PDF exporter is pure
rendering, and the RAG layer only answers questions — it never feeds drafting.
The only thing that carries a fitting correction into a later pattern is the
manually-marked reference sloper. Phase 9 closes that gap deliberately rather
than by accident.

The AI is also currently in the **wrong place**: `POST /croquis` makes a vision
call that writes the design intent, which the person then corrects. Intent is
meant to be author-written — the model should be called *after* it, not before,
and should never author it. Phase 9 reverses this: croquis becomes a plain
upload, intent is typed, and the sketch feeds the drafting call instead. Net
effect is one AI call per pattern where there are currently two, and it sits
after every human decision rather than ahead of one.

**2026-10-09:** `api/drafting/bodice.py` (Phase 2's output) was rebuilt
twice following the inthefolds.com bodice-block tutorial. First pass fixed
the armhole-bulge bug recorded in `docs/CASE_STUDY.md` §6 — direct
pivot/compass shoulder construction, two darts per piece,
`armscye_depth`/`shoulder_length` added to `Measurements`. Second pass,
prompted by the user's own measurements and reference images, re-fetched
the tutorial in full and implemented its actual seven-point armhole
construction instead of a single-bezier approximation; this surfaced and
fixed two more real bugs (a canceled-out clamp, and point A1 measured from
the wrong reference) that were only visible once rendered — not caught by
the test suite, consistent with this project's "render and look" rule.
`nape_to_bust` added to `Measurements`; `back_width` now read as a FULL
measurement, not pre-halved. Tested against the user's own real
measurements; visually re-checked via a rendered PDF; still not checked
against a muslin. See CASE_STUDY.md §6 for exactly what's faithful vs.
simplified vs. still flagged as an open judgment call.

Phase 4 (croquis interpretation) or Phase 5 (revision API) is next — PLAN.md
calls these independent once Phase 2 landed, so either order works.

---

## Phase 0 — Repo setup

**Goal:** the project runs and tests execute.

- `uv init`, add `fastapi`, `uvicorn`, `pytest`, `anthropic`, `svgwrite`,
  `reportlab`
- `.env` for `ANTHROPIC_API_KEY`, gitignored
- `api/db.py`: connection helper that executes `schema.DDL` on first run
- Confirm `tests/example_bodice.py` still runs from the repo root

**Done when:** `uv run pytest` passes with one trivial test and the example
script prints its diff.

---

## Phase 1 — Harden the schema

**Goal:** the data model is trustworthy before anything is built on it.

- Port dataclasses to Pydantic models (all at once — see CLAUDE.md)
- Add `infer_change_type(deltas) -> ChangeType` so revisions classify
  themselves: paths containing `.darts[` or dart leg points → `DART`,
  `seam_allowance_mm` → `SEAM_ALLOWANCE`, `ease_` → `EASE`, and so on
- Tests for `diff_patterns`: reordering a list produces no deltas; renaming a
  piece produces an add plus a remove; a nested bezier control point change is
  caught; an unchanged pattern diffs to an empty list

**Done when:** diff behaviour is covered by tests, including the reorder case.
That one matters most — it is the whole reason lists are keyed by id.

---

## Phase 2 — Drafting engine ⚠️ highest risk

**Goal:** `(Measurements, DesignIntent) -> list[PatternPiece]` for a basic
fitted bodice block, front and back.

**Decide first, before writing code:** which drafting system to follow. Aldrich
*Metric Pattern Cutting* is the usual reference and is already metric, which
matches the mm convention. Pick one and cite it in a module docstring. Do not
blend systems or work from half-remembered formulas.

- `api/drafting/bodice.py`, pure functions, no I/O
- Derive construction values from measurements and ease: back width, chest
  width, armhole depth, neck width and drop, shoulder slope, dart intake and
  placement
- Generate points in the order a human drafts them, naming each after the
  drafting step so revision paths stay readable
- Tests assert *relationships*, not magic numbers: the front armhole is longer
  than the back; total waist width equals waist measurement plus ease minus
  dart intakes; shoulder seam lengths match between front and back; the neckline
  is a smooth curve with no backtracking

**Done when:** a bodice generates from real measurements and the relationship
tests pass. Treat the output as a first draft, never as a fitted pattern.

**Risk:** this is where a model is most likely to produce confident nonsense.
Arithmetically valid geometry can still be unwearable. Compare generated values
against a published block for the same measurements before trusting it.

---

## Phase 3 — PDF export

**Goal:** a printable, correctly scaled, tiled PDF.

- `api/export/svg.py`: geometry → SVG. Offset the stitching line outward by
  `seam_allowance_mm` to produce the cut line; draw both. Render darts, notches,
  grainline, and a piece label with cut count and mirroring.
- `api/export/pdf.py`: tile the SVG across Letter or A4 pages with overlap
  margins, page-edge join markers, and page numbers
- **A 50mm calibration square on page 1 is mandatory.** Print at 100% scale and
  measure it with a ruler before trusting any pattern that comes out.
- Endpoint: `GET /patterns/{id}/versions/{v}/pdf`

**Done when:** a printed page's calibration square measures 50mm, and taped
tiles line up at the join markers.

**Note:** curve offsetting is genuinely fiddly. Offsetting a cubic bezier
exactly is not possible in closed form — flatten to a polyline, offset, then
refit, or just offset the flattened polyline and accept it.

---

## Phase 4 — Croquis interpretation

**Goal:** sketch upload → `DesignIntent`.

> **Superseded 2026-10-17 — see "Design intent is author-written" below.** The
> vision-proposes-intent design described in this section was built and tested,
> but it has the AI authoring design intent, which is explicitly not wanted.
> Phase 9 replaces it: the croquis is stored as a plain upload, intent is typed
> by a person, and the sketch becomes *input* to drafting rather than a source
> of decisions.

- `POST /croquis` accepts an image, stores it, returns a draft `DesignIntent`
- Vision call to the Anthropic API asking for **strict JSON matching the
  `DesignIntent` schema and nothing else** — no prose, no markdown fences
- Strip fences defensively anyway, validate against the schema, and on a parse
  failure return the raw text for manual correction rather than retrying blindly
- The returned intent is a **proposal you edit**, never applied automatically.
  Vision will misread seam lines and style details, and a wrong `garment_type`
  silently sends Phase 2 down the wrong drafting path.
- Don't re-collect body measurements on every croquis upload. Default a new
  pattern's `Measurements` from the most recent existing pattern (single user,
  these rarely change) and let the review screen confirm/adjust before
  creating version 1 — not a separate measurements endpoint. **Done:** the
  Phase 7 frontend reads it straight off `GET /patterns` (newest-first) +
  `GET /patterns/{id}`, a plain lookup against existing data — no RAG/`ask`
  call, which is for synthesizing answers over revision history, not for
  fetching a value that's just sitting in the most recent row. Measurement
  changes after that go through the normal `POST /patterns/{id}/revise` flow
  (Phase 5) like any other edit — `infer_change_type` already classifies
  `measurements.*` paths as `MEASUREMENT`, so this needs no new machinery.
- Creating the pattern from an approved intent writes version 1 with a
  `CROQUIS_DRAFT` revision whose reason records what the vision step concluded
- Before drafting (Phase 2) runs, check whether a reference sloper exists for
  this garment type (Phase 6). If one does, draft from its proven geometry
  instead of the generic formulas; if not, fall back to the formulas as
  today. Either way the result is still a first draft, not a fitted pattern.

**Done when:** a sketch yields a sensible editable intent, and approving it
produces a version 1 that exports to PDF.

---

## Phase 5 — Revision API

**Goal:** edits are recorded with your reasoning attached.

- `POST /patterns/{id}/revise` — takes a modified pattern plus `reason`,
  `fit_notes`, `source`; diffs against current, infers change type, writes a new
  snapshot and a revision row, bumps `current_version`
- `GET /patterns/{id}/revisions` — timeline, newest first
- `GET /patterns/{id}/diff?from=2&to=5` — aggregate deltas across versions
- Reject a revise call whose diff is empty, and one whose `reason` is blank —
  an unexplained revision is invisible to the whole RAG layer later
- `POST /patterns/{id}/versions/{v}/mark-sloper` — explicitly marks this
  version's bodice piece(s) as the reference block for this garment type.
  Deliberately a separate, human-initiated action, not something the system
  infers from "N clean revisions in a row" — whether a pattern fits is
  exactly the kind of judgment CLAUDE.md already says needs a human, not a
  model guessing from revision-count heuristics.

**Done when:** a round trip of load → edit → revise → read back shows the
correct deltas and reason, and marking a version as the reference sloper is
reflected in what Phase 4 drafts from next time.

---

## Phase 6 — Body profile, reference sloper, and the RAG layer

**Goal:** make later drafts better than earlier ones, specifically for this
body — not just a Q&A box over old revisions, though that's part of it.

**Why this is the core of the project, not a bolt-on:** an LLM drafting a
bodice has no embodied feedback and can be confidently, consistently wrong
(see CLAUDE.md's new "Why AI-generated drafting specifically needs this"
note — Phase 2's armhole passed every automated check while being thirty
times too deep). The fix isn't a better prompt each time; it's giving the
next draft real prior correction data for *this* body instead of starting
from the same generic formulas and the same blind spots every time.

**Body profile:**
- A persistent record, seeded from the full measurement set (the 42-point
  chart this project's measurements were scoped from — `Measurements`
  currently covers a curated subset; expand it as a real garment type
  actually needs a field, not all at once speculatively)
- Distinct from a single pattern's `Measurements` snapshot — the profile is
  "current best known measurements," a pattern version is "what was true
  when this version was drafted." Phase 4's existing behavior (default a new
  pattern's measurements from the most recent one, changes flow through
  `/revise`) already treats the most recent pattern as a de facto profile;
  this phase makes that explicit as its own entity rather than an implicit
  convention, so it stops drifting if patterns are deleted or a garment type
  with different relevant measurements is added.
- **Still unbuilt.** Phase 9.1 picks this up and gives it a larger role (the
  "body model" that feeds AI-mediated drafting), including the rule that it
  holds measured body facts only — fit corrections live in Phase 9.2's
  garment learnings, not here.

**Reference sloper:**
- A pattern version explicitly marked via Phase 5's `mark-sloper` endpoint —
  never inferred automatically from revision count or any other heuristic
- When Phase 4 drafts a new bodice-derived garment and a sloper exists for
  that garment type, draft from the sloper's actual fitted geometry
  (`PatternPiece` points/edges/darts) rather than re-deriving from
  `api/drafting/bodice.py`'s formulas — adapted for new design features, not
  regenerated from scratch
- No sloper yet (first bodice, or a garment type with no block relationship
  to one) → formulas remain the fallback, exactly as Phase 2 built them

**RAG layer** (needs Phase 5):
- `api/rag/embed.py`: embed `Revision.embedding_text()`, store float32 BLOBs.
  **Embeddings are local hashed bag-of-words, not a semantic embeddings API**
  (Anthropic has no embeddings endpoint, and Voyage/OpenAI were considered and
  rejected — see CLAUDE.md's "Why local embeddings, not a semantic embeddings
  API" for the reasoning, not just the conclusion). Tokenize, drop a small
  stoplist of the revision-text template's own boilerplate words, hash each
  token into one of N buckets, L2-normalize — no vocabulary table, no training
  step, no new dependency or API key. `revision_embeddings.model` records
  which method produced a given vector, so this is swappable later without
  guessing what's already stored.
- Called automatically at the end of `POST /patterns/{id}/revise`, not only as
  a separate backfill step — a revision that isn't embedded the moment it's
  written is invisible to retrieval until someone remembers to run the
  backfill, which defeats the point of "ask about what just happened."
- `api/rag/retrieve.py`: cosine over the pattern's revisions. Filter by
  `pattern_id` **first**, then rank — cross-pattern bleed is the obvious failure
  here. Support a date-range filter for "what changed between March and June"
- `api/rag/answer.py`: retrieve, then pass the full revision records — deltas
  included — to the LLM. The prompt must instruct it to quote measurements only
  from the structured deltas, and to say so plainly when the notes do not
  explain a change
- `POST /patterns/{id}/ask`, returning the answer plus the revisions it used so
  you can check it
- Backfill embeddings for revisions written before this phase (anything from
  Phase 5 usage prior to Phase 6 landing)

**What "auto-apply" means here, and what it doesn't:** the body profile and
sloper feed a new draft's *starting point* automatically, without asking
first — no extra confirmation step before generation. What doesn't change:
the output is still never reported as correct, only generated. A smarter
starting point is not a fit guarantee; the muslin check CLAUDE.md already
requires stays required every time.

**Done when:** a second bodice-derived garment, drafted after a sloper is
marked, visibly starts from the sloper's geometry rather than the generic
formulas — and "why did I change the bust dart?" still returns the real
reason and cites the revision it came from.

---

## Phase 7 — Frontend

**Goal:** usable without touching the API directly — but NOT a live embedded
feature on the portfolio site (see CLAUDE.md's intro: local-only). Lives at
`personal-profile/pattern-tracker-app/`, plain HTML/CSS/JS matching that
repo's actual stack, reusing its colour tokens and type pairing for visual
consistency. Reachable locally (`personal-profile`'s existing static server,
`python3 serve.py`, then open `pattern-tracker-app/index.html`) but never
linked from the public nav — the portfolio's sewing box / tin shelf item
still only links to the case-study write-up (`docs/CASE_STUDY.md`), not this
tool.

- Measurement form and croquis upload — done. Measurements come from the
  body model (9.1), and the five optional ones the bodice draft would
  otherwise estimate are offered with an inline explanation of what each
  decides (9.6a).
- Design intent — author-written, never AI-proposed (see "Design intent is
  author-written" above) — done
- "Create pattern" shows a pending state, since drafting now makes an API
  call — done
- The timeline tags each draft with whether fitting history was applied,
  whether it fell back to plain formulas, or whether measurements were
  estimated — done, so the quality variance 9.4's fallback introduces is
  visible at a glance rather than buried in the reason text
- Version timeline with reason and fit notes per entry — done
- Pattern viewer rendering the SVG, with an overlay mode showing two versions in
  different colours — done (backed by a new `GET /patterns/{id}/versions/{v}/outline`
  endpoint that reuses `api.export.geometry`'s tested curve-flattening, so the
  frontend only ever draws straight-line SVG)
- Ask box, showing cited revisions alongside the answer — done (blocked at
  runtime on a real `ANTHROPIC_API_KEY`, same as `POST /croquis`)
- PDF download per version — done

**Done when:** a pattern can be created, revised and queried without touching
the API directly. **Done** — verified end to end in a real browser: create
(from standard formulas, then from a marked sloper), revise via a point
editor, overlay/diff two versions, mark-sloper, PDF link. Also turned up and
fixed a real pre-existing concurrency bug in `api/db.py`'s `get_connection()`
(missing `check_same_thread=False` — FastAPI's generator dependency can open
and close a connection on two different threadpool threads).

---

## Phase 8 — Polish

- Backup: the SQLite file is the entire archive. Automate a copy somewhere.
- Export a pattern plus its full history as JSON, so the work outlives the app.
- Revision tags for muslin rounds, so "first fitting" groups several edits.

---

## Phase 9 — AI-mediated drafting and the learning loop

**Goal:** the flow as specified on 2026-10-17. Croquis and design choices go
in; the AI drafts against the block math, the body model, and what previous
garments taught us; a PDF comes out; the muslin's corrections flow back into
the body model and the learnings, so the next garment starts better than the
last one.

**What this actually changes.** Today drafting is deterministic — `POST
/patterns` runs `draft_bodice()` (or `draft_bodice_from_sloper()`) and makes
no LLM call at all. The AI's entire current role is reading a sketch (`POST
/croquis`) and explaining history in prose (`POST /{id}/ask`); neither touches
geometry. This phase puts a model in the drafting path for the first time,
which is the highest-risk change in the project. Read "Limitations" at the
bottom before building any of it — several of this project's own invariants
have to be renegotiated, not quietly broken.

### Design intent is author-written, not AI-written

**The model is called *after* design intent, never before it, and never writes
it.** Garment type, fabric class, fit, closures and style notes are typed by a
person. This reverses what Phase 4 built, where `POST /croquis` makes a vision
call that fills those fields in and the person corrects them afterwards.

Why the reversal is the right call, beyond it being the stated preference: an
AI-proposed intent that you then edit looks like a time-saver but quietly
changes who decided. A wrong `garment_type` sends drafting down the wrong path
entirely, and a pre-filled field is far less likely to be read critically than
an empty one — the proposal anchors the answer. Design intent is the one part
of this pipeline that is *pure authorial decision*: there's no body to measure
and no muslin to check it against, only what you meant to make. That is the
last thing worth automating.

What this changes concretely:

- `POST /croquis` becomes a **plain upload** — store the image, return
  `croquis_ref`. No vision call, no `intent`/`parse_error` in the response.
- The croquis is still *used* by the AI, just later and differently: the image
  goes into the Phase 9.4 drafting prompt alongside the author-written intent,
  where it conveys what the text fields can't — how deep the neckline actually
  sits, where a seam falls, how a sleeve is shaped. Seeing the sketch while
  proposing geometry is a different act from deciding what the garment is.
- The Phase 7 "Analyze sketch" button becomes a plain upload control, and the
  "intent review screen" stops being a review — it's just a form.
- Pattern creation stops requiring an API key for the *upload* step. Only the
  9.4 drafting call needs one, and it has a fallback. Useful side effect:
  attaching a croquis works offline again.
- `_DESIGN_INTENT_PROMPT`, `_strip_fences`, and the croquis intent-parsing
  tests are no longer needed for this path. Keep `_strip_fences` — 9.4's JSON
  response needs the same defensive treatment.

**Recommended shape: the AI proposes deltas, code applies them.** The model
does *not* emit a coordinate set. It receives a complete, already-valid draft
from the existing math plus the body model and the learnings, and returns a
bounded list of adjustments (a point id with dx/dy, or a named dart/ease
change) that deterministic code applies. Reasons:

- A delta can be range-checked. A regenerated coordinate set can only be
  re-validated wholesale, which is exactly how Phase 2's armhole passed every
  automated check while being thirty times too deep.
- It reuses `FieldDelta` / `diff_patterns` as they already exist, so an AI
  adjustment has the same shape as a manual one and shows up in the timeline,
  the diff view and the RAG index with no new machinery.
- The fallback is free: reject the proposal, ship the formula draft, record
  why.

The alternative — the model emits the whole `Pattern` JSON — is recorded here
as considered and rejected, for the reason in the first bullet.

### 9.1 Body model — completes Phase 6's unbuilt half

- `body_profiles` table: one row per profile (realistically one, single user),
  holding the full `Measurements` field set plus `updated_at`.
- `GET /body-profile` and `PUT /body-profile`. Seeded from the first pattern
  created, or entered directly.
- A revision whose deltas touch `measurements.*` updates the profile — this is
  the "adjustments update the body model" half of step 5. `infer_change_type`
  already classifies those paths as `MEASUREMENT`, so the hook is a filter on
  existing data, not new parsing.
- The Phase 7 frontend stops defaulting measurements from "the most recent
  pattern" and reads the profile instead. The current behaviour is a de facto
  profile; this makes it the real one, so it stops drifting when patterns are
  deleted.
- **Scope boundary:** the profile holds *measured body facts*. It does not
  hold fit corrections ("needs 4mm more back shoulder dart") — those are
  garment learnings, below. Mixing the two is what makes a profile
  un-auditable.

### 9.2 Garment learnings

- `garment_learnings` table: `garment_type`, the learning text, the
  `revision_id` it was derived from, a `confidence` or `times_confirmed`
  counter, `created_at`.
- Written when a revision is recorded **and** that revision is credible
  evidence — see the gating rule in 9.6. Not every edit is a lesson.
- Distinct from the revision log: a revision is "what changed on this pattern
  on this date"; a learning is "what this body/this garment type needs,
  generalised." One is a fact, the other is an inference — keep the row that
  generated it so the inference can always be traced back.

### 9.3 Garment-type-scoped retrieval

- `retrieve()` currently filters by `pattern_id` **first**, deliberately —
  cross-pattern bleed is called out in Phase 6 as the obvious failure. For
  learnings to transfer between garments, drafting needs a second scope:
  retrieve across patterns sharing a `garment_type`.
- Add that as an explicit parameter, used **only** by the drafting path.
  `POST /{id}/ask` keeps pattern-scoped retrieval — asking "why did I change
  this dart" must never answer from a different garment's history.
- `patterns` has no `garment_type` column today; it's inside the version
  snapshot JSON. Either denormalise it onto `patterns` or join through the
  current version's snapshot. Denormalising is simpler and it never changes
  after creation.

### 9.4 The AI drafting step

In `POST /patterns`, after the existing sloper-or-formula draft:

1. Build the prompt: the drafted block as structured points, the body model,
   the retrieved garment learnings, the author-written `DesignIntent`, and the
   croquis image itself. This is the *only* call in the creation path, and the
   only place the sketch is interpreted — it informs geometry, never intent.
2. Ask for a JSON list of proposed adjustments and a one-line reason each.
   Same strict-JSON + fence-stripping discipline `POST /croquis` already uses.
3. Validate (9.5). Apply the survivors deterministically.
4. Record the result as the version-1 `CROQUIS_DRAFT` revision, with the
   proposal and anything rejected in the reason — so the AI's contribution is
   visible in the timeline rather than baked invisibly into the geometry.

On any failure — no key, network error, unparseable JSON, failed validation —
**fall back to the formula draft and say so in the reason.** Drafting must not
become unavailable because an API call failed.

### 9.6a Ask for measurements rather than guess them (decided 2026-10-17)

Building 9.4 surfaced a drafting question: with only the seven required
measurements, the front armhole deepens relative to its chord as the body
grows (0.269 at bust 920, 0.374 at 1300), because the draft derives
`armscye_depth`, `back_width`, `neck_circumference` and `nape_to_bust` from
bust — and falls back to a **flat 12.5cm** for `shoulder_length`, the same
for every body, which feeds the pivot that places the shoulder tip and so
shapes the armhole.

**Decision: always use the measurements that are available, and ask for the
missing ones rather than tune a fallback formula.** Tuning the derivation
until the output looked plausible is precisely the "averaging remembered
numbers" CLAUDE.md warns against — it would make the draft look better
while still being a guess about a body nobody measured. A measurement costs
one pass with a tape; a guess costs a muslin.

- `api/drafting/needs.py` reports which fields the draft had to estimate,
  what each one actually decides, how to take it, and what it's being
  guessed as meanwhile. A fallback that doesn't scale with the body ranks
  first, since it's equally wrong for every body but one.
- Surfaced three ways: on the create response, in the version-1 revision
  reason (so "which numbers were real when this was drafted?" stays
  answerable six months on), and at `GET /body-profile/needs`.
- The frontend takes all five as optional inputs and shows the outstanding
  asks inline.
- Deliberately *not* done: changing any fallback formula. They stay as the
  documented defaults they always were; the tool now just says when it's
  leaning on one.

### 9.6b Garments are composed of blocks (decided 2026-10-17)

`GarmentType.SLEEVE` was wrong: a sleeve is a block you draft *as part of* a
garment, never the thing being made. Replaced with `SHIRT`, defined as a
bodice block plus a sleeve block.

- `api/drafting/needs.py` declares `Block` (bodice, sleeve, skirt, trouser),
  which measurements each block requires, and which blocks each garment is
  built from — shirt = bodice + sleeve, dress = bodice + skirt.
- A garment's required measurements are **derived** from its blocks, not
  listed per garment, so a change to what a bodice needs reaches everything
  built on one. Shared fields collapse: a dress needs waist once.
- `is_draftable` is "every constituent block is implemented," so garments
  become draftable automatically as blocks land rather than needing a list
  updated in two places.
- A refusal names the missing *block*, not just the garment: half a shirt
  already drafts, so "not supported" would hide what's actually outstanding.
- Removing an enum value is normally forbidden (snapshots carry old values
  forever, per CLAUDE.md). Safe only because nothing had been stored with
  `sleeve` — verified against `patterns`, `reference_slopers` and
  `pattern_versions` before the change.

### 9.5 The validator — the part that makes this safe

The relationship assertions in `tests/test_bodice.py` are currently only test
code. Promote them to a reusable `api/drafting/validate.py` that both the
tests and the drafting path call:

- armholes comparable front-to-back; armhole concave, not a bulge or a zigzag
- total waist width equals waist plus ease minus darts
- necklines monotonic, meeting the fold with a horizontal tangent
- dart apexes straddling the bust point; vertical bust line colinear
- plus a new bound specific to this phase: **no single AI adjustment may move
  a point more than N mm** (start at 15mm, tune from real use). An adjustment
  outside the bound is dropped, not clamped — a model asking for a 300mm move
  is wrong about something, and silently shrinking it hides that.

A proposal that fails validation is rejected wholesale, logged, and the
formula draft ships. Never partially apply a failed set.

### 9.6 The feedback loop, and what gates it

Step 5 of the described flow says adjustments update the body model and the
learnings. The body-model half (9.1) is unambiguous — a changed measurement is
a changed fact. The learnings half needs a gate, because **not every revision
is evidence**:

- A revision with `source=MUSLIN_FITTING` is a real fitting observation →
  eligible to become a learning.
- `MANUAL` and `AI_ADJUST` revisions are design changes or guesses → recorded
  in history, queryable via `/ask`, but they do **not** become learnings.
- A learning that appears again in a later fitting increments
  `times_confirmed`; the prompt orders by that, so a repeatedly-confirmed
  correction outranks a one-off.

Without this gate, a single speculative edit propagates into every future
garment with no muslin ever having validated it — which is the exact failure
CLAUDE.md's "things that need a human" section exists to prevent.

**Done when:** a second garment of the same type, drafted after a first one
has been fitted and its corrections recorded, visibly starts closer to the
fitted shape than the formulas alone would produce — *and* the reason on its
version 1 states which learnings were applied and which proposals were
rejected.

### Limitations and risks — read before building

1. **LLMs are bad at coordinate geometry.** This isn't speculative here;
   Phase 2's first armhole was thirty times too deep and passed every check it
   was given. The delta-not-coordinates design (above) and the 9.5 validator
   are what make this tolerable. Neither makes it *safe* — they make failures
   loud instead of silent.
2. **Drafting becomes non-deterministic — accepted, 2026-10-17.** Same
   croquis, same body, same learnings → possibly different patterns.
   **Decision: adaptation over reproducibility.** The reasoning: reproducing a
   draft exactly is only worth something if the draft was right, and this
   project's entire premise is that it isn't right until a muslin says so.
   "Re-run the draft and get the same formula answer back" was never the
   useful operation; "get a draft that starts closer to the last fitted one"
   is. What's given up is regenerating a version from its inputs. What makes
   that survivable is CLAUDE.md's existing invariant that every version is a
   full stored snapshot — a draft can't be re-derived, but it's never lost.
   One consequence to carry forward: Phase 8's JSON export stops being a
   convenience and becomes the only durable copy of a given draft.
3. **Drafting gains a hard dependency on network and a funded API key.**
   Today `POST /patterns` is pure local math — no key, no network,
   milliseconds, free. After 9.4 the good path needs a reachable Anthropic API
   and a key with credit on it (separate from a Claude subscription). Specifics
   worth planning for rather than discovering:
   - **One call per pattern**, at drafting time — the croquis vision call goes
     away with the author-written-intent change above, and the sketch rides
     into the drafting call instead. Still seconds rather than milliseconds,
     and the Phase 7 "Create pattern" button has no pending state today — it
     needs one, or the tool will look hung.
   - **Cost is genuinely negligible at this scale.** A few thousand input
     tokens per pattern, for one person sewing garments by hand. The one item
     on this list not worth optimising.
   - **The fallback hides quality variance.** Shipping the formula draft when
     the call fails keeps the tool working, but two patterns in the timeline
     can then look like peers when one had learnings applied and one didn't.
     Recording it in the revision reason is the minimum; the timeline should
     mark it visually or the difference goes unnoticed.
   - **Body measurements now leave the machine on every draft.** The drafting
     prompt carries the body model. Nothing is stored remotely and the repo
     stays local, but CLAUDE.md's own local-only argument names body
     measurements as the sensitive thing — so this is a deliberate change to
     what gets transmitted, not an incidental one. Local data *sovereignty* is
     kept; local *autonomy* is not.
   - **The test suite must never require a key.** 137 tests run offline in
     ~2.6s today. The drafting path gets the same `_FakeAnthropicClient`
     treatment the croquis tests already use, and the fallback needs its own
     test: API fails → formula draft ships → the reason says so.
4. **The purity invariant has to be scoped, not broken.** CLAUDE.md: "drafting
   functions are pure: (Measurements, DesignIntent) -> list[Point]. No I/O, no
   database, no API calls. This is what makes them testable." Keep that true —
   `draft_bodice()` stays pure, and the AI step lives in an orchestration
   layer above it in `main.py`. If the API call migrates inside the drafting
   module, the whole test suite's value goes with it.
5. **Automatic learning sits in tension with "only a muslin knows."** The 9.6
   gate is the compromise. It's a judgment call, not a solved problem: a
   fitting observation that was itself mistaken will still propagate.
6. **Retrieval is lexical, not semantic.** The local hashed bag-of-words
   embedder (see CLAUDE.md for why it was chosen) matches shared vocabulary.
   That's a good fit for "why did I widen this dart" in your own words; it's a
   weaker fit for "find every learning relevant to a garment I haven't made
   before." Expect misses, and keep `revision_embeddings.model` honest so the
   method can be swapped without guessing what's stored.
7. **The measurement-authority invariant still holds.** "Never let the RAG
   layer be the source of a measurement it reports." Learnings may carry
   *prose*; every number the drafting step acts on must come from the body
   model or a structured delta. A learning that says "add 4mm" must resolve
   that 4mm against a delta, not against the sentence.
8. **Cross-garment bleed is a new failure mode.** Phase 6 guarded against it
   by filtering on `pattern_id`. 9.3 deliberately opens that up for drafting.
   A dress learning reaching a skirt draft is now possible; `garment_type`
   scoping is the only thing preventing it.
9. **Marking a sloper stays manual.** This phase does not make the sloper
   automatic. "Whether a pattern fits" remains a human judgment — the AI now
   gets more context, but it still never gets to declare the result correct,
   only generated.

---

## Suggested session prompts

Phase 2: *"Read CLAUDE.md and docs/PLAN.md. Implement Phase 2 using Aldrich
Metric Pattern Cutting for the bodice block. Pure functions only. Write the
relationship tests first, then the drafting code."*

Phase 6: *"Read CLAUDE.md and docs/PLAN.md. Implement Phase 6 — the body
profile, the reference sloper, and the RAG layer. Filter by pattern_id
before ranking, and make the answer prompt read measurements from
structured deltas rather than from embedded text. The profile and sloper
feed Phase 4's drafting step automatically, without an extra confirmation
step, but the output is still never reported as correct, only generated."*

Phase 9: *"Read CLAUDE.md and docs/PLAN.md, including Phase 9's 'Limitations
and risks' in full. Build it in order: 9.1 body model, 9.5 validator, then
9.2/9.3 learnings and scoped retrieval, and only then 9.4's AI drafting step
— the validator must exist before anything lets a model touch geometry. The
model proposes bounded deltas, never coordinates; `draft_bodice()` stays
pure; every failure path falls back to the formula draft and records why."*

## Ordering notes

Phases 3, 4 and 5 are independent once Phase 2 lands, so take them in whatever
order is appealing. Phase 7 needs whatever it displays.

Phase 6's RAG layer needs Phase 5 (it embeds revisions). The body
profile/sloper half of Phase 6 and Phase 4 reference each other (Phase 4
checks for a sloper before drafting; the sloper only has something to mark
once Phase 4 and Phase 5 exist) but aren't a hard sequencing dependency —
build Phase 4 first with the sloper-check always falling back to the Phase 2
formulas, then Phase 6 makes that check real.

Build Phase 5 before accumulating real pattern work — revisions made before
logging exists are lost, and the history is the point of the project.
