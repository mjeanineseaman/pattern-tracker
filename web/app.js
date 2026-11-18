"use strict";
/* The local UI: plain fetch() calls to this project's own FastAPI backend
 * (local-only, see CLAUDE.md). No framework, no build step — the page is
 * served by that same app from web/, so it loads straight from disk with
 * nothing to compile. */

const LS_API_BASE = "pattern-tracker:api-base";

// Served by the API itself, so same-origin is the normal case and a
// relative base just works. The absolute fallback covers opening
// web/index.html directly off disk, where the origin is file:// and
// there's no server to be relative to.
const DEFAULT_API_BASE =
  window.location.protocol === "file:" ? "http://localhost:8000" : window.location.origin;

const state = {
  apiBase: localStorage.getItem(LS_API_BASE) || DEFAULT_API_BASE,
  patterns: [],
  currentPatternId: null,
  currentPattern: null,   // full Pattern dict for the version shown in the viewer
  revisions: [],
  garmentTypes: [],       // from GET /garment-types — what's draftable, and what each needs
  croquisRef: null,       // set once /croquis has stored & analyzed an image — required to create a pattern
};

const el = (id) => document.getElementById(id);

function apiUrl(path) {
  return state.apiBase.replace(/\/+$/, "") + path;
}

async function api(path, options) {
  const res = await fetch(apiUrl(path), options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch (_) { /* no json body */ }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.status === 204 ? null : res.json();
}

function showBanner(message) {
  const banner = el("banner");
  banner.textContent = message;
  banner.hidden = false;
}

function clearBanner() {
  el("banner").hidden = true;
}

function showView(name) {
  for (const view of document.querySelectorAll(".view")) {
    view.hidden = view.id !== `view-${name}`;
  }
}

// --------------------------------------------------------------------
// API base + connection status
// --------------------------------------------------------------------

async function checkConnection() {
  const dot = el("api-status");
  try {
    await api("/patterns");
    dot.classList.remove("down");
    dot.classList.add("ok");
    dot.title = "Connected";
  } catch (e) {
    dot.classList.remove("ok");
    dot.classList.add("down");
    dot.title = "Can't reach the API at this address — is uvicorn running?";
  }
}

el("api-base-input").value = state.apiBase;
el("api-base-input").addEventListener("change", async (e) => {
  state.apiBase = e.target.value.trim() || DEFAULT_API_BASE;
  localStorage.setItem(LS_API_BASE, state.apiBase);
  await checkConnection();
  await loadPatternList();
});

// --------------------------------------------------------------------
// Pattern list (sidebar)
// --------------------------------------------------------------------

async function loadPatternList() {
  try {
    state.patterns = await api("/patterns");
  } catch (e) {
    showBanner(`Couldn't load patterns: ${e.message}`);
    return;
  }
  const ul = el("pattern-list-items");
  ul.innerHTML = "";
  if (state.patterns.length === 0) {
    ul.innerHTML = `<li class="pattern-list__empty">No patterns yet.</li>`;
    return;
  }
  for (const p of state.patterns) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = "pattern-list__item" + (p.id === state.currentPatternId ? " active" : "");
    btn.dataset.patternId = p.id;  // the id itself is never shown — see openPattern()
    btn.innerHTML = `${escapeHtml(p.name)}<small>v${p.current_version}</small>`;
    btn.addEventListener("click", () => openPattern(p.id));
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// --------------------------------------------------------------------
// New pattern flow
// --------------------------------------------------------------------

el("new-pattern-btn").addEventListener("click", async () => {
  state.currentPatternId = null;
  resetNewPatternForm();
  showView("new");
  renderPatternListActive();
  await renderGarmentTypes();
  await prefillMeasurementsFromBodyProfile();
});
el("cancel-new-btn").addEventListener("click", () => showView("empty"));

function resetNewPatternForm() {
  state.croquisRef = null;
  el("croquis-file").value = "";
  el("croquis-status").textContent = "";
  el("create-pattern-btn").disabled = true;
}

const MEASUREMENT_FIELDS = {
  bust: "m-bust", waist: "m-waist", hip: "m-hip", shoulder_width: "m-shoulder-width",
  back_length: "m-back-length", bust_height: "m-bust-height", bust_span: "m-bust-span",
  // Optional, but the draft estimates each one when it's missing — see the
  // "measurements being estimated" panel below.
  shoulder_length: "m-shoulder-length", armscye_depth: "m-armscye-depth",
  back_width: "m-back-width", nape_to_bust: "m-nape-to-bust",
  neck_circumference: "m-neck-circumference",
};

// CLAUDE.md: "All lengths are millimetres, everywhere... convert to inches
// or centimetres only at the UI and PDF edges." This form is that edge — it
// shows and accepts centimetres, and nothing leaves these two functions in
// anything but mm.
//
// Both round to 0.1mm, which makes the conversion exactly reversible: a
// stored 863.6mm displays as 86.36cm and comes back as 863.6mm. Rounding
// the display to a single decimal instead would quietly rewrite the body
// profile by a fraction of a millimetre every time the form was opened.
function mmToCm(mm) {
  return Math.round(mm * 10) / 100;
}

function cmToMm(cm) {
  return Math.round(cm * 100) / 10;
}

// Phase 9.1: measurements come from the body model — the current best known
// numbers for this body — rather than from whichever pattern happened to be
// created last. A plain local lookup; no RAG/LLM round trip.
async function prefillMeasurementsFromBodyProfile() {
  const hint = el("measurements-source-hint");
  try {
    const profile = await api("/body-profile");
    for (const [field, inputId] of Object.entries(MEASUREMENT_FIELDS)) {
      el(inputId).value = profile[field] != null ? mmToCm(profile[field]) : "";
    }
    hint.textContent =
      "Filled in from your body profile — edit any of them and the profile updates when you create the pattern.";
  } catch (e) {
    // 404 means nothing's recorded yet. The fields stay empty rather than
    // showing invented numbers — a measurement you didn't take shouldn't
    // arrive pre-filled and get submitted as if you had.
    for (const inputId of Object.values(MEASUREMENT_FIELDS)) el(inputId).value = "";
    hint.textContent = "No measurements recorded yet — what you enter here becomes your body profile.";
  }
  await renderMeasurementNeeds();
}

// "Always use the measurements that are available, and ask for more rather
// than guess." Shows which values the draft would otherwise estimate, worst
// guess first, with how to take each one.
async function renderMeasurementNeeds() {
  const panel = el("measurement-needs");
  const measurements = readMeasurements();
  const missing = Object.entries(MEASUREMENT_FIELDS)
    .filter(([field]) => measurements[field] == null)
    .map(([field]) => field);

  if (missing.length === 0) {
    panel.hidden = true;
    return;
  }
  try {
    const { needs } = await api("/body-profile/needs");
    const relevant = needs.filter((n) => missing.includes(n.field));
    if (relevant.length === 0) {
      panel.hidden = true;
      return;
    }
    panel.innerHTML = `
      <p class="needs__lead">These are being <strong>estimated, not measured</strong>.
      Filling any of them in makes the draft fit closer — and the estimate for the
      first one is the same for every body, whatever its size.</p>
      <ul class="needs__list">${relevant.map((n) => `
        <li>
          <span class="needs__field">${escapeHtml(n.field.replace(/_/g, " "))}</span>
          <span class="needs__affects">sets ${escapeHtml(n.affects)}</span>
          <span class="needs__how">Measure: ${escapeHtml(n.how_to_measure)}</span>
          <span class="needs__estimate">Currently guessed as ${escapeHtml(n.estimated_as)}</span>
        </li>`).join("")}
      </ul>`;
    panel.hidden = false;
  } catch (e) {
    // No profile recorded yet — the per-pattern response still reports needs.
    panel.hidden = true;
  }
}

// What the tool can actually draft, reported by the API rather than
// hardcoded here, so the dropdown and the backend can't drift apart.
async function renderGarmentTypes() {
  const select = el("intent-garment-type");
  let types;
  try {
    types = await api("/garment-types");
  } catch (e) {
    return;
  }
  state.garmentTypes = types;
  const previous = select.value;
  select.innerHTML = types.map((t) => `
    <option value="${escapeHtml(t.garment_type)}"${t.draftable ? "" : " disabled"}>
      ${escapeHtml(t.garment_type)}${t.draftable ? "" : " — not drafted yet"}
    </option>`).join("");
  select.value = previous && types.some((t) => t.garment_type === previous && t.draftable)
    ? previous
    : (types.find((t) => t.draftable) || {}).garment_type || "";
  renderGarmentTypeNote();
}

function renderGarmentTypeNote() {
  const note = el("garment-type-note");
  const chosen = (state.garmentTypes || []).find(
    (t) => t.garment_type === el("intent-garment-type").value,
  );
  if (!chosen) {
    note.textContent = "";
    return;
  }
  if (chosen.draftable) {
    note.textContent =
      "A bodice block with no sleeve — a sleeveless top. That's what this tool drafts today.";
    return;
  }
  // Which block is missing matters: half a shirt (the bodice) already
  // drafts, so "not supported" would hide what's actually outstanding.
  const parts = [];
  if (chosen.blocks && chosen.blocks.length > 1) {
    parts.push(`Made from the ${chosen.blocks.join(" + ")} blocks.`);
  }
  if (chosen.missing_blocks && chosen.missing_blocks.length) {
    parts.push(`The ${chosen.missing_blocks.join(" and ")} block isn't drafted yet.`);
  } else {
    parts.push("Not drafted yet.");
  }
  if (chosen.missing.length) {
    parts.push(`It'll also need measuring: ${chosen.missing.join(", ")}.`);
  }
  note.textContent = parts.join(" ");
}

el("intent-garment-type").addEventListener("change", renderGarmentTypeNote);

// Returns millimetres. The inputs are centimetres; this is where that stops
// being true, so nothing downstream has to know the form used other units.
function readMeasurements() {
  const out = {};
  for (const [field, inputId] of Object.entries(MEASUREMENT_FIELDS)) {
    const raw = el(inputId).value.trim();
    const cm = raw === "" ? null : parseFloat(raw);
    out[field] = cm == null || Number.isNaN(cm) ? null : cmToMm(cm);
  }
  return out;
}

function renderPatternListActive() {
  for (const btn of document.querySelectorAll(".pattern-list__item")) {
    btn.classList.remove("active");
  }
}

// Upload only. The sketch is never read for design intent — you write that
// yourself below. It reaches a model later, during drafting, where it
// informs geometry rather than deciding what the garment is.
el("attach-croquis-btn").addEventListener("click", async () => {
  const fileInput = el("croquis-file");
  const status = el("croquis-status");
  if (!fileInput.files[0]) {
    status.textContent = "Choose an image first.";
    return;
  }
  status.textContent = "Uploading...";
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  try {
    const res = await fetch(apiUrl("/croquis"), { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);

    state.croquisRef = body.croquis_ref;
    el("create-pattern-btn").disabled = false;
    status.textContent = "Sketch attached.";
  } catch (e) {
    status.textContent = `Upload failed: ${e.message}`;
  }
});

el("create-pattern-btn").addEventListener("click", async () => {
  clearBanner();
  if (!state.croquisRef) {
    showBanner("Attach a croquis sketch first — a pattern can't be created without one.");
    return;
  }
  const closures = el("intent-closures").value.split(",").map((s) => s.trim()).filter(Boolean);
  const measurements = readMeasurements();
  // Optional measurements left blank are omitted entirely rather than sent
  // as null, so the API sees "not measured" rather than "measured as nothing".
  for (const [field, value] of Object.entries(measurements)) {
    if (value == null || Number.isNaN(value)) delete measurements[field];
  }
  const body = {
    name: el("pattern-name").value.trim() || "Untitled pattern",
    intent: {
      garment_type: el("intent-garment-type").value,
      fabric_class: el("intent-fabric-class").value,
      fit: el("intent-fit").value,
      closures,
      style_notes: el("intent-style-notes").value,
    },
    measurements,
    croquis_ref: state.croquisRef,
  };

  // Drafting now makes an API call (Phase 9.4), so this takes seconds
  // rather than milliseconds — without a pending state the tool looks hung.
  const btn = el("create-pattern-btn");
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Drafting…";
  try {
    const pattern = await api("/patterns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await loadPatternList();
    await openPattern(pattern.id);
    const notes = [];

    // A cleared measurement is called out separately from an edited one:
    // losing a value you really took sends the draft back to estimating
    // it, which is a quiet downgrade in fit rather than a correction.
    const changes = pattern.body_profile_updated || [];
    const cleared = changes.filter((c) => c.to === null);
    const edited = changes.filter((c) => c.to !== null);
    if (edited.length > 0) {
      notes.push(
        `Body profile updated: ${edited.map((c) => c.field.replace(/_/g, " ")).join(", ")}.`,
      );
    }
    if (cleared.length > 0) {
      notes.push(
        `Cleared from your body profile: ${cleared.map((c) => c.field.replace(/_/g, " ")).join(", ")} ` +
        `— the draft will estimate ${cleared.length > 1 ? "these" : "this"} until measured again.`,
      );
    }

    if (pattern.measurement_needs && pattern.measurement_needs.length > 0) {
      const fields = pattern.measurement_needs.map((n) => n.field.replace(/_/g, " "));
      notes.push(
        `${fields.length} measurement${fields.length > 1 ? "s were" : " was"} ` +
        `estimated rather than measured: ${fields.join(", ")}.`,
      );
    }
    if (notes.length > 0) showBanner(notes.join(" "));
  } catch (e) {
    showBanner(`Couldn't create pattern: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
});

// --------------------------------------------------------------------
// Pattern detail view
// --------------------------------------------------------------------

async function openPattern(patternId) {
  clearBanner();
  state.currentPatternId = patternId;
  showView("pattern");
  renderPatternListActive();
  for (const btn of document.querySelectorAll(".pattern-list__item")) {
    if (btn.dataset.patternId === patternId) btn.classList.add("active");
  }

  let pattern, revisions;
  try {
    [pattern, revisions] = await Promise.all([
      api(`/patterns/${patternId}`),
      api(`/patterns/${patternId}/revisions`),
    ]);
  } catch (e) {
    showBanner(`Couldn't load pattern: ${e.message}`);
    return;
  }
  state.currentPattern = pattern;
  state.revisions = revisions;

  el("pattern-title").textContent = pattern.name;
  el("pattern-meta").textContent =
    `Version ${pattern.version} — ${pattern.intent.garment_type}, ${pattern.intent.fabric_class}`;

  el("overlay-toggle").checked = false;
  el("overlay-version-wrap").hidden = true;
  el("viewer-legend").hidden = true;
  populateVersionSelects(pattern.version);
  await renderViewer();
  renderTimeline();
  renderRevisePoints(pattern);
  el("revise-form").hidden = true;
  el("ask-answer").hidden = true;
  el("ask-question").value = "";
}

function populateVersionSelects(currentVersion) {
  const versions = [];
  for (let v = currentVersion; v >= 1; v--) versions.push(v);

  for (const selectId of ["viewer-version", "overlay-version"]) {
    const select = el(selectId);
    select.innerHTML = "";
    for (const v of versions) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = `v${v}`;
      select.appendChild(opt);
    }
  }
  el("overlay-version").value = versions.length > 1 ? versions[1] : versions[0];
}

el("viewer-version").addEventListener("change", renderViewer);
el("overlay-version").addEventListener("change", renderViewer);
el("overlay-toggle").addEventListener("change", (e) => {
  el("overlay-version-wrap").hidden = !e.target.checked;
  el("viewer-legend").hidden = !e.target.checked;
  renderViewer();
});

el("mark-sloper-btn").addEventListener("click", async () => {
  try {
    const v = state.currentPattern.version;
    const result = await api(`/patterns/${state.currentPatternId}/versions/${v}/mark-sloper`, { method: "POST" });
    showBanner(`v${result.version} marked as the reference sloper for ${result.garment_type}.`);
  } catch (e) {
    showBanner(`Couldn't mark sloper: ${e.message}`);
  }
});

el("pdf-link").addEventListener("click", (e) => {
  const v = el("viewer-version").value;
  e.target.href = apiUrl(`/patterns/${state.currentPatternId}/versions/${v}/pdf`);
});

// ---- SVG viewer, with optional two-version overlay -------------------

async function renderViewer() {
  const wrap = el("viewer-svg-wrap");
  const diffList = el("diff-list");
  const v = el("viewer-version").value;
  el("pdf-link").href = apiUrl(`/patterns/${state.currentPatternId}/versions/${v}/pdf`);

  const overlayOn = el("overlay-toggle").checked;
  try {
    if (!overlayOn) {
      const outline = await api(`/patterns/${state.currentPatternId}/versions/${v}/outline`);
      wrap.innerHTML = buildOutlineSvg([{ outline, color: "var(--ink)" }]);
      diffList.hidden = true;
    } else {
      const vb = el("overlay-version").value;
      const [outlineA, outlineB, diff] = await Promise.all([
        api(`/patterns/${state.currentPatternId}/versions/${v}/outline`),
        api(`/patterns/${state.currentPatternId}/versions/${vb}/outline`),
        api(`/patterns/${state.currentPatternId}/diff?from=${vb}&to=${v}`),
      ]);
      wrap.innerHTML = buildOutlineSvg([
        { outline: outlineB, color: "var(--indigo)" },
        { outline: outlineA, color: "var(--rust)" },
      ]);
      renderDiffList(diff);
      diffList.hidden = false;
    }
  } catch (e) {
    wrap.innerHTML = `<p class="hint">Couldn't render: ${escapeHtml(e.message)}</p>`;
  }
}

function renderDiffList(diff) {
  const diffList = el("diff-list");
  if (diff.deltas.length === 0) {
    diffList.innerHTML = `<p class="hint">v${diff.from} and v${diff.to} are identical.</p>`;
    return;
  }
  const items = diff.deltas.map((d) =>
    `<li><code>${escapeHtml(d.path)}</code>: ${escapeHtml(JSON.stringify(d.old))} → ${escapeHtml(JSON.stringify(d.new))}</li>`,
  ).join("");
  diffList.innerHTML = `<p class="hint">Changes from v${diff.from} to v${diff.to}:</p><ul>${items}</ul>`;
}

// Every piece comes back from the API in its own drafting-local space —
// api/drafting/bodice.py anchors each piece's own origin point (cb_neck,
// for the back) at (0, 0) regardless of measurements, the same way every
// pattern piece does on paper. Drawn raw, bodice_front and bodice_back
// would sit on top of each other. api/export/pdf.py already solves this
// for print (_layout_pieces, left-to-right with a 20mm gap); this mirrors
// that same idea for the on-screen viewer, since the problem and the fix
// are identical — it's placement, not geometry, so no curve math is
// duplicated here, matching why /outline exists in the first place.
//
// For an overlay, the two versions of ONE piece must share the exact same
// offset rather than each being independently centred — centring each
// layer on its own bounding box would erase real differences (a dart
// that grew would re-shift the whole piece to compensate) instead of
// showing them. Because every piece already shares one fixed anatomical
// origin across versions, one offset per piece id, reused for every
// layer, keeps that alignment true: same piece overlays for comparison,
// different pieces sit side by side.
function layoutPieces(layers) {
  const pieceIds = [];
  const seen = new Set();
  for (const { outline } of layers) {
    for (const piece of outline.pieces) {
      if (!seen.has(piece.id)) { seen.add(piece.id); pieceIds.push(piece.id); }
    }
  }

  const margin = 20;
  const gap = 20; // matches api/export/pdf.py's _PIECE_GAP_MM
  const offsets = {};
  let cursorX = margin;
  let maxBottom = 0;

  for (const pieceId of pieceIds) {
    // Union bounding box across every layer that has this piece, so the
    // slot fits whichever version is larger.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const { outline } of layers) {
      const piece = outline.pieces.find((p) => p.id === pieceId);
      if (!piece) continue;
      for (const [x, y] of piece.cut) {
        minX = Math.min(minX, x); minY = Math.min(minY, y);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
      }
    }
    const dx = cursorX - minX, dy = margin - minY;
    offsets[pieceId] = { dx, dy };
    cursorX += (maxX - minX) + gap;
    maxBottom = Math.max(maxBottom, maxY + dy);
  }

  return { offsets, width: cursorX - gap + margin, height: maxBottom + margin };
}

function buildOutlineSvg(layers) {
  const hasGeometry = layers.some(({ outline }) => outline.pieces.length > 0);
  if (!hasGeometry) return `<p class="hint">No geometry to show.</p>`;

  const { offsets, width, height } = layoutPieces(layers);
  const shiftPt = ([x, y], off) => [(x + off.dx).toFixed(1), (y + off.dy).toFixed(1)];
  const shiftPoly = (poly, off) => poly.map((p) => shiftPt(p, off).join(",")).join(" ");
  const shiftLine = (a, b, off, attrs) => {
    const [x1, y1] = shiftPt(a, off), [x2, y2] = shiftPt(b, off);
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
  };

  let body = "";
  for (const { outline, color } of layers) {
    for (const piece of outline.pieces) {
      const off = offsets[piece.id];
      body += `<polygon points="${shiftPoly(piece.cut, off)}" fill="none" stroke="${color}" stroke-width="1" stroke-dasharray="4,2" opacity="0.6"/>`;
      body += `<polygon points="${shiftPoly(piece.stitching, off)}" fill="none" stroke="${color}" stroke-width="1.5"/>`;
      for (const [a, b] of piece.darts) {
        body += shiftLine(a, b, off, `stroke="${color}" stroke-width="1" stroke-dasharray="2,1"`);
      }
      if (piece.grainline) {
        const [g0, g1] = piece.grainline;
        body += shiftLine(g0, g1, off, `stroke="${color}" stroke-width="1" opacity="0.7"`);
      }
    }
  }
  const maxWidthPx = 760;
  const displayWidth = Math.min(maxWidthPx, width);
  const displayHeight = height * (displayWidth / width);
  return `<svg viewBox="0 0 ${width.toFixed(1)} ${height.toFixed(1)}" width="${displayWidth.toFixed(0)}" height="${displayHeight.toFixed(0)}">${body}</svg>`;
}

// ---- timeline ----------------------------------------------------------

function renderTimeline() {
  const ol = el("revision-timeline");
  ol.innerHTML = "";
  if (state.revisions.length === 0) {
    ol.innerHTML = `<li class="timeline__empty">No revisions yet — this is still version 1.</li>`;
    return;
  }
  for (const rev of state.revisions) {
    const li = document.createElement("li");
    li.className = "timeline__entry";
    li.innerHTML = `
      <div class="timeline__entry-head">
        <span class="timeline__version">v${rev.parent_version ?? "—"} → v${rev.version}</span>
        <span class="timeline__change-type">${escapeHtml(rev.change_type)} · ${escapeHtml(rev.source)}</span>
        ${draftOriginTag(rev)}
        <span class="timeline__date">${escapeHtml(rev.created_at)}</span>
      </div>
      ${rev.reason ? `<p class="timeline__reason">${escapeHtml(rev.reason)}</p>` : ""}
      ${rev.fit_notes ? `<p class="timeline__fit-notes">${escapeHtml(rev.fit_notes)}</p>` : ""}
    `;
    ol.appendChild(li);
  }
}

// Without this, a draft that got the fitting history applied and one that
// fell back to plain formulas (API down, proposal rejected) look like peers
// in the timeline — the difference only visible to someone reading the full
// reason text. It's exactly the quality variance worth seeing at a glance.
function draftOriginTag(rev) {
  const reason = rev.reason || "";
  if (reason.includes("Adjusted from fitting history")) {
    return `<span class="timeline__tag timeline__tag--adjusted"
      title="Previous fittings were applied to this draft">fitting history applied</span>`;
  }
  if (reason.includes("Drafted from the formulas —")) {
    return `<span class="timeline__tag timeline__tag--fallback"
      title="The adjustment step didn't run; this is the plain formula draft">formulas only</span>`;
  }
  if (reason.includes("Estimated (not measured)")) {
    return `<span class="timeline__tag timeline__tag--estimated"
      title="Some measurements were estimated rather than measured">estimated measurements</span>`;
  }
  return "";
}

// ---- revise --------------------------------------------------------

el("revise-toggle-btn").addEventListener("click", () => {
  const form = el("revise-form");
  form.hidden = !form.hidden;
});

function renderRevisePoints(pattern) {
  const container = el("revise-points");
  container.innerHTML = "";
  for (const piece of pattern.pieces) {
    const block = document.createElement("div");
    block.className = "revise-piece";
    block.dataset.pieceId = piece.id;
    const pointsHtml = piece.points.map((p) => `
      <span class="revise-point" data-point-id="${escapeHtml(p.id)}">
        <span>${escapeHtml(p.id)}</span>
        x<input type="number" step="any" value="${p.x}" data-axis="x">
        y<input type="number" step="any" value="${p.y}" data-axis="y">
      </span>
    `).join("");
    block.innerHTML = `<p class="revise-piece__name">${escapeHtml(piece.name)}</p>
      <div class="revise-piece__points">${pointsHtml}</div>`;
    container.appendChild(block);
  }
}

el("submit-revision-btn").addEventListener("click", async () => {
  clearBanner();
  const reason = el("revise-reason").value.trim();
  if (!reason) {
    showBanner("A reason is required to submit a revision.");
    return;
  }

  // Deep-copy the currently-loaded pattern and overwrite each point's x/y
  // from the form — everything else (edges, darts, notches...) stays as
  // it was, since they're defined in terms of point ids, not coordinates.
  const edited = JSON.parse(JSON.stringify(state.currentPattern));
  for (const pieceEl of document.querySelectorAll(".revise-piece")) {
    const piece = edited.pieces.find((p) => p.id === pieceEl.dataset.pieceId);
    for (const pointEl of pieceEl.querySelectorAll(".revise-point")) {
      const point = piece.points.find((p) => p.id === pointEl.dataset.pointId);
      point.x = parseFloat(pointEl.querySelector('[data-axis="x"]').value);
      point.y = parseFloat(pointEl.querySelector('[data-axis="y"]').value);
    }
  }

  try {
    await api(`/patterns/${state.currentPatternId}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pattern: edited,
        reason,
        fit_notes: el("revise-fit-notes").value.trim(),
        source: el("revise-source").value,
      }),
    });
    el("revise-reason").value = "";
    el("revise-fit-notes").value = "";
    el("revise-form").hidden = true;
    await loadPatternList();
    await openPattern(state.currentPatternId);
  } catch (e) {
    showBanner(`Couldn't submit revision: ${e.message}`);
  }
});

// ---- ask box --------------------------------------------------------

el("ask-btn").addEventListener("click", async () => {
  const question = el("ask-question").value.trim();
  if (!question) return;
  const answerBox = el("ask-answer");
  answerBox.hidden = false;
  answerBox.innerHTML = `<p class="hint">Thinking...</p>`;
  try {
    const result = await api(`/patterns/${state.currentPatternId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const used = result.revisions_used.map((r) =>
      `<li>v${r.parent_version ?? "—"} → v${r.version}: ${escapeHtml(r.reason || "(no reason recorded)")}</li>`,
    ).join("");
    answerBox.innerHTML = `
      <p class="ask-answer__text">${escapeHtml(result.answer)}</p>
      ${result.revisions_used.length > 0
        ? `<p class="ask-answer__used">Cited revisions:</p><ul class="ask-answer__used">${used}</ul>`
        : ""}
    `;
  } catch (e) {
    answerBox.innerHTML = `<p class="hint">Couldn't get an answer: ${escapeHtml(e.message)}</p>`;
  }
});

// --------------------------------------------------------------------

(async function init() {
  await checkConnection();
  await loadPatternList();
})();
