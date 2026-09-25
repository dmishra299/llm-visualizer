/* Shared formatting + visualization primitives (single implementation used everywhere). */
import { statsOf } from "./engine.js";

export function fmt(v, precision = 4) {
  if (!isFinite(v)) return v < 0 ? "-∞" : "+∞";
  if (v <= -1e8) return "-∞"; // masked sentinel
  const p = Math.pow(10, precision);
  const r = Math.round(v * p) / p;
  return (Object.is(r, -0) ? 0 : r).toFixed(precision);
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

export function shapeBadge(shapeArr) {
  return `<span class="shape-badge">[${shapeArr.join(", ")}]</span>`;
}

function cellBg(v, maxAbs) {
  if (v <= -1e8 || !isFinite(v)) return "background:#f1f3f7;";
  if (maxAbs < 1e-12) return "";
  const a = 0.06 + 0.55 * Math.min(1, Math.abs(v) / maxAbs);
  return v >= 0
    ? `background:rgba(36,86,214,${a.toFixed(3)});color:${a > 0.42 ? "#fff" : "inherit"};`
    : `background:rgba(194,72,31,${a.toFixed(3)});color:${a > 0.42 ? "#fff" : "inherit"};`;
}

/** Reusable matrix/heatmap table. M is 2D array. */
export function matrixHTML(M, opts = {}) {
  const {
    rowLabels = null, colLabels = null, precision = 4,
    hlRow = -1, hlCol = -1, maxRows = 12, maxCols = 16,
    caption = null, compact = false,
  } = opts;
  if (!M || !M.length) return `<p class="stage-sub">No data — enter text to run the computation.</p>`;
  const R = M.length, C = M[0].length;
  const rN = Math.min(R, maxRows), cN = Math.min(C, maxCols);
  let maxAbs = 1e-12;
  for (let i = 0; i < rN; i++) for (let j = 0; j < cN; j++) {
    const v = M[i][j];
    if (isFinite(v) && v > -1e8) maxAbs = Math.max(maxAbs, Math.abs(v));
  }
  let h = `<div class="matrix-wrap"${caption ? ` aria-label="${esc(caption)}"` : ""}><table class="matrix">`;
  if (colLabels) {
    h += `<tr><th class="rowlab"></th>`;
    for (let j = 0; j < cN; j++) h += `<th>${esc(colLabels[j])}</th>`;
    if (C > cN) h += `<th>…</th>`;
    h += `</tr>`;
  }
  for (let i = 0; i < rN; i++) {
    h += `<tr>`;
    if (rowLabels) h += `<td class="rowlab" title="${esc(rowLabels[i])}">${esc(rowLabels[i])}</td>`;
    for (let j = 0; j < cN; j++) {
      const v = M[i][j];
      const cls = [
        i === hlRow ? "hl-row" : "",
        j === hlCol ? "hl-col" : "",
        v <= -1e8 ? "masked" : "",
      ].join(" ");
      const shown = compact && Math.abs(v) < 0.00005 && v !== 0 ? "0" : fmt(v, precision);
      h += `<td class="${cls}" style="${cellBg(v, maxAbs)}" title="row ${i}, col ${j} = ${fmt(v, 6)}">${shown}</td>`;
    }
    if (C > cN) h += `<td>…</td>`;
    h += `</tr>`;
  }
  if (R > rN) h += `<tr><td class="rowlab">…</td><td colspan="${cN}">+${R - rN} more rows</td></tr>`;
  h += `</table></div>`;
  return h;
}

/** Horizontal bar list for logits / probabilities. items: [{label, value, kept?, selected?, extra?}] */
export function barsHTML(items, opts = {}) {
  const { precision = 4, maxAbs = null, colorByKept = false } = opts;
  const mx = maxAbs ?? Math.max(1e-9, ...items.map((d) => Math.abs(d.value)));
  return `<div class="bars">` + items.map((d) => {
    const w = Math.max(1.5, (Math.abs(d.value) / mx) * 100);
    const cls = ["bar-row", d.kept === false ? "cut" : "kept", d.selected ? "selected" : ""].join(" ");
    return `<div class="${cls}" title="${esc(d.label)} = ${fmt(d.value, 6)}${d.extra ? " · " + esc(d.extra) : ""}">
      <span class="tok">${esc(d.label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${w.toFixed(1)}%"></span></span>
      <span class="num">${fmt(d.value, precision)}${colorByKept && d.kept === false ? " ✕" : ""}</span>
    </div>`;
  }).join("") + `</div>`;
}

export function tokenChips(tokens, ids, opts = {}) {
  const { newIndex = -1, selected = -1, clickable = false, action = "selQuery" } = opts;
  return tokens.map((t, i) => {
    const cls = ["token-chip", i === newIndex ? "new" : "", i === selected ? "qsel" : "", clickable ? "clickable" : ""].join(" ");
    const attr = clickable ? ` data-action="${action}" data-i="${i}" role="button" tabindex="0" title="Select token ${esc(t)}"` : ` title="token ${esc(t)} → id ${ids[i]}"`;
    return `<span class="${cls}"${attr}>${esc(t)}<span class="id">id ${ids[i]}</span></span>`;
  }).join("");
}

export function legendHTML() {
  return `<div class="legend" aria-hidden="true">
    <span><span class="sw" style="background:rgba(36,86,214,.55)"></span>positive</span>
    <span><span class="sw" style="background:rgba(194,72,31,.55)"></span>negative</span>
    <span>deeper = larger magnitude · every cell also shows its number</span>
  </div>`;
}

export function statsGrid(name, M) {
  const s = statsOf(M);
  const sh = `[${M.length}, ${M[0]?.length ?? 0}]`;
  return `<div class="insp-grid">
    <div class="insp-stat"><div class="k">Tensor</div><div class="v">${esc(name)}</div></div>
    <div class="insp-stat"><div class="k">Shape</div><div class="v">${sh}</div></div>
    <div class="insp-stat"><div class="k">dtype</div><div class="v">float64</div></div>
    <div class="insp-stat"><div class="k">Min</div><div class="v">${fmt(s.min)}</div></div>
    <div class="insp-stat"><div class="k">Max</div><div class="v">${fmt(s.max)}</div></div>
    <div class="insp-stat"><div class="k">Mean</div><div class="v">${fmt(s.mean)}</div></div>
  </div>`;
}

/** GELU plot as inline SVG with hover readout. */
export function geluSVG(hlX = null) {
  const W = 460, H = 200, P = 28;
  const xs = [], minX = -3, maxX = 3;
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  for (let i = 0; i <= 120; i++) xs.push(minX + ((maxX - minX) * i) / 120);
  const ys = xs.map(gelu);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const X = (x) => P + ((x - minX) / (maxX - minX)) * (W - 2 * P);
  const Y = (y) => H - P - ((y - minY) / (maxY - minY)) * (H - 2 * P);
  const path = xs.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join(" ");
  const zeroY = Y(0), zeroX = X(0);
  return `<svg class="gelu-plot" id="geluPlot" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="GELU activation curve">
    <line x1="${P}" y1="${zeroY}" x2="${W - P}" y2="${zeroY}" stroke="#c9d2e0"/>
    <line x1="${zeroX}" y1="${P}" x2="${zeroX}" y2="${H - P}" stroke="#c9d2e0"/>
    <path d="${path}" fill="none" stroke="#2456d6" stroke-width="2"/>
    <circle id="geluDot" r="5" fill="#1a2332" style="display:none"/>
    <text x="${W - P}" y="${zeroY - 6}" font-size="10" fill="#5b6474" text-anchor="end">x</text>
    <text x="${zeroX + 6}" y="${P + 4}" font-size="10" fill="#5b6474">GELU(x)</text>
    <text x="${P}" y="${H - 8}" font-size="10" fill="#5b6474">-3</text>
    <text x="${W - P - 6}" y="${H - 8}" font-size="10" fill="#5b6474">+3</text>
  </svg>
  <div class="kv"><span class="k">hover readout</span><span class="v" id="geluReadout">${hlX === null ? "move over the curve" : `x=${fmt(hlX, 3)} → GELU=${fmt(gelu(hlX), 4)}`}</span></div>`;
}
