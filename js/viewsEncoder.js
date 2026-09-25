/* Encoder mode (Mode A) stage views. All numbers come from S.encTrace. */
import { fnv1a, statsOf } from "./engine.js";
import { fmt, esc, shapeBadge, matrixHTML, barsHTML, tokenChips, legendHTML, geluSVG } from "./format.js";

export const ENCODER_STAGES = [
  { id: "tokenize", nav: "Tokenize", title: "Tokenization", sub: "Text → tokens → token IDs (toy tokenizer)", shape: "text → [seq]" },
  { id: "embed", nav: "Embed", title: "Embedding Lookup", sub: "Token ID → embedding vector (toy deterministic table)", shape: "[seq] → [seq, d_model]" },
  { id: "pos", nav: "Position", title: "Positional Encoding", sub: "Embedding + sinusoidal position code", shape: "[seq, d] + [seq, d]" },
  { id: "qkv", nav: "Q / K / V", title: "Q, K, V Projection", sub: "One input, three learned linear projections", shape: "[seq, d] × [d, d]" },
  { id: "heads", nav: "Heads", title: "Multi-Head Split", sub: "Reshape [seq, d_model] → [n_heads, seq, d_head]", shape: "[seq, d] → [h, seq, d_h]" },
  { id: "attn", nav: "Attention", title: "Scaled Dot-Product Attention", sub: "softmax(QKᵀ / √d)V — the core operation", shape: "[seq, seq] → [seq, d_h]" },
  { id: "concat", nav: "Concat+Proj", title: "Concatenate + Output Projection", sub: "Reunite heads, mix with W_O", shape: "[h,seq,d_h] → [seq, d]" },
  { id: "ffn", nav: "FFN", title: "Feed-Forward Network", sub: "GELU(xW₁+b₁)W₂+b₂, per position", shape: "[seq, d] → [seq, d_ff] → [seq, d]" },
];

function emptyState() {
  return `<div class="simplify-note"><b>Empty input.</b> Type text above (or keep the demo “The cat sat”) to run the computation.</div>`;
}

export function renderEncoderStage(id, S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const seq = T ? T.ids.length : 0;
  if (!T || !seq) {
    if (id === "tokenize") return vTokenize(S);
    return emptyState();
  }
  switch (id) {
    case "tokenize": return vTokenize(S);
    case "embed": return vEmbed(S);
    case "pos": return vPos(S);
    case "qkv": return vQKV(S);
    case "heads": return vHeads(S);
    case "attn": return vAttn(S);
    case "concat": return vConcat(S);
    case "ffn": return vFFN(S);
    default: return "";
  }
}

/* ---------- Stage 1: Tokenization ---------- */
function vTokenize(S) {
  const T = S.encTrace, cfg = S.config;
  const rows = T.tokens.map((t, i) => {
    const h = fnv1a(t);
    return `<tr><td class="rowlab">${esc(t)}</td><td style="text-align:left">${h}</td><td>id ${T.ids[i]}</td></tr>`;
  }).join("");
  return `
    <div class="simplify-note"><b>Educational simplification ·</b> Toy tokenizer: whitespace split + FNV-1a hash mod ${cfg.vocabSize}. Real GPT-family tokenizers use learned subword schemes (e.g. BPE), not hashing.</div>
    <label for="encInput"><b>Input text</b></label>
    <textarea class="input-text" id="encInput" rows="2">${esc(S.encInput)}</textarea>
    <div class="flow"><span class="tensor-box">“${esc(S.encInput)}”</span><span class="arrow">→</span><span class="op-box">split + hash</span><span class="arrow">→</span><span class="tensor-box">[${T.ids.join(", ") || "—"}]</span></div>
    <div>${tokenChips(T.tokens, T.ids)}</div>
    <div style="margin-top:8px">${shapeBadge([T.ids.length])} <span class="stage-sub">token ID sequence</span></div>
    <div class="matrix-wrap"><table class="matrix">
      <tr><th class="rowlab">token</th><th>FNV-1a hash</th><th>hash mod ${cfg.vocabSize}</th></tr>
      ${rows || `<tr><td class="rowlab">—</td><td>—</td><td>—</td></tr>`}
    </table></div>
    <div class="controls-row"><button class="btn small" data-action="inspect" data-tensor="ids">Inspect in Tensor Inspector ↓</button></div>`;
}

/* ---------- Stage 2: Embedding ---------- */
function vEmbed(S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const dCols = Array.from({ length: cfg.dModel }, (_, j) => `d${j + 1}`);
  return `
    <div class="simplify-note"><b>Educational simplification ·</b> Toy embedding table: seeded Gaussian values (seed ${cfg.seed}). Real embeddings are <i>learned during training</i> — never sampled at inference time.</div>
    <div class="flow"><span class="tensor-box">ids [${T.ids.join(", ")}]</span><span class="arrow">→ lookup row of E</span><span class="op-box">E ∈ R^{${cfg.vocabSize}×${cfg.dModel}}</span><span class="arrow">→</span><span class="tensor-box">X</span></div>
    <div>${shapeBadge([T.ids.length])} <span class="arrow">→</span> ${shapeBadge([T.ids.length, cfg.dModel])}</div>
    ${legendHTML()}
    ${matrixHTML(T.X, { rowLabels: T.tokens.map((t, i) => `${t} (id ${T.ids[i]})`), colLabels: dCols, precision: P, caption: "Embedding matrix" })}
    <div class="controls-row"><span class="stage-sub">Hover any cell for its exact value.</span><button class="btn small" data-action="inspect" data-tensor="X">Inspect X ↓</button></div>`;
}

/* ---------- Stage 3: Positional encoding ---------- */
function vPos(S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const view = S.embView;
  const dCols = Array.from({ length: cfg.dModel }, (_, j) => `d${j + 1}`);
  const tabs = [["X", "Embedding"], ["PE", "Position Encoding"], ["sum", "Embedding + Position"]].map(([v, l]) =>
    `<button class="subtab${view === v ? " active" : ""}" data-action="embView" data-v="${v}">${l}</button>`).join("");
  const M = view === "X" ? T.X : view === "PE" ? T.PE : T.Xpos;
  const name = view === "X" ? "Embedding X" : view === "PE" ? "Positional encoding PE" : "Position-aware X + PE";
  return `
    <div class="flow"><span class="tensor-box">X ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">+</span><span class="tensor-box">PE (sin/cos) ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">→</span><span class="tensor-box"><b>X + PE</b></span></div>
    <div class="eq-block">PE(pos,2i) = sin(pos / 10000<sup>2i/d</sup>) &nbsp;&nbsp; PE(pos,2i+1) = cos(pos / 10000<sup>2i/d</sup>)</div>
    <div class="subtabs" role="tablist">${tabs}</div>
    <div><b>${name}</b> · X axis = position →, Y axis = dimension ↓, intensity = value</div>
    ${legendHTML()}
    ${matrixHTML(M, { rowLabels: T.tokens.map((t, i) => `pos ${i}: ${t}`), colLabels: dCols, precision: P, caption: name })}
    <div class="controls-row"><button class="btn small" data-action="inspect" data-tensor="Xpos">Inspect X+PE ↓</button></div>`;
}

/* ---------- Stage 4: QKV ---------- */
function vQKV(S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const e = T.enc, w = T.weights.layers[0];
  const q = S.qkvTab;
  const tabs = [["Q", "Q = XW_Q"], ["K", "K = XW_K"], ["V", "V = XW_V"]].map(([v, l]) =>
    `<button class="subtab${q === v ? " active" : ""}" data-action="qkvTab" data-v="${v}">${l}</button>`).join("");
  const W = q === "Q" ? w.WQ : q === "K" ? w.WK : w.WV;
  const O = q === "Q" ? e.Q : q === "K" ? e.K : e.V;
  const dCols = Array.from({ length: cfg.dModel }, (_, j) => `d${j + 1}`);
  let calc = "";
  if (S.showCalc) {
    const i = Math.min(S.selQuery, T.ids.length - 1);
    const terms = T.Xpos[i].map((x, k) => `${fmt(x, 3)}×${fmt(W[k][0], 3)}`);
    const sum = T.Xpos[i].reduce((a, x, k) => a + x * W[k][0], 0);
    calc = `<div class="calc-box">${q}[${i},0] = X[${i}]·W&lt;col 0&gt; = ${terms.slice(0, 4).join(" + ")}${terms.length > 4 ? " + …" : ""} = <b>${fmt(sum, P)}</b> (matches the matrix ✓)</div>`;
  }
  return `
    <div class="flow"><span class="tensor-box">X+PE ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div class="flow"><span class="arrow">├── × W_Q →</span><b>Q</b><span class="arrow">├── × W_K →</span><b>K</b><span class="arrow">└── × W_V →</span><b>V</b></div>
    <div class="subtabs">${tabs}</div>
    <div class="flow"><span class="tensor-box">X ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">×</span><span class="tensor-box">W_${q} ${shapeBadge([cfg.dModel, cfg.dModel])}</span><span class="arrow">→</span><span class="tensor-box"><b>${q}</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div><b>W_${q}</b> (toy learned-stand-in weights, seed ${cfg.seed})</div>
    ${matrixHTML(W, { colLabels: dCols, rowLabels: dCols, precision: P, maxRows: 16, maxCols: 16, caption: `W_${q}` })}
    <div><b>${q}</b> output</div>
    ${matrixHTML(O, { rowLabels: T.tokens, colLabels: dCols, precision: P, caption: q })}
    <div class="controls-row">
      <button class="btn small" data-action="toggleCalc">${S.showCalc ? "Hide calculation" : "Show calculation"}</button>
      <button class="btn small" data-action="inspect" data-tensor="${q}">Inspect ${q} ↓</button>
    </div>${calc}`;
}

/* ---------- Stage 5: heads ---------- */
function vHeads(S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const e = T.enc, h = S.selHead;
  const pills = Array.from({ length: cfg.nHeads }, (_, i) =>
    `<button class="subtab${h === i ? " active" : ""}" data-action="selHead" data-i="${i}">Head ${i + 1}</button>`).join("");
  const dc = Array.from({ length: e.dHead }, (_, j) => `h${j + 1}`);
  return `
    <div class="flow"><span class="tensor-box">Q/K/V ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">→ reshape →</span><span class="tensor-box"><b>[${cfg.nHeads}, ${T.ids.length}, ${e.dHead}]</b></span></div>
    <div class="eq-block">d_model = ${cfg.dModel},&nbsp; n_heads = ${cfg.nHeads},&nbsp; d_head = ${cfg.dModel}/${cfg.nHeads} = ${e.dHead}</div>
    <div class="head-pills">${pills}</div>
    <div><b>Head ${h + 1}</b> — Q slice ${shapeBadge([T.ids.length, e.dHead])}</div>
    ${matrixHTML(e.Qh[h], { rowLabels: T.tokens, colLabels: dc, precision: P })}
    <div><b>Head ${h + 1}</b> — K slice</div>
    ${matrixHTML(e.Kh[h], { rowLabels: T.tokens, colLabels: dc, precision: P })}
    <div><b>Head ${h + 1}</b> — V slice</div>
    ${matrixHTML(e.Vh[h], { rowLabels: T.tokens, colLabels: dc, precision: P })}`;
}

/* ---------- Stage 6: attention ---------- */
function vAttn(S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const e = T.enc, h = S.selHead, a = e.attn[h];
  const q = Math.min(S.selQuery, T.ids.length - 1);
  const pills = Array.from({ length: cfg.nHeads }, (_, i) =>
    `<button class="subtab${h === i ? " active" : ""}" data-action="selHead" data-i="${i}">Head ${i + 1}</button>`).join("");
  const steps = [["scores", "6.1 scores QKᵀ"], ["scaled", "6.2 scaled ÷√d"], ["weights", "6.3 softmax"], ["out", "6.4 ×V output"]].map(([v, l]) =>
    `<button class="subtab${S.attnView === v ? " active" : ""}" data-action="attnView" data-v="${v}">${l}</button>`).join("");
  let body = "";
  const tokLabs = T.tokens;
  if (S.attnView === "scores") {
    body = `<div><b>Attention scores</b> Q×Kᵀ ${shapeBadge([T.ids.length, T.ids.length])} — raw query/key compatibility (not probabilities)</div>
    ${matrixHTML(a.scores, { rowLabels: tokLabs.map((t) => `q:${t}`), colLabels: tokLabs.map((t) => `k:${t}`), precision: P, hlRow: q })}`;
  } else if (S.attnView === "scaled") {
    body = `<div><b>Scaled scores</b> ÷ √${e.dHead} = ÷ ${fmt(Math.sqrt(e.dHead), 4)} — keeps softmax inputs in a stable range</div>
    <div class="eq-block">scaled = scores / √d_head = scores / ${fmt(Math.sqrt(e.dHead), 4)}</div>
    ${matrixHTML(a.scaled, { rowLabels: tokLabs.map((t) => `q:${t}`), colLabels: tokLabs.map((t) => `k:${t}`), precision: P, hlRow: q })}`;
  } else if (S.attnView === "weights") {
    const sums = a.weights.map((r) => r.reduce((x, y) => x + y, 0));
    body = `<div><b>Attention weights</b> ${shapeBadge([T.ids.length, T.ids.length])} — each row is a distribution over keys. <span class="row-sum">Row sums: ${sums.map((s) => fmt(s, 4)).join(" · ")}</span></div>
    ${matrixHTML(a.weights, { rowLabels: tokLabs.map((t) => `q:${t}`), colLabels: tokLabs.map((t) => `k:${t}`), precision: P, hlRow: q })}`;
  } else {
    const wrow = a.weights[q];
    const contrib = T.tokens.map((t, j) => ({ label: `${t}`, value: wrow[j], kept: true, selected: j === q, extra: `V-row weight` }));
    body = `<div><b>Context vectors</b> = weights × V ${shapeBadge([T.ids.length, e.dHead])} — query <b>${esc(T.tokens[q])}</b> highlighted</div>
    ${matrixHTML(a.output, { rowLabels: tokLabs, colLabels: Array.from({ length: e.dHead }, (_, j) => `c${j + 1}`), precision: P, hlRow: q })}
    <div style="margin-top:6px"><b>Which tokens contributed to “${esc(T.tokens[q])}”?</b> (attention weights, row ${q})</div>
    ${barsHTML(contrib, { precision: P })}`;
  }
  let calc = "";
  if (S.showCalc) {
    const k = q;
    const terms = e.Qh[h][q].map((x, d) => x * e.Kh[h][k][d]);
    const sum = terms.reduce((x, y) => x + y, 0);
    calc = `<div class="calc-box">scores[${q},${k}] = Q[${q}]·K[${k}] = ${terms.map((t) => fmt(t, 3)).join(" + ")} = <b>${fmt(sum, P)}</b> → ÷√${e.dHead} = <b>${fmt(sum / Math.sqrt(e.dHead), P)}</b> ✓ matches the matrices</div>`;
  }
  return `
    <div class="eq-block">Attention(Q,K,V) = softmax(QKᵀ / √d<sub>head</sub>) · V &nbsp;&nbsp;·&nbsp;&nbsp; head ${h + 1}/${cfg.nHeads}, d_head=${e.dHead}</div>
    <div class="head-pills">${pills}</div>
    <div><b>Query token:</b></div>
    <div>${tokenChips(T.tokens, T.ids, { selected: q, clickable: true })}</div>
    <div class="subtabs">${steps}</div>
    <div class="attn-flow">
      ${["scores", "scaled", "weights", "out"].map((v, i) => `${i ? '<span class="arrow">→</span>' : ""}<span class="attn-node${S.attnView === v ? " active" : ""}" data-action="attnView" data-v="${v}" role="button" tabindex="0">${["QKᵀ", "÷√d", "softmax", "×V"][i]}</span>`).join("")}
    </div>
    ${body}
    <div class="controls-row"><button class="btn small" data-action="toggleCalc">${S.showCalc ? "Hide calculation" : "Show calculation"}</button></div>${calc}`;
}

/* ---------- Stage 7: concat + WO ---------- */
function vConcat(S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const e = T.enc, w = T.weights.layers[0];
  const dCols = Array.from({ length: cfg.dModel }, (_, j) => `d${j + 1}`);
  const headList = Array.from({ length: cfg.nHeads }, (_, i) => `<span class="tensor-box">Head ${i + 1} ctx ${shapeBadge([T.ids.length, e.dHead])}</span>`).join(' <span class="arrow">+</span> ');
  return `
    <div class="flow">${headList}</div>
    <div class="flow"><span class="arrow">↓ concatenate (reorder only, no math) ↓</span></div>
    <div class="flow"><span class="tensor-box"><b>concat</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">× W_O ${shapeBadge([cfg.dModel, cfg.dModel])} →</span><span class="tensor-box"><b>output</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div class="eq-block">out = concat(head<sub>1</sub> … head<sub>${cfg.nHeads}</sub>) · W<sub>O</sub> &nbsp;&nbsp;—&nbsp;&nbsp; concatenate ≠ project</div>
    <div><b>Concatenated heads</b></div>
    ${matrixHTML(e.concat, { rowLabels: T.tokens, colLabels: dCols, precision: P })}
    <div><b>After W_O</b></div>
    ${matrixHTML(e.proj, { rowLabels: T.tokens, colLabels: dCols, precision: P })}
    <div class="controls-row"><button class="btn small" data-action="inspect" data-tensor="proj">Inspect output ↓</button></div>`;
}

/* ---------- Stage 8: FFN ---------- */
function vFFN(S) {
  const T = S.encTrace, cfg = S.config, P = cfg.precision;
  const f = T.enc.ffn;
  const dCols = Array.from({ length: cfg.dModel }, (_, j) => `d${j + 1}`);
  const fCols = Array.from({ length: cfg.dFF }, (_, j) => `f${j + 1}`);
  return `
    <div class="eq-block">FFN(x) = GELU(xW<sub>1</sub>+b<sub>1</sub>)W<sub>2</sub>+b<sub>2</sub></div>
    <div class="flow"><span class="tensor-box">attn out ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">→ ×W₁ →</span><span class="tensor-box">hidden ${shapeBadge([T.ids.length, cfg.dFF])}</span><span class="arrow">→ GELU →</span><span class="tensor-box">×W₂ →</span><span class="tensor-box"><b>final</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div class="subtabs"><button class="subtab${S.ffnView === "pre" ? " active" : ""}" data-action="ffnView" data-v="pre">Hidden (pre-GELU)</button><button class="subtab${S.ffnView === "act" ? " active" : ""}" data-action="ffnView" data-v="act">Hidden (GELU)</button><button class="subtab${S.ffnView === "out" ? " active" : ""}" data-action="ffnView" data-v="out">Output</button><button class="subtab${S.ffnView === "gelu" ? " active" : ""}" data-action="ffnView" data-v="gelu">GELU graph</button></div>
    ${S.ffnView === "gelu" ? `${geluSVG()}<div class="eq-block">GELU(x) ≈ 0.5x·(1 + tanh(√(2/π)·(x + 0.044715x³)))</div>`
      : S.ffnView === "pre" ? `<div><b>Hidden pre-activation</b> xW₁+b₁ ${shapeBadge([T.ids.length, cfg.dFF])}</div>${matrixHTML(f.hiddenPre, { rowLabels: T.tokens, colLabels: fCols, precision: P })}`
      : S.ffnView === "act" ? `<div><b>Hidden activated</b> GELU(·) ${shapeBadge([T.ids.length, cfg.dFF])} — negatives squash toward 0</div>${matrixHTML(f.hiddenAct, { rowLabels: T.tokens, colLabels: fCols, precision: P })}`
      : `<div><b>FFN output</b> (final encoder representation) ${shapeBadge([T.ids.length, cfg.dModel])}</div>${matrixHTML(f.output, { rowLabels: T.tokens, colLabels: dCols, precision: P })}`}
    <div class="simplify-note"><b>Note ·</b> This Mode-A path follows the spec pipeline (no residuals/LayerNorm). Production encoders add residual + norm around attention and FFN — see the full pre-LN block in <b>Mode B</b>.</div>`;
}
