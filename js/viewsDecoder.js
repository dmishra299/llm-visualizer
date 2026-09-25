/* Decoder mode (Mode B) stage views. Numbers come from S.decTrace / S.decStep. */
import { gelu } from "./engine.js";
import { fmt, esc, shapeBadge, matrixHTML, barsHTML, tokenChips, legendHTML, geluSVG } from "./format.js";

export const DECODER_STAGES = [
  { id: "dtok", nav: "Tokenize", title: "Tokenization", sub: "Prompt → tokens → IDs (toy tokenizer)", shape: "text → [seq]" },
  { id: "demb", nav: "Embed+Pos", title: "Embedding + Positional Encoding", sub: "IDs → vectors + sinusoidal codes", shape: "[seq] → [seq, d]" },
  { id: "dln1", nav: "LayerNorm 1", title: "LayerNorm (pre-attention)", sub: "Per-token feature normalization", shape: "[seq, d] → [seq, d]" },
  { id: "dqkv", nav: "Q/K/V+Heads", title: "Q/K/V Projection + Head Split", sub: "Project, then reshape per head", shape: "[seq,d] → [h,seq,d_h]" },
  { id: "dmask", nav: "Mask+Attn", title: "Causal Mask + Attention", sub: "No peeking at the future, then softmax(QKᵀ/√d)V", shape: "[seq, seq]" },
  { id: "dres1", nav: "Concat+Res", title: "Concat + W_O + Residual", sub: "Heads reunite, project, add back to stream", shape: "x + attn" },
  { id: "dffn", nav: "LN+FFN+Res", title: "LayerNorm + FFN + Residual", sub: "Second half of the pre-LN block", shape: "x′ + FFN" },
  { id: "dblocks", nav: "Stack ×N", title: "Transformer Block Stack", sub: "Repeat the block N times (progressive disclosure)", shape: "×N blocks" },
  { id: "dfinal", nav: "Final LN", title: "Final LayerNorm", sub: "Normalize last block's output", shape: "[seq, d]" },
  { id: "dlm", nav: "LM Head", title: "LM Head + Logits", sub: "Hidden states → raw vocabulary scores", shape: "[seq, d] → [seq, V]" },
  { id: "dtemp", nav: "Temp+Soft", title: "Temperature + Softmax", sub: "Shape the distribution, normalize to probabilities", shape: "[V] → [V], Σ=1" },
  { id: "dsample", nav: "Sample+Loop", title: "Sampling + Generation Loop", sub: "Draw a token, append, repeat (autoregressive)", shape: "[V] → 1 token" },
];

export function renderDecoderStage(id, S) {
  const T = S.decTrace;
  if (!T || !T.ids.length) {
    if (id === "dtok" || id === "dsample") return id === "dtok" ? vDTok(S) : vDSample(S);
    return `<div class="simplify-note"><b>Empty context.</b> Enter a prompt to run the decoder.</div>`;
  }
  switch (id) {
    case "dtok": return vDTok(S);
    case "demb": return vDEmb(S);
    case "dln1": return vDLN1(S);
    case "dqkv": return vDQKV(S);
    case "dmask": return vDMask(S);
    case "dres1": return vDRes1(S);
    case "dffn": return vDFFN(S);
    case "dblocks": return vDBlocks(S);
    case "dfinal": return vDFinal(S);
    case "dlm": return vDLM(S);
    case "dtemp": return vDTemp(S);
    case "dsample": return vDSample(S);
    default: return "";
  }
}

function blockOf(S) {
  const T = S.decTrace;
  return T.blocks[Math.min(S.selBlock, T.blocks.length - 1)];
}
function blockBar(S) {
  const T = S.decTrace;
  if (T.blocks.length < 2) return "";
  return `<div class="controls-row"><span class="stage-sub">Block:</span>` + T.blocks.map((_, l) =>
    `<button class="subtab${S.selBlock === l ? " active" : ""}" data-action="selBlock" data-i="${l}">Block ${l + 1}</button>`).join("") + `</div>`;
}
function dCols(cfg) { return Array.from({ length: cfg.dModel }, (_, j) => `d${j + 1}`); }

/* ---------- D1 tokenize ---------- */
function vDTok(S) {
  const T = S.decTrace, cfg = S.config;
  const promptIds = T.promptIds || [];
  const genStart = promptIds.length;
  return `
    <div class="simplify-note"><b>Educational simplification ·</b> Toy tokenizer: whitespace split + FNV-1a hash mod ${cfg.vocabSize}. <b>Inference demo</b> — weights are fixed; nothing is trained here.</div>
    <label for="decInput"><b>Prompt</b> (Reset re-tokenizes this; generated tokens append below)</label>
    <textarea class="input-text" id="decInput" rows="2">${esc(S.decPrompt)}</textarea>
    <div class="controls-row"><button class="btn small primary" data-action="resetGen">Tokenize prompt & Reset</button></div>
    <div><b>Current context</b> (${T.ids.length} tokens):</div>
    <div>${T.tokens.map((t, i) => `<span class="token-chip${i >= genStart ? " new" : ""}" title="${i >= genStart ? "generated" : "prompt"} token → id ${T.ids[i]}">${esc(t)}<span class="id">id ${T.ids[i]}</span></span>`).join("")}</div>
    <div style="margin-top:6px">${shapeBadge([T.ids.length])} <span class="stage-sub">context IDs: [${T.ids.join(", ")}]</span></div>`;
}

/* ---------- D2 embed ---------- */
function vDEmb(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const view = S.embView;
  const tabs = [["X", "Embedding"], ["PE", "Position"], ["sum", "X + PE"]].map(([v, l]) =>
    `<button class="subtab${view === v ? " active" : ""}" data-action="embView" data-v="${v}">${l}</button>`).join("");
  const M = view === "X" ? T.X : view === "PE" ? T.PE : T.Xpos;
  return `
    <div class="flow"><span class="tensor-box">ids ${shapeBadge([T.ids.length])}</span><span class="arrow">→ E →</span><span class="tensor-box">X</span><span class="arrow">+ PE →</span><span class="tensor-box"><b>X+PE</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div class="subtabs">${tabs}</div>
    ${legendHTML()}
    ${matrixHTML(M, { rowLabels: T.tokens.map((t, i) => `pos ${i}: ${t}`), colLabels: dCols(cfg), precision: P })}
    <div class="controls-row"><button class="btn small" data-action="inspect" data-tensor="dXpos">Inspect X+PE ↓</button></div>`;
}

/* ---------- D3 LayerNorm ---------- */
function vDLN1(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const b = blockOf(S);
  const i = Math.min(S.selQuery, T.ids.length - 1);
  const ln = b.ln1;
  return `
    ${blockBar(S)}
    <div class="eq-block">LN(x) = γ ⊙ (x − μ)/σ + β &nbsp;&nbsp;·&nbsp;&nbsp; μ,σ over the ${cfg.dModel} features of <b>each token separately</b></div>
    <div><b>Token:</b></div><div>${tokenChips(T.tokens, T.ids, { selected: i, clickable: true })}</div>
    <div class="kv">
      <span class="k">input row x</span><span class="v">[${T.blocks[Math.min(S.selBlock, T.blocks.length - 1)] ? fmtRow(blockInput(S)[i], P) : ""}]</span>
      <span class="k">mean μ</span><span class="v">${fmt(ln.means[i], P)}</span>
      <span class="k">variance σ²</span><span class="v">${fmt(ln.variances[i], P)}</span>
      <span class="k">normalized</span><span class="v">[${fmtRow(ln.normalized[i], P)}] <span class="row-sum">mean≈0 var≈1</span></span>
      <span class="k">γ, β</span><span class="v">γ=1… β=0… (toy init) → output = normalized</span>
    </div>
    <div><b>LayerNorm output</b> ${shapeBadge([T.ids.length, cfg.dModel])}</div>
    ${matrixHTML(ln.output, { rowLabels: T.tokens, colLabels: dCols(cfg), precision: P, hlRow: i })}
    <div class="simplify-note"><b>LayerNorm ≠ BatchNorm.</b> Normalization is over the feature dimension per token — never across the batch or sequence.</div>`;
}
function blockInput(S) {
  const T = S.decTrace, l = Math.min(S.selBlock, T.blocks.length - 1);
  return l === 0 ? T.Xpos : T.blocks[l - 1].output;
}
function fmtRow(row, P) { return row.map((v) => fmt(v, Math.min(P, 3))).join(", "); }

/* ---------- D4 QKV + heads ---------- */
function vDQKV(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const b = blockOf(S), l = Math.min(S.selBlock, T.blocks.length - 1), w = T.weights.layers[l];
  const q = S.qkvTab, h = S.selHead;
  const tabs = [["Q", "Q"], ["K", "K"], ["V", "V"]].map(([v, lbl]) =>
    `<button class="subtab${q === v ? " active" : ""}" data-action="qkvTab" data-v="${v}">${lbl}</button>`).join("");
  const pills = Array.from({ length: cfg.nHeads }, (_, i) =>
    `<button class="subtab${h === i ? " active" : ""}" data-action="selHead" data-i="${i}">Head ${i + 1}</button>`).join("");
  const full = q === "Q" ? b.Q : q === "K" ? b.K : b.V;
  const hs = q === "Q" ? b.Qh : q === "K" ? b.Kh : b.Vh;
  const W = q === "Q" ? w.WQ : q === "K" ? w.WK : w.WV;
  return `
    ${blockBar(S)}
    <div class="subtabs">${tabs}</div>
    <div class="flow"><span class="tensor-box">LN₁(x) ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">× W_${q} →</span><span class="tensor-box"><b>${q}</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">→ split →</span><span class="tensor-box"><b>${q} head ${h + 1}</b> ${shapeBadge([T.ids.length, b.dHead])}</span></div>
    <div class="head-pills">${pills}</div>
    <div><b>W_${q}</b> (block ${l + 1} toy weights)</div>
    ${matrixHTML(W, { colLabels: dCols(cfg), rowLabels: dCols(cfg), precision: P, maxRows: 16, maxCols: 16 })}
    <div><b>${q} head ${h + 1}</b></div>
    ${matrixHTML(hs[h], { rowLabels: T.tokens, colLabels: Array.from({ length: b.dHead }, (_, j) => `h${j + 1}`), precision: P })}
    <details style="margin-top:6px"><summary style="cursor:pointer;font-weight:600">Show full (unsplit) ${q}</summary>
    ${matrixHTML(full, { rowLabels: T.tokens, colLabels: dCols(cfg), precision: P })}</details>`;
}

/* ---------- D5 mask + attention ---------- */
function vDMask(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const b = blockOf(S), h = S.selHead, a = b.attn[h];
  const q = Math.min(S.selQuery, T.ids.length - 1);
  const pills = Array.from({ length: cfg.nHeads }, (_, i) =>
    `<button class="subtab${h === i ? " active" : ""}" data-action="selHead" data-i="${i}">Head ${i + 1}</button>`).join("");
  const steps = [["scores", "scores"], ["masked", "masked"], ["weights", "weights"], ["out", "×V out"]].map(([v, l]) =>
    `<button class="subtab${S.dAttnView === v ? " active" : ""}" data-action="dAttnView" data-v="${v}">${l}</button>`).join("");
  const labs = T.tokens;
  let body = "";
  if (S.dAttnView === "scores") {
    body = `<div><b>Unmasked scaled scores</b> QKᵀ/√d — every query <i>could</i> see every key:</div>
    <div class="resid-diagram">       ${labs.map((t) => t.slice(0, 4).padEnd(4)).join(" ")}
${labs.map((t, i) => `Q:${t.slice(0, 4).padEnd(4)}` + labs.map(() => " ✓  ").join(" ")).join("\n")}</div>
    ${matrixHTML(a.scaled, { rowLabels: labs.map((t) => `q:${t}`), colLabels: labs.map((t) => `k:${t}`), precision: P, hlRow: q })}`;
  } else if (S.dAttnView === "masked") {
    body = `<div><b>After causal mask</b> — token may attend to itself and earlier tokens, never future ones. Masked entries are <b>−∞</b> <i>before</i> softmax:</div>
    <div class="resid-diagram">       ${labs.map((t) => t.slice(0, 4).padEnd(4)).join(" ")}
${labs.map((t, i) => `Q:${t.slice(0, 4).padEnd(4)}` + labs.map((_, j) => (j > i ? " ✕  " : " ✓  ")).join(" ")).join("\n")}</div>
    ${matrixHTML(S.showMask ? a.masked : a.scaled, { rowLabels: labs.map((t) => `q:${t}`), colLabels: labs.map((t) => `k:${t}`), precision: P, hlRow: q })}
    <div class="controls-row"><button class="btn small" data-action="toggleMask">${S.showMask ? "Hide mask" : "Show mask"}</button>
    <span class="stage-sub">${S.showMask ? "Mask ON: future = −∞" : "Mask OFF (for comparison only — the model always masks)"}</span></div>`;
  } else if (S.dAttnView === "weights") {
    const sums = a.weights.map((r) => r.reduce((x, y) => x + y, 0));
    const wrow = a.weights[q];
    body = `<div><b>Attention weights</b> (softmax after masking). <span class="row-sum">Row sums: ${sums.map((s) => fmt(s, 4)).join(" · ")}</span> Future columns are exactly 0.</div>
    ${matrixHTML(a.weights, { rowLabels: labs.map((t) => `q:${t}`), colLabels: labs.map((t) => `k:${t}`), precision: P, hlRow: q })}
    <div><b>“${esc(T.tokens[q])}” drew context from:</b></div>
    ${barsHTML(labs.map((t, j) => ({ label: t, value: wrow[j], kept: true, selected: j === q })), { precision: P })}`;
  } else {
    body = `<div><b>Head ${h + 1} output</b> = weights × V ${shapeBadge([T.ids.length, b.dHead])}:</div>
    ${matrixHTML(a.output, { rowLabels: labs, colLabels: Array.from({ length: b.dHead }, (_, j) => `c${j + 1}`), precision: P, hlRow: q })}`;
  }
  return `
    ${blockBar(S)}
    <div class="eq-block">scores = QKᵀ/√d → <b>mask j&gt;i to −∞</b> → softmax → ×V &nbsp;&nbsp;·&nbsp;&nbsp; masking happens <b>before</b> softmax</div>
    <div class="head-pills">${pills}</div>
    <div><b>Query token:</b></div><div>${tokenChips(T.tokens, T.ids, { selected: q, clickable: true })}</div>
    <div class="subtabs">${steps}</div>${body}`;
}

/* ---------- D6 concat + WO + residual ---------- */
function vDRes1(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const b = blockOf(S), l = Math.min(S.selBlock, T.blocks.length - 1);
  const xin = blockInput(S);
  const i = Math.min(S.selQuery, T.ids.length - 1);
  return `
    ${blockBar(S)}
    <div class="resid-diagram">             ┌───────────────┐
x ─────────────→ │               │
│                │   Attention   │
│                │  (masked mha) │
│                └───────┬───────┘
│                        │  proj = concat·W_O
└────────────── + ◄──────┘
                 ↓
            x + proj  (residual stream preserved)</div>
    <div class="flow"><span class="tensor-box">concat ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">× W_O →</span><span class="tensor-box">proj</span><span class="arrow">+ x →</span><span class="tensor-box"><b>resid₁</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div><b>Token:</b></div><div>${tokenChips(T.tokens, T.ids, { selected: i, clickable: true })}</div>
    <div class="kv">
      <span class="k">x (stream in)</span><span class="v">[${fmtRow(xin[i], P)}]</span>
      <span class="k">proj (attn)</span><span class="v">[${fmtRow(b.proj[i], P)}]</span>
      <span class="k">x + proj</span><span class="v">[${fmtRow(b.resid1[i], P)}] ✓ element-wise sum</span>
    </div>
    <div class="subtabs"><button class="subtab${S.resView === "concat" ? " active" : ""}" data-action="resView" data-v="concat">Concat</button><button class="subtab${S.resView === "proj" ? " active" : ""}" data-action="resView" data-v="proj">Proj</button><button class="subtab${S.resView === "res" ? " active" : ""}" data-action="resView" data-v="res">x + proj</button></div>
    ${matrixHTML(S.resView === "concat" ? b.concat : S.resView === "proj" ? b.proj : b.resid1, { rowLabels: T.tokens, colLabels: dCols(cfg), precision: P, hlRow: i })}`;
}

/* ---------- D7 LN2 + FFN + residual ---------- */
function vDFFN(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const b = blockOf(S);
  const i = Math.min(S.selQuery, T.ids.length - 1);
  const f = b.ffn;
  const tabs = [["pre", "LN₂ out"], ["hid", "GELU hidden"], ["out", "FFN out"], ["res", "resid₂ = x′+FFN"], ["gelu", "GELU graph"]].map(([v, l]) =>
    `<button class="subtab${S.dffnView === v ? " active" : ""}" data-action="dffnView" data-v="${v}">${l}</button>`).join("");
  let body = "";
  if (S.dffnView === "pre") body = `${matrixHTML(b.ln2.output, { rowLabels: T.tokens, colLabels: dCols(cfg), precision: P, hlRow: i })}`;
  else if (S.dffnView === "hid") body = `<div><b>GELU hidden</b> ${shapeBadge([T.ids.length, cfg.dFF])}</div>${matrixHTML(f.hiddenAct, { rowLabels: T.tokens, colLabels: Array.from({ length: cfg.dFF }, (_, j) => `f${j + 1}`), precision: P, hlRow: i })}`;
  else if (S.dffnView === "out") body = `${matrixHTML(f.output, { rowLabels: T.tokens, colLabels: dCols(cfg), precision: P, hlRow: i })}`;
  else if (S.dffnView === "gelu") body = `${geluSVG()}<div class="eq-block">GELU(x) ≈ 0.5x·(1 + tanh(√(2/π)·(x + 0.044715x³)))</div>`;
  else body = `<div class="kv"><span class="k">x′ (resid₁)</span><span class="v">[${fmtRow(b.resid1[i], P)}]</span><span class="k">FFN out</span><span class="v">[${fmtRow(f.output[i], P)}]</span><span class="k">x′ + FFN</span><span class="v">[${fmtRow(b.output[i], P)}] ✓</span></div>
    ${matrixHTML(b.output, { rowLabels: T.tokens, colLabels: dCols(cfg), precision: P, hlRow: i })}`;
  return `
    ${blockBar(S)}
    <div class="resid-diagram">x′ ─────────────→ + ◄── FFN(LN₂(x′))
                 ↓
          x″ = x′ + FFN(LN₂(x′))   [pre-LN block complete]</div>
    <div><b>Token:</b></div><div>${tokenChips(T.tokens, T.ids, { selected: i, clickable: true })}</div>
    <div class="subtabs">${tabs}</div>${body}`;
}

/* ---------- D8 block stack ---------- */
function vDBlocks(S) {
  const T = S.decTrace, cfg = S.config;
  const cards = T.blocks.map((b, l) => `
    ${l ? '<div class="block-arrow">↓</div>' : ""}
    <div class="block-card${S.selBlock === l ? " active" : ""}" data-action="selBlock" data-i="${l}" role="button" tabindex="0">
      <span class="bt">Block ${l + 1}</span> <span class="bs">LN→masked-attn→+res → LN→FFN→+res · [${T.ids.length}, ${cfg.dModel}] → [${T.ids.length}, ${cfg.dModel}]</span>
    </div>`).join("");
  return `
    <div class="simplify-note"><b>Pre-LN decoder stack ·</b> ${cfg.nLayers} block${cfg.nLayers > 1 ? "s" : ""}. Only shapes are shown here — select a block, then revisit stages 3–7 to inspect its internals (progressive disclosure).</div>
    <div class="flow"><span class="tensor-box">X+PE ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div class="block-stack">${cards}</div>
    <div class="flow"><span class="tensor-box"><b>h<sub>N</sub></b> ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    <div class="eq-block">h<sub>l+1</sub> = Block<sub>l</sub>(h<sub>l</sub>) &nbsp;&nbsp;·&nbsp;&nbsp; currently inspecting: <b>Block ${S.selBlock + 1}</b></div>`;
}

/* ---------- D9 final LN ---------- */
function vDFinal(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  return `
    <div class="flow"><span class="tensor-box">h<sub>N</sub> ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">→ LayerNorm →</span><span class="tensor-box"><b>final states</b> ${shapeBadge([T.ids.length, cfg.dModel])}</span></div>
    ${matrixHTML(T.finalLN.output, { rowLabels: T.tokens, colLabels: dCols(cfg), precision: P })}`;
}

/* ---------- D10 LM head + logits ---------- */
function vDLM(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const last = T.ids.length - 1;
  const items = T.logits[last].map((v, id) => ({ id, tok: T.vocab[id], v }))
    .sort((a, b) => b.v - a.v).slice(0, S.topN)
    .map((d) => ({ label: `${d.tok}`, value: d.v, kept: true, extra: `id ${d.id}` }));
  const mx = Math.max(...items.map((d) => Math.abs(d.value)));
  return `
    <div class="flow"><span class="tensor-box">final ${shapeBadge([T.ids.length, cfg.dModel])}</span><span class="arrow">× W<sub>LM</sub> ${shapeBadge([cfg.dModel, cfg.vocabSize])} →</span><span class="tensor-box"><b>logits</b> ${shapeBadge([T.ids.length, cfg.vocabSize])}</span></div>
    <div class="simplify-note"><b>Logit ≠ probability.</b> Each row holds one raw score per vocabulary token. Only the <b>last row</b> (position ${last}, “${esc(T.tokens[last])}”) predicts the next token.</div>
    <div class="controls-row"><span class="stage-sub">Show top</span>
      ${[5, 12, 20].map((n) => `<button class="subtab${S.topN === n ? " active" : ""}" data-action="topN" data-i="${n}">${n}</button>`).join("")}
      <span class="stage-sub">of ${cfg.vocabSize}</span></div>
    ${barsHTML(items, { precision: P, maxAbs: mx })}
    <div class="controls-row"><button class="btn small" data-action="inspect" data-tensor="logits">Inspect full logits ↓</button></div>`;
}

/* ---------- D11 temperature + softmax ---------- */
function vDTemp(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  const st = S.decStep;
  const order = st.probs.map((p, id) => ({ id, p, z: st.logitsLast[id], s: st.scaled[id] }))
    .sort((a, b) => b.p - a.p).slice(0, S.topN);
  const sum = st.probs.reduce((a, b) => a + b, 0);
  return `
    <div class="eq-block">p<sub>i</sub> = e<sup>z<sub>i</sub>/T</sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub>/T</sup> &nbsp;&nbsp;·&nbsp;&nbsp; T = ${cfg.temperature.toFixed(2)} &nbsp;&nbsp;·&nbsp;&nbsp; <span class="row-sum">Σp = ${sum.toFixed(4)}</span></div>
    <div class="controls-row"><label for="tempSlider2"><b>Temperature</b> (logits ÷ T <i>before</i> softmax)</label>
      <input type="range" id="tempSlider2" min="0.1" max="2" step="0.05" value="${cfg.temperature}" data-action="temp" aria-label="Temperature" />
      <span class="val shape-badge">T = ${cfg.temperature.toFixed(2)}</span></div>
    <div class="stage-sub">Lower T → sharper (peaked) · Higher T → flatter (uniform). Shape only — not intelligence.</div>
    <div class="matrix-wrap"><table class="matrix">
      <tr><th class="rowlab">token</th><th>logit z</th><th>z / T</th><th>e<sup>z/T</sup></th><th>probability</th></tr>
      ${order.map((d) => {
        const e = Math.exp(d.s - Math.max(...st.scaled));
        return `<tr><td class="rowlab">${esc(T.vocab[d.id])} <span class="stage-sub">id ${d.id}</span></td><td>${fmt(d.z, P)}</td><td>${fmt(d.s, P)}</td><td>${fmt(e, P)}</td><td><b>${fmt(d.p, P)}</b></td></tr>`;
      }).join("")}
    </table></div>
    ${barsHTML(order.map((d) => ({ label: T.vocab[d.id], value: d.p, kept: true })), { precision: P })}`;
}

/* ---------- D12 sampling + loop ---------- */
function vDSample(S) {
  const T = S.decTrace, cfg = S.config, P = cfg.precision;
  if (!T || !T.ids.length) return `<div class="simplify-note"><b>Empty context.</b> Go to stage 1 and tokenize a prompt first.</div>`;
  const st = S.decStep;
  const sm = st.sampling;
  const order = sm.candidates.map((c) => ({ ...c, tok: T.vocab[c.id] }))
    .sort((a, b) => b.prob - a.prob).slice(0, S.topN);
  const keptN = sm.candidates.filter((c) => c.kept).length;
  const methodBtns = [["greedy", "Greedy"], ["top-k", "Top-k"], ["top-p", "Top-p"]].map(([v, l]) =>
    `<button class="subtab${cfg.method === v ? " active" : ""}" data-action="method" data-v="${v}">${l}</button>`).join("");
  let methodCtl = "";
  if (cfg.method === "top-k") methodCtl = `<span class="stage-sub">k =</span>` + [1, 3, 5, 10].map((k) =>
    `<button class="subtab${cfg.k === k ? " active" : ""}" data-action="k" data-i="${k}">${k}</button>`).join("");
  else if (cfg.method === "top-p") methodCtl = `<input type="range" min="0.1" max="1" step="0.05" value="${cfg.p}" data-action="p" aria-label="Top-p threshold" style="width:120px;accent-color:var(--accent)" /> <span class="shape-badge">p = ${cfg.p.toFixed(2)}</span>`;
  else methodCtl = `<span class="stage-sub">argmax — always the single best token</span>`;
  const hist = S.history.map((h, i) => `
    <div class="hist-step${S.selHist === i ? " active" : ""}" data-action="selHist" data-i="${i}" role="button" tabindex="0">
      <span class="step">step ${h.step}</span>${esc(h.contextTokens.join(" "))} <span class="newtok">+ ${esc(h.selectedToken)}</span>
      <span class="stage-sub">p=${fmt(h.probs[h.selectedId], 4)} · ${h.method}${h.method === "top-k" ? " k=" + h.k : h.method === "top-p" ? " p=" + h.p : ""}</span>
    </div>`).join("");
  let histDetail = "";
  if (S.selHist >= 0 && S.history[S.selHist]) {
    const h = S.history[S.selHist];
    const ho = h.probs.map((p, id) => ({ id, p })).sort((a, b) => b.p - a.p).slice(0, 10);
    histDetail = `<div><b>Step ${h.step}</b> distribution (selected: “${esc(h.selectedToken)}” id ${h.selectedId}):</div>
    ${barsHTML(ho.map((d) => ({ label: T.vocab[d.id], value: d.p, kept: true, selected: d.id === h.selectedId })), { precision: P })}`;
  }
  return `
    <div class="eq-block">${cfg.method === "greedy" ? "next = argmax(p)" : cfg.method === "top-k" ? `keep top ${cfg.k}, renormalize, sample` : `keep smallest set with cumsum ≥ ${cfg.p.toFixed(2)}, renormalize, sample`} &nbsp;&nbsp;·&nbsp;&nbsp; eligible: <b>${keptN}</b> / ${cfg.vocabSize}</div>
    <div class="subtabs">${methodBtns}</div>
    <div class="controls-row">${methodCtl}</div>
    ${barsHTML(order.map((d) => ({ label: d.tok, value: d.prob, kept: d.kept, selected: d.id === sm.selectedId, extra: `id ${d.id}` })), { precision: P, colorByKept: true })}
    <div class="kv"><span class="k">selected (preview)</span><span class="v">“${esc(T.vocab[sm.selectedId])}” id ${sm.selectedId} · p=${fmt(st.probs[sm.selectedId], P)} — press <b>Next Token</b> to append it</span></div>
    <div class="controls-row">
      <button class="btn primary" data-action="nextToken">Next Token →</button>
      <button class="btn" data-action="autoGen">${S.autoGen ? "Pause" : "Auto Generate"}</button>
      <button class="btn" data-action="stepBack" ${S.history.length ? "" : "disabled"}>Step Back</button>
      <button class="btn ghost" data-action="resetGen">Reset</button>
    </div>
    <div><b>Generation history</b> — click any step to inspect the exact distribution that produced it:</div>
    <div class="gen-history">${hist || `<span class="stage-sub">No tokens generated yet. Press “Next Token”.</span>`}</div>
    ${histDetail}
    <div class="simplify-note"><b>Autoregressive loop ·</b> the model generates a token, appends it to the sequence, and uses the resulting context to predict the next token. Sampling seed = ${cfg.seed} ⊕ step, so runs reproduce exactly.</div>`;
}
