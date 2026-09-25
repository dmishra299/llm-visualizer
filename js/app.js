/* App shell: central state, navigation, config, inspector, generation loop.
 * The UI renders from ONE computation (S.encTrace / S.decTrace + S.decStep).
 */
import * as E from "./engine.js";
import { ENCODER_EXPLANATIONS, DECODER_EXPLANATIONS, ABOUT_TEXT } from "./explanations.js";
import { fmt, esc, shapeBadge, matrixHTML, barsHTML, statsGrid } from "./format.js";
import { ENCODER_STAGES, renderEncoderStage } from "./viewsEncoder.js";
import { DECODER_STAGES, renderDecoderStage } from "./viewsDecoder.js";
import { runTests } from "./tests.js";

/* ---------------- State ---------------- */
const S = {
  mode: "encoder",
  stage: 0,
  config: { dModel: 8, dFF: 16, nHeads: 2, nLayers: 2, vocabSize: 32, temperature: 1.0, method: "top-k", k: 5, p: 0.9, seed: 42, precision: 4 },
  encInput: "The cat sat",
  decPrompt: "The cat",
  decIds: [], decTokens: [], promptLen: 0,
  history: [], sampleCounter: 0, selHist: -1,
  encTrace: null, decTrace: null, decStep: null,
  truncated: false,
  // view selections
  selHead: 0, selQuery: 0, selBlock: 0,
  embView: "sum", qkvTab: "Q", attnView: "weights", dAttnView: "weights",
  ffnView: "out", dffnView: "res", resView: "res",
  showMask: true, showCalc: false, showMore: false, showMath: false,
  topN: 12, selTensor: "ids",
  playing: false, playTimer: null, autoGen: false, autoTimer: null,
};

const $ = (id) => document.getElementById(id);
const stages = () => (S.mode === "encoder" ? ENCODER_STAGES : DECODER_STAGES);
const explanations = () => (S.mode === "encoder" ? ENCODER_EXPLANATIONS : DECODER_EXPLANATIONS);

/* ---------------- Computation (single source of truth) ---------------- */
function configError() {
  const c = S.config;
  if (c.dModel % c.nHeads !== 0) return `Invalid configuration: d_model (${c.dModel}) must be divisible by n_heads (${c.nHeads}).`;
  return null;
}

function recomputeEncoder() {
  const words = S.encInput.trim().split(/\s+/).filter(Boolean);
  S.truncated = words.length > 12;
  const text = S.truncated ? words.slice(0, 12).join(" ") : S.encInput;
  S.encTrace = E.forwardEncoder(text, S.config, S.config.seed);
}

function resetDecoder() {
  const { tokens, ids } = E.tokenize(S.decPrompt, S.config.vocabSize);
  S.decTokens = tokens.slice(0, 16);
  S.decIds = ids.slice(0, 16);
  S.promptLen = S.decIds.length;
  S.history = []; S.sampleCounter = 0; S.selHist = -1;
  recomputeDecoder();
}

function recomputeDecoder() {
  S.decTrace = E.forwardDecoder(S.decIds, S.decTokens, S.config, S.config.seed);
  S.decTrace.promptIds = S.decIds.slice(0, S.promptLen);
  if (S.decIds.length) {
    const c = S.config;
    S.decStep = E.decodeLastPosition(S.decTrace, {
      temperature: c.temperature, method: c.method, k: c.k, p: c.p,
      sampleSeed: (c.seed + S.sampleCounter * 1013904223) >>> 0,
    });
  } else S.decStep = null;
}

function recomputeAll() {
  const err = configError();
  $("configError").textContent = err || "";
  $("configError").classList.toggle("show", !!err);
  if (err) return false;
  recomputeEncoder();
  recomputeDecoder();
  S.selQuery = Math.min(S.selQuery, Math.max(0, (S.mode === "encoder" ? S.encTrace.ids.length : S.decIds.length) - 1));
  S.selHead = Math.min(S.selHead, S.config.nHeads - 1);
  S.selBlock = Math.min(S.selBlock, S.config.nLayers - 1);
  return true;
}

/* ---------------- Generation ---------------- */
function nextToken() {
  if (!S.decStep || S.decIds.length >= 16) return;
  const st = S.decStep, c = S.config;
  const id = st.sampling.selectedId;
  const tok = S.decTrace.vocab[id];
  S.history.push({
    step: S.history.length + 1,
    contextIds: [...S.decIds], contextTokens: [...S.decTokens],
    logitsLast: [...st.logitsLast], probs: [...st.probs],
    selectedId: id, selectedToken: tok,
    method: c.method, k: c.k, p: c.p, temperature: c.temperature,
  });
  S.decIds.push(id); S.decTokens.push(tok);
  S.sampleCounter++;
  S.selHist = S.history.length - 1;
  S.selQuery = S.decIds.length - 1;
  recomputeDecoder();
  renderDynamic();
}

function stepBack() {
  if (!S.history.length) return;
  S.history.pop();
  S.decIds.pop(); S.decTokens.pop();
  S.sampleCounter = Math.max(0, S.sampleCounter - 1);
  S.selHist = S.history.length - 1;
  recomputeDecoder();
  renderDynamic();
}

function setAutoGen(on) {
  S.autoGen = on;
  clearInterval(S.autoTimer);
  if (on) {
    S.autoTimer = setInterval(() => {
      if (S.decIds.length >= 16) { setAutoGen(false); renderDynamic(); return; }
      nextToken();
    }, 1500);
  }
}

/* ---------------- Config bar ---------------- */
function selectHTML(key, options, label) {
  return `<div class="cfg"><label for="cfg_${key}">${label}</label>
    <select id="cfg_${key}" data-cfg="${key}">${options.map((o) =>
      `<option value="${o}"${S.config[key] === o ? " selected" : ""}>${o}</option>`).join("")}</select></div>`;
}

function renderConfig() {
  const c = S.config;
  $("configBar").innerHTML = `
    ${selectHTML("dModel", [4, 8, 16], "d_model")}
    ${selectHTML("dFF", [8, 16, 32], "d_ff")}
    ${selectHTML("nHeads", [1, 2, 4], "heads")}
    ${selectHTML("nLayers", [1, 2, 4], "blocks")}
    ${selectHTML("vocabSize", [16, 32, 64], "vocab")}
    <div class="cfg"><label for="cfg_temperature">temperature</label>
      <span><input type="range" id="cfg_temperature" min="0.1" max="2" step="0.05" value="${c.temperature}" data-cfg-range="temperature" aria-label="Temperature" />
      <span class="val" id="tempVal">${c.temperature.toFixed(2)}</span></span></div>
    <div class="cfg"><label for="cfg_method">sampling</label>
      <select id="cfg_method" data-cfg="method">${["greedy", "top-k", "top-p"].map((o) =>
        `<option${c.method === o ? " selected" : ""}>${o}</option>`).join("")}</select></div>
    ${c.method === "top-k" ? selectHTML("k", [1, 3, 5, 10], "top-k") : ""}
    ${c.method === "top-p" ? `<div class="cfg"><label for="cfg_p">top-p</label>
      <span><input type="range" id="cfg_p" min="0.1" max="1" step="0.05" value="${c.p}" data-cfg-range="p" aria-label="Top-p" />
      <span class="val" id="pVal">${c.p.toFixed(2)}</span></span></div>` : ""}
    ${selectHTML("precision", [2, 3, 4], "decimals")}
    <div class="cfg"><label>&nbsp;</label><span class="stage-sub">deterministic · seed ${c.seed}</span></div>`;
}

/* ---------------- Pipeline + stage chrome ---------------- */
const STAGE_DEFAULT_TENSOR = {
  tokenize: "ids", embed: "X", pos: "Xpos", qkv: "Q", heads: "Qh", attn: "W", concat: "proj", ffn: "FFNout",
  dtok: "dids", demb: "dXpos", dln1: "ln1", dqkv: "dQh", dmask: "dW", dres1: "dresid1",
  dffn: "dblockOut", dblocks: "dblockOut", dfinal: "finalLN", dlm: "logits", dtemp: "probs", dsample: "probs",
};

function renderPipe() {
  const st = stages();
  $("pipeList").innerHTML = st.map((s, i) =>
    `<button class="pipe-item${i === S.stage ? " active" : ""}${i < S.stage ? " done" : ""}" data-stage="${i}" aria-current="${i === S.stage}">
      <span class="pipe-num">${i < S.stage ? "✓" : i + 1}</span>
      <span><span class="t">${esc(s.nav)}</span><span class="s">${esc(s.shape)}</span></span>
    </button>`).join("");
  $("stagePos").textContent = `Stage ${S.stage + 1} / ${st.length}`;
  $("progressDots").innerHTML = st.map((_, i) => `<span class="${i <= S.stage ? "on" : ""}"></span>`).join("");
}

function genBarHTML() {
  if (S.mode !== "decoder") return "";
  const T = S.decTrace;
  return `<div class="controls-row" style="background:#f6f8fb;border:1px solid var(--line);border-radius:8px;padding:7px 10px">
    <span class="stage-sub"><b>Context (${T ? T.ids.length : 0}/16):</b> ${T ? esc(T.tokens.join(" ")) : ""}</span>
    <span style="flex:1"></span>
    <button class="btn small primary" data-action="nextToken">Next Token →</button>
    <button class="btn small" data-action="autoGen">${S.autoGen ? "Pause" : "Auto"}</button>
    <button class="btn small" data-action="stepBack" ${S.history.length ? "" : "disabled"}>Back</button>
    <button class="btn small ghost" data-action="resetGen">Reset</button>
  </div>`;
}

function encInputBarHTML() {
  if (S.mode !== "encoder" || stages()[S.stage].id === "tokenize") return "";
  return `<div class="controls-row"><input class="input-text" id="encInputMini" style="min-height:0" value="${esc(S.encInput)}" aria-label="Input text" />
    <button class="btn small" data-action="applyEncInput">Run</button></div>`;
}

function renderStage() {
  const st = stages()[S.stage];
  $("stageTitle").textContent = `${S.stage + 1}. ${st.title}`;
  $("stageSub").textContent = `${st.sub} · ${st.shape}`;
  const body = S.mode === "encoder" ? renderEncoderStage(st.id, S) : renderDecoderStage(st.id, S);
  $("stageView").innerHTML = genBarHTML() + encInputBarHTML() +
    (S.truncated && S.mode === "encoder" ? `<div class="simplify-note"><b>Viz cap ·</b> Showing the first 12 tokens so matrices stay inspectable.</div>` : "") + body;
  wireGelu();
}

function mathDetailHTML() {
  const st = stages()[S.stage];
  const c = S.config, P = c.precision;
  const back = (inner) => `<div class="calc-box">${inner}</div>`;
  try {
    if (S.mode === "encoder") {
      const T = S.encTrace;
      if (!T || !T.ids.length) return "";
      if (st.id === "tokenize") return back(T.tokens.map((t) => `FNV-1a(“${esc(t)}”) = ${E.fnv1a(t)} → mod ${c.vocabSize} = <b>${T.ids[T.tokens.indexOf(t)]}</b>`).join("<br>"));
      if (st.id === "attn") {
        const a = T.enc.attn[S.selHead], q = Math.min(S.selQuery, T.ids.length - 1);
        const exps = a.scaled[q].map((x) => Math.exp(x - Math.max(...a.scaled[q])));
        const sum = exps.reduce((x, y) => x + y, 0);
        return back(`row q=${q} (head ${S.selHead + 1}):<br>scaled=[${a.scaled[q].map((x) => fmt(x, 3)).join(", ")}]<br>exp=[${exps.map((x) => fmt(x, 3)).join(", ")}] Σ=${fmt(sum, 4)}<br>weights=[${a.weights[q].map((x) => fmt(x, P)).join(", ")}] Σ=<b>${fmt(a.weights[q].reduce((x, y) => x + y, 0), 4)}</b>`);
      }
      if (st.id === "qkv" || st.id === "concat") {
        const w = T.weights.layers[0];
        return back(`Toy weights: seeded Gaussian × √(1/d) from seed ${c.seed} (stream “L0|W${S.qkvTab || "O"}”).<br>Real weights are <b>learned in training</b>; the seed only makes this demo reproducible.`);
      }
    } else {
      const T = S.decTrace;
      if (!T || !T.ids.length) return "";
      const b = T.blocks[Math.min(S.selBlock, T.blocks.length - 1)];
      if (st.id === "dln1" || st.id === "dfinal") {
        const ln = st.id === "dln1" ? b.ln1 : T.finalLN;
        const i = Math.min(S.selQuery, T.ids.length - 1);
        return back(`token “${esc(T.tokens[i])}”: μ=${fmt(ln.means[i])}, σ²=${fmt(ln.variances[i])} → normalized row mean≈0, var≈1, then ×γ(1)+β(0).`);
      }
      if (st.id === "dmask") {
        const a = b.attn[S.selHead];
        const leaked = a.weights.map((r, i) => r.slice(i + 1).reduce((x, y) => x + y, 0));
        return back(`Future-mass per query row after softmax: [${leaked.map((x) => fmt(x, 6)).join(", ")}] — all <b>exactly 0</b> ✓ (mask applied <i>before</i> softmax as −∞).`);
      }
      if (st.id === "dtemp" && S.decStep) {
        const sum = S.decStep.probs.reduce((x, y) => x + y, 0);
        return back(`softmax(z/T): max logit ${fmt(Math.max(...S.decStep.logitsLast))} → T=${c.temperature.toFixed(2)} → <b>Σp=${fmt(sum, 6)}</b> ✓`);
      }
      if (st.id === "dsample" && S.decStep) {
        const sm = S.decStep.sampling;
        return back(`${esc(c.method)}: ${sm.candidates.filter((x) => x.kept).length} eligible → sampled id <b>${sm.selectedId}</b> (“${esc(T.vocab[sm.selectedId])}”) · sampling seed = ${c.seed} ⊕ step ${S.sampleCounter}.`);
      }
    }
  } catch { return ""; }
  return "";
}

function renderExplain() {
  const e = explanations()[S.stage];
  $("explainBox").innerHTML = `
    <h3>What happens?</h3><p>${esc(e.what)}</p>
    <h3>Why?</h3><p>${esc(e.why)}</p>
    <h3>Shape</h3><div class="eq-block">${esc(e.shape)}</div>
    ${e.equation ? `<h3>Key equation</h3><div class="eq-block">${esc(e.equation)}</div>` : ""}
    <div class="controls-row">
      <button class="link-btn" data-action="explainMore">${S.showMore ? "Hide explanation ▲" : "Explain this ▸"}</button>
      <button class="link-btn" data-action="explainMath">${S.showMath ? "Hide the math ▲" : "Show me the math ▸"}</button>
    </div>
    <div class="more${S.showMore ? " show" : ""}">${esc(e.more)}</div>
    ${S.showMath ? mathDetailHTML() : ""}`;
}

/* ---------------- Tensor inspector ---------------- */
function currentTensors() {
  const c = S.config, T = S.mode === "encoder" ? S.encTrace : S.decTrace;
  if (!T || !T.ids.length) return [];
  const dH = (n) => Array.from({ length: n }, (_, j) => `h${j + 1}`);
  const dc = Array.from({ length: c.dModel }, (_, j) => `d${j + 1}`);
  const df = Array.from({ length: c.dFF }, (_, j) => `f${j + 1}`);
  if (S.mode === "encoder") {
    const e = T.enc, w = T.weights.layers[0], h = Math.min(S.selHead, c.nHeads - 1), a = e.attn[h];
    return [
      { key: "ids", label: "Token IDs", M: T.ids.map((id) => [id]), rowLabels: T.tokens, colLabels: ["id"] },
      { key: "X", label: "Embedding X", M: T.X, rowLabels: T.tokens, colLabels: dc },
      { key: "PE", label: "Positional encoding", M: T.PE, rowLabels: T.tokens, colLabels: dc },
      { key: "Xpos", label: "X + PE", M: T.Xpos, rowLabels: T.tokens, colLabels: dc },
      { key: "WQ", label: "W_Q", M: w.WQ, rowLabels: dc, colLabels: dc },
      { key: "Q", label: "Q", M: e.Q, rowLabels: T.tokens, colLabels: dc },
      { key: "WK", label: "W_K", M: w.WK, rowLabels: dc, colLabels: dc },
      { key: "K", label: "K", M: e.K, rowLabels: T.tokens, colLabels: dc },
      { key: "WV", label: "W_V", M: w.WV, rowLabels: dc, colLabels: dc },
      { key: "V", label: "V", M: e.V, rowLabels: T.tokens, colLabels: dc },
      { key: "Qh", label: `Q head ${h + 1}`, M: e.Qh[h], rowLabels: T.tokens, colLabels: dH(e.dHead) },
      { key: "Kh", label: `K head ${h + 1}`, M: e.Kh[h], rowLabels: T.tokens, colLabels: dH(e.dHead) },
      { key: "S", label: `Scores QKᵀ (h${h + 1})`, M: a.scores, rowLabels: T.tokens, colLabels: T.tokens },
      { key: "Ssc", label: `Scaled ÷√${e.dHead} (h${h + 1})`, M: a.scaled, rowLabels: T.tokens, colLabels: T.tokens },
      { key: "W", label: `Attention weights (h${h + 1})`, M: a.weights, rowLabels: T.tokens, colLabels: T.tokens },
      { key: "C", label: `Context out (h${h + 1})`, M: a.output, rowLabels: T.tokens, colLabels: dH(e.dHead) },
      { key: "concat", label: "Concat heads", M: e.concat, rowLabels: T.tokens, colLabels: dc },
      { key: "WO", label: "W_O", M: w.WO, rowLabels: dc, colLabels: dc },
      { key: "proj", label: "Attention output", M: e.proj, rowLabels: T.tokens, colLabels: dc },
      { key: "W1", label: "FFN W₁", M: w.W1, rowLabels: dc, colLabels: df },
      { key: "hidPre", label: "FFN hidden pre-GELU", M: e.ffn.hiddenPre, rowLabels: T.tokens, colLabels: df },
      { key: "hidAct", label: "FFN hidden GELU", M: e.ffn.hiddenAct, rowLabels: T.tokens, colLabels: df },
      { key: "W2", label: "FFN W₂", M: w.W2, rowLabels: df, colLabels: dc },
      { key: "FFNout", label: "FFN output (final)", M: e.ffn.output, rowLabels: T.tokens, colLabels: dc },
    ];
  }
  const l = Math.min(S.selBlock, T.blocks.length - 1), b = T.blocks[l];
  const h = Math.min(S.selHead, c.nHeads - 1), a = b.attn[h];
  const last = T.ids.length - 1;
  const list = [
    { key: "dids", label: "Context IDs", M: T.ids.map((id) => [id]), rowLabels: T.tokens, colLabels: ["id"] },
    { key: "dXpos", label: "X + PE", M: T.Xpos, rowLabels: T.tokens, colLabels: dc },
    { key: "ln1", label: `Block ${l + 1} · LN₁ out`, M: b.ln1.output, rowLabels: T.tokens, colLabels: dc },
    { key: "dQh", label: `Block ${l + 1} · Q head ${h + 1}`, M: b.Qh[h], rowLabels: T.tokens, colLabels: dH(b.dHead) },
    { key: "dKh", label: `Block ${l + 1} · K head ${h + 1}`, M: b.Kh[h], rowLabels: T.tokens, colLabels: dH(b.dHead) },
    { key: "dS", label: `Block ${l + 1} · scores (h${h + 1})`, M: a.scores, rowLabels: T.tokens, colLabels: T.tokens },
    { key: "dM", label: `Block ${l + 1} · masked (h${h + 1})`, M: a.masked, rowLabels: T.tokens, colLabels: T.tokens },
    { key: "dW", label: `Block ${l + 1} · attn weights (h${h + 1})`, M: a.weights, rowLabels: T.tokens, colLabels: T.tokens },
    { key: "dC", label: `Block ${l + 1} · ctx out (h${h + 1})`, M: a.output, rowLabels: T.tokens, colLabels: dH(b.dHead) },
    { key: "dconcat", label: `Block ${l + 1} · concat`, M: b.concat, rowLabels: T.tokens, colLabels: dc },
    { key: "dproj", label: `Block ${l + 1} · proj (W_O)`, M: b.proj, rowLabels: T.tokens, colLabels: dc },
    { key: "dresid1", label: `Block ${l + 1} · resid₁ x+proj`, M: b.resid1, rowLabels: T.tokens, colLabels: dc },
    { key: "ln2", label: `Block ${l + 1} · LN₂ out`, M: b.ln2.output, rowLabels: T.tokens, colLabels: dc },
    { key: "dhid", label: `Block ${l + 1} · FFN GELU hidden`, M: b.ffn.hiddenAct, rowLabels: T.tokens, colLabels: df },
    { key: "dffn", label: `Block ${l + 1} · FFN out`, M: b.ffn.output, rowLabels: T.tokens, colLabels: dc },
    { key: "dblockOut", label: `Block ${l + 1} · output x″`, M: b.output, rowLabels: T.tokens, colLabels: dc },
    { key: "finalLN", label: "Final LayerNorm out", M: T.finalLN.output, rowLabels: T.tokens, colLabels: dc },
    { key: "logits", label: "Logits (last row = next-token scores)", M: T.logits, rowLabels: T.tokens, colLabels: T.vocab, maxCols: 16 },
  ];
  if (S.decStep) {
    list.push({ key: "probs", label: "Probabilities (last position)", vec: S.decStep.probs, vocab: T.vocab });
  }
  return list;
}

function renderInspector() {
  const tensors = currentTensors();
  if (!tensors.find((t) => t.key === S.selTensor)) {
    S.selTensor = tensors.length ? tensors[0].key : null;
  }
  $("tensorList").innerHTML = tensors.map((t) => {
    const sh = t.M ? `[${t.M.length}, ${t.M[0].length}]` : `[${t.vec.length}]`;
    return `<button class="tensor-btn${t.key === S.selTensor ? " active" : ""}" data-tensor="${t.key}" role="option" aria-selected="${t.key === S.selTensor}">${esc(t.label)}<span class="sh">${sh}</span></button>`;
  }).join("") || `<span class="stage-sub">No tensors yet.</span>`;
  const t = tensors.find((x) => x.key === S.selTensor);
  if (!t) { $("tensorDetail").innerHTML = ""; return; }
  const P = S.config.precision;
  if (t.vec) {
    const s = E.vecStats(t.vec);
    const order = t.vec.map((p, id) => ({ id, p })).sort((a, b) => b.p - a.p).slice(0, 12);
    $("tensorDetail").innerHTML = `
      <div><b>${esc(t.label)}</b> ${shapeBadge([t.vec.length])} <span class="row-sum">Σ = ${fmt(t.vec.reduce((a, b) => a + b, 0), 4)}</span></div>
      <div class="insp-grid">
        <div class="insp-stat"><div class="k">Min</div><div class="v">${fmt(s.min)}</div></div>
        <div class="insp-stat"><div class="k">Max</div><div class="v">${fmt(s.max)}</div></div>
        <div class="insp-stat"><div class="k">Mean</div><div class="v">${fmt(s.mean)}</div></div>
      </div>
      <div class="stage-sub">Top 12 of ${t.vec.length}:</div>
      ${barsHTML(order.map((d) => ({ label: `${t.vocab[d.id]} (id ${d.id})`, value: d.p, kept: true })), { precision: P })}`;
  } else {
    $("tensorDetail").innerHTML = `
      <div><b>${esc(t.label)}</b> ${shapeBadge([t.M.length, t.M[0].length])}</div>
      ${statsGrid(t.label, t.M)}
      ${matrixHTML(t.M, { rowLabels: t.rowLabels, colLabels: t.colLabels, precision: P, maxCols: t.maxCols ?? 16, maxRows: 16 })}`;
  }
}

/* ---------------- Render orchestration ---------------- */
function renderDynamic() {
  if (!recomputeAll()) { renderPipe(); return; }
  renderPipe();
  renderStage();
  renderExplain();
  renderInspector();
  $("tempVal") && ($("tempVal").textContent = S.config.temperature.toFixed(2));
  $("pVal") && ($("pVal").textContent = S.config.p.toFixed(2));
}

function gotoStage(i) {
  const st = stages();
  S.stage = Math.max(0, Math.min(st.length - 1, i));
  S.showMore = false; S.showMath = false;
  S.selTensor = STAGE_DEFAULT_TENSOR[st[S.stage].id] || S.selTensor;
  stopPlay();
  renderPipe(); renderStage(); renderExplain(); renderInspector();
}

function setMode(m) {
  if (S.mode === m) return;
  S.mode = m; S.stage = 0; S.showMore = false; S.showMath = false;
  setAutoGen(false); stopPlay();
  $("tabEncoder").classList.toggle("active", m === "encoder");
  $("tabDecoder").classList.toggle("active", m === "decoder");
  $("tabEncoder").setAttribute("aria-pressed", m === "encoder");
  $("tabDecoder").setAttribute("aria-pressed", m === "decoder");
  S.selTensor = STAGE_DEFAULT_TENSOR[stages()[0].id];
  renderDynamic();
}

function stopPlay() {
  S.playing = false;
  clearInterval(S.playTimer);
  $("playBtn").textContent = "▶ Play";
}

function togglePlay() {
  if (S.playing) { stopPlay(); return; }
  S.playing = true;
  $("playBtn").textContent = "⏸ Pause";
  S.playTimer = setInterval(() => {
    if (S.stage >= stages().length - 1) { stopPlay(); return; }
    S.stage++;
    S.selTensor = STAGE_DEFAULT_TENSOR[stages()[S.stage].id] || S.selTensor;
    renderPipe(); renderStage(); renderExplain(); renderInspector();
  }, 2400);
}

/* ---------------- GELU hover ---------------- */
function wireGelu() {
  const svg = $("geluPlot");
  if (!svg) return;
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  const W = 460, P = 28, minX = -3, maxX = 3;
  const dot = $("geluDot"), out = $("geluReadout");
  svg.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const x = minX + ((Math.min(W - P, Math.max(P, px)) - P) / (W - 2 * P)) * (maxX - minX);
    const ys = [];
    for (let i = 0; i <= 120; i++) ys.push(gelu(minX + ((maxX - minX) * i) / 120));
    const minY = Math.min(...ys), maxY = Math.max(...ys), H = 200;
    const X = (v) => P + ((v - minX) / (maxX - minX)) * (W - 2 * P);
    const Y = (v) => H - P - ((v - minY) / (maxY - minY)) * (H - 2 * P);
    dot.setAttribute("cx", X(x)); dot.setAttribute("cy", Y(gelu(x))); dot.style.display = "block";
    out.textContent = `x=${fmt(x, 3)} → GELU=${fmt(gelu(x), 4)}`;
  });
  svg.addEventListener("mouseleave", () => { dot.style.display = "none"; });
}

/* ---------------- Modal ---------------- */
function openModal(title, bodyHTML) {
  $("modalTitle").textContent = title;
  $("modalBody").innerHTML = bodyHTML;
  $("modalBack").classList.add("show");
}
function showAbout() {
  openModal("About this simulator", `<p>${esc(ABOUT_TEXT).replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br>")}</p>
    <p><b>What went in? What operation happened? What came out? What is its shape? Why does it exist?</b> — every stage answers these five questions. Priority: <b>correctness → clarity → consistency → interaction → aesthetics</b>.</p>
    <p>Keyboard: ←/→ move between stages · N generates the next token in Mode B.</p>`);
}
function showTests() {
  const res = runTests();
  const pass = res.filter((r) => r.pass).length;
  openModal(`Engine tests — ${pass}/${res.length} passing`,
    res.map((r) => `<div class="test-row"><span class="${r.pass ? "pass" : "fail"}">${r.pass ? "PASS" : "FAIL"}</span><span>${esc(r.name)}${r.error ? ` — <i>${esc(r.error)}</i>` : ""}</span></div>`).join(""));
}

/* ---------------- Events ---------------- */
function onAction(action, el) {
  const v = el.dataset.v, i = el.dataset.i !== undefined ? +el.dataset.i : null;
  switch (action) {
    case "embView": S.embView = v; break;
    case "qkvTab": S.qkvTab = v; break;
    case "selHead": S.selHead = i; break;
    case "selBlock": S.selBlock = i; break;
    case "selQuery": S.selQuery = i; break;
    case "attnView": S.attnView = v; break;
    case "dAttnView": S.dAttnView = v; break;
    case "ffnView": S.ffnView = v; break;
    case "dffnView": S.dffnView = v; break;
    case "resView": S.resView = v; break;
    case "toggleCalc": S.showCalc = !S.showCalc; break;
    case "toggleMask": S.showMask = !S.showMask; break;
    case "topN": S.topN = i; break;
    case "method": S.config.method = v; renderConfig(); break;
    case "k": S.config.k = i; break;
    case "inspect": S.selTensor = el.dataset.tensor; renderInspector(); document.querySelector(".bottom-bar").scrollIntoView({ behavior: "smooth", block: "nearest" }); return;
    case "explainMore": S.showMore = !S.showMore; renderExplain(); return;
    case "explainMath": S.showMath = !S.showMath; renderExplain(); return;
    case "nextToken": setAutoGen(false); nextToken(); return;
    case "autoGen": setAutoGen(!S.autoGen); break;
    case "stepBack": setAutoGen(false); stepBack(); return;
    case "resetGen": setAutoGen(false); S.decPrompt = ($("decInput")?.value ?? S.decPrompt); resetDecoder(); break;
    case "selHist": S.selHist = i; break;
    case "applyEncInput": S.encInput = $("encInputMini").value; break;
    default: return;
  }
  renderDynamic();
}

let encDebounce = null;
function initEvents() {
  document.addEventListener("click", (ev) => {
    const st = ev.target.closest("[data-stage]");
    if (st) { gotoStage(+st.dataset.stage); return; }
    const tb = ev.target.closest("[data-tensor]");
    if (tb) { S.selTensor = tb.dataset.tensor; renderInspector(); return; }
    const a = ev.target.closest("[data-action]");
    if (a) { onAction(a.dataset.action, a); return; }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && ev.target.matches?.('[role="button"][data-action]')) { ev.target.click(); return; }
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
    if (typing) return;
    if (ev.key === "ArrowRight") gotoStage(S.stage + 1);
    else if (ev.key === "ArrowLeft") gotoStage(S.stage - 1);
    else if ((ev.key === "n" || ev.key === "N") && S.mode === "decoder") { setAutoGen(false); nextToken(); }
  });
  document.addEventListener("input", (ev) => {
    const el = ev.target;
    if (el.id === "encInput") {
      S.encInput = el.value;
      clearTimeout(encDebounce);
      encDebounce = setTimeout(renderDynamic, 500);
    } else if (el.id === "encInputMini") {
      S.encInput = el.value;
      clearTimeout(encDebounce);
      encDebounce = setTimeout(() => { renderDynamic(); }, 600);
    } else if (el.id === "decInput") {
      S.decPrompt = el.value; // applied on Reset
    } else if (el.dataset.cfgRange) {
      S.config[el.dataset.cfgRange] = +el.value;
      if (el.dataset.cfgRange === "temperature") { const t2 = $("tempSlider2"); if (t2 && t2 !== el) t2.value = el.value; }
      renderDynamic();
    } else if (el.dataset.action === "temp") {
      S.config.temperature = +el.value; renderDynamic();
    } else if (el.dataset.action === "p") {
      S.config.p = +el.value; renderDynamic();
    }
  });
  document.addEventListener("change", (ev) => {
    const el = ev.target;
    if (el.dataset.cfg) {
      const key = el.dataset.cfg;
      S.config[key] = key === "method" ? el.value : +el.value;
      if (["dModel", "dFF", "nHeads", "nLayers", "vocabSize"].includes(key)) { setAutoGen(false); resetDecoder(); }
      renderConfig();
      renderDynamic();
    } else if (el.id === "seedInput") {
      S.config.seed = Math.max(0, +el.value || 0);
      setAutoGen(false); resetDecoder(); renderConfig(); renderDynamic();
    } else if (el.id === "encInput" || el.id === "encInputMini") {
      S.encInput = el.value; renderDynamic();
    }
  });
  $("prevBtn").onclick = () => gotoStage(S.stage - 1);
  $("nextBtn").onclick = () => gotoStage(S.stage + 1);
  $("playBtn").onclick = togglePlay;
  $("tabEncoder").onclick = () => setMode("encoder");
  $("tabDecoder").onclick = () => setMode("decoder");
  $("aboutBtn").onclick = showAbout;
  $("testsBtn").onclick = showTests;
  $("modalClose").onclick = () => $("modalBack").classList.remove("show");
  $("modalBack").addEventListener("click", (ev) => { if (ev.target === $("modalBack")) $("modalBack").classList.remove("show"); });
}

/* ---------------- Init ---------------- */
renderConfig();
resetDecoder();
recomputeEncoder();
S.selTensor = STAGE_DEFAULT_TENSOR[stages()[0].id];
initEvents();
renderDynamic();
