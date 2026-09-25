/* ============================================================
 * Tiny Transformer — Pure Mathematical Engine (no DOM, no UI)
 * Deterministic, seeded, independently testable.
 * All functions are pure: same input => same output.
 * ============================================================ */

/** FNV-1a 32-bit hash. Known test vector: fnv1a("hello") = 1335831723. */
export function fnv1a(str) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** Seeded PRNG (mulberry32). Returns function () => [0,1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** RNG stream derived from master seed + salt string (independent streams). */
export function streamRng(seed, salt) {
  return mulberry32(((seed >>> 0) ^ fnv1a(String(salt))) >>> 0);
}

/** Standard normal sample via Box–Muller using provided rng. */
export function randn(rng) {
  let u1 = 0;
  while (u1 === 0) u1 = rng();
  const u2 = rng();
  const r = Math.sqrt(-2.0 * Math.log(u1));
  const theta = 2.0 * Math.PI * u2;
  return r * Math.cos(theta);
}

/* ---------------- Vocabulary ---------------- */

export const BASE_VOCAB = [
  "The", "cat", "sat", "on", "mat", "dog", "ran", "a",
  "the", "and", "is", "was", "it", "to", "in", "he",
  "she", "we", "they", "you", "I", "man", "woman", "child",
  "bird", "fish", "tree", "house", "car", "book", "sun", "moon",
  "star", "sky", "sea", "fire", "water", "earth", "wind", "day",
  "night", "light", "dark", "big", "small", "red", "blue", "green",
  "happy", "sad", "fast", "slow", "up", "down", "over", "under",
  "with", "at", "of", "for", "from", "that", "this", "not"
];

export function buildVocab(vocabSize) {
  const v = [];
  for (let i = 0; i < vocabSize; i++) {
    v.push(i < BASE_VOCAB.length ? BASE_VOCAB[i] : `tok${i}`);
  }
  return v;
}

/* ---------------- Tokenizer (EDUCATIONAL SIMPLIFICATION) ----------------
 * Toy tokenizer: whitespace split + FNV-1a hash mod vocab_size.
 * Real GPT-family tokenizers use learned subword schemes (e.g. BPE).
 */
export function tokenize(text, vocabSize) {
  const tokens = String(text ?? "").trim().split(/\s+/).filter(Boolean);
  const ids = tokens.map((t) => fnv1a(t) % vocabSize);
  return { tokens, ids };
}

export function detokenize(id, vocab) {
  return vocab[id] ?? `<${id}>`;
}

/* ---------------- Tensor helpers ---------------- */

export function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

export function createMatrix(rows, cols, fn) {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => fn(r, c))
  );
}

export function shapeOf(m) {
  if (!Array.isArray(m)) return [];
  if (m.length && Array.isArray(m[0])) return [m.length, m[0].length];
  return [m.length];
}

export function matmul(A, B) {
  const m = A.length, k = A[0].length, n = B[0].length;
  if (B.length !== k) throw new Error(`matmul shape mismatch: [${m},${k}] x [${B.length},${n}]`);
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      for (let j = 0; j < n; j++) C[i][j] += a * B[p][j];
    }
  }
  return C;
}

export function transpose(M) {
  const r = M.length, c = M[0].length;
  return createMatrix(c, r, (i, j) => M[j][i]);
}

export function addMatrices(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

export function scaleMatrix(M, s) {
  return M.map((row) => row.map((v) => v * s));
}

export function addBiasRows(M, b) {
  return M.map((row) => row.map((v, j) => v + b[j]));
}

/** Y = X @ W (+ b). X:[seq,din] W:[din,dout] */
export function linear(X, W, b = null) {
  const Y = matmul(X, W);
  return b ? addBiasRows(Y, b) : Y;
}

/* ---------------- Embeddings (toy deterministic weights) ---------------- */

export function createEmbedding(vocabSize, dModel, seed) {
  const rng = streamRng(seed, `embed|v=${vocabSize}|d=${dModel}`);
  return createMatrix(vocabSize, dModel, () => randn(rng) * 0.6);
}

export function embedLookup(ids, embeddingMatrix) {
  return ids.map((id) => [...embeddingMatrix[id]]);
}

/* ---------------- Sinusoidal positional encoding (Vaswani et al. 2017) ---------------- */

export function createSinusoidalPositionEncoding(seqLen, dModel) {
  return createMatrix(seqLen, dModel, (pos, i) => {
    const div = Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
    return i % 2 === 0 ? Math.sin(pos / div) : Math.cos(pos / div);
  });
}

/* ---------------- Model weights (toy deterministic) ---------------- */

function gaussianMatrix(rows, cols, rng, scale) {
  return createMatrix(rows, cols, () => randn(rng) * scale);
}

export function createWeights(config, seed) {
  const { dModel, dFF, nLayers, vocabSize } = config;
  const W = { embedding: createEmbedding(vocabSize, dModel, seed), layers: [] };
  for (let l = 0; l < nLayers; l++) {
    const s = (name) => streamRng(seed, `L${l}|${name}|d=${dModel}|dff=${dFF}|v=${vocabSize}`);
    const sc = Math.sqrt(1 / dModel);
    W.layers.push({
      WQ: gaussianMatrix(dModel, dModel, s("WQ"), sc),
      WK: gaussianMatrix(dModel, dModel, s("WK"), sc),
      WVO: null, // placeholder replaced below (kept for clarity)
      WV: gaussianMatrix(dModel, dModel, s("WV"), sc),
      WO: gaussianMatrix(dModel, dModel, s("WO"), sc),
      ln1Gamma: new Array(dModel).fill(1),
      ln1Beta: new Array(dModel).fill(0),
      W1: gaussianMatrix(dModel, dFF, s("W1"), Math.sqrt(1 / dModel)),
      b1: new Array(dFF).fill(0),
      W2: gaussianMatrix(dFF, dModel, s("W2"), Math.sqrt(1 / dFF)),
      b2: new Array(dModel).fill(0),
      ln2Gamma: new Array(dModel).fill(1),
      ln2Beta: new Array(dModel).fill(0),
    });
    delete W.layers[l].WVO;
  }
  W.finalGamma = new Array(dModel).fill(1);
  W.finalBeta = new Array(dModel).fill(0);
  W.lmHead = gaussianMatrix(dModel, vocabSize, streamRng(seed, `lmHead|d=${dModel}|v=${vocabSize}`), Math.sqrt(1 / dModel));
  return W;
}

/* ---------------- Attention ---------------- */

/** [seq,dModel] -> array of nHeads x [seq,dHead] */
export function splitHeads(X, nHeads) {
  const seq = X.length, dModel = X[0].length;
  if (dModel % nHeads !== 0) throw new Error(`d_model (${dModel}) must be divisible by n_heads (${nHeads})`);
  const dHead = dModel / nHeads;
  const heads = [];
  for (let h = 0; h < nHeads; h++) {
    heads.push(X.map((row) => row.slice(h * dHead, (h + 1) * dHead)));
  }
  return { heads, dHead };
}

/** array of [seq,dHead] -> [seq,dModel] */
export function concatHeads(heads) {
  const seq = heads[0].length;
  const out = [];
  for (let i = 0; i < seq; i++) {
    const row = [];
    for (const h of heads) row.push(...h[i]);
    out.push(row);
  }
  return out;
}

export function softmaxVec(v) {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

export function softmax(M) {
  return M.map(softmaxVec);
}

/** Set future positions (j > i) to -Infinity equivalent (-1e9). */
export function applyCausalMask(scores) {
  return scores.map((row, i) => row.map((v, j) => (j > i ? -1e9 : v)));
}

export function scaledDotProductAttention(Q, K, V, { causal = false } = {}) {
  const dHead = Q[0].length;
  const scores = matmul(Q, transpose(K));              // [seq,seq]
  const scaled = scaleMatrix(scores, 1 / Math.sqrt(dHead));
  const masked = causal ? applyCausalMask(scaled) : scaled;
  const weights = softmax(masked);                     // rows sum to 1
  const output = matmul(weights, V);                   // [seq,dHead]
  return { scores, scaled, masked, weights, output, dHead };
}

/* ---------------- Normalization ---------------- */

/** LayerNorm over last dim, per row. Returns normalized-before-affine too. */
export function layerNorm(X, gamma, beta, eps = 1e-5) {
  const d = X[0].length;
  const means = [], variances = [], normalized = [];
  const output = X.map((row) => {
    const mean = row.reduce((a, b) => a + b, 0) / d;
    const varr = row.reduce((a, b) => a + (b - mean) ** 2, 0) / d;
    means.push(mean); variances.push(varr);
    const nrow = row.map((v) => (v - mean) / Math.sqrt(varr + eps));
    normalized.push(nrow);
    return nrow.map((v, j) => v * gamma[j] + beta[j]);
  });
  return { output, means, variances, normalized };
}

/* ---------------- FFN ---------------- */

/** GELU (tanh approximation, Hendrycks & Gimpel). */
export function gelu(x) {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

export function geluMatrix(M) {
  return M.map((row) => row.map(gelu));
}

/** FFN(x) = GELU(xW1+b1)W2+b2 */
export function feedForward(X, W1, b1, W2, b2) {
  const hiddenPre = addBiasRows(matmul(X, W1), b1);  // [seq,dFF]
  const hiddenAct = geluMatrix(hiddenPre);
  const output = addBiasRows(matmul(hiddenAct, W2), b2);
  return { hiddenPre, hiddenAct, output };
}

/* ---------------- Transformer blocks ---------------- */

/** Pre-LN decoder block with causal attention + 2 residuals. Full trace. */
export function transformerBlock(X, w, nHeads, { causal = true } = {}) {
  const ln1 = layerNorm(X, w.ln1Gamma, w.ln1Beta);
  const Q = linear(ln1.output, w.WQ);
  const K = linear(ln1.output, w.WK);
  const V = linear(ln1.output, w.WV);
  const { heads: Qh, dHead } = splitHeads(Q, nHeads);
  const { heads: Kh } = splitHeads(K, nHeads);
  const { heads: Vh } = splitHeads(V, nHeads);
  const attn = Qh.map((q, h) => scaledDotProductAttention(q, Kh[h], Vh[h], { causal }));
  const ctxHeads = attn.map((a) => a.output);
  const concat = concatHeads(ctxHeads);
  const proj = linear(concat, w.WO);
  const resid1 = addMatrices(X, proj);
  const ln2 = layerNorm(resid1, w.ln2Gamma, w.ln2Beta);
  const ffn = feedForward(ln2.output, w.W1, w.b1, w.W2, w.b2);
  const out = addMatrices(resid1, ffn.output);
  return { ln1, Q, K, V, Qh, Kh, Vh, dHead, attn, ctxHeads, concat, proj, resid1, ln2, ffn, output: out };
}

/** Encoder attention path (spec Mode A: no mask, no LN/residual — noted in UI). */
export function encoderAttention(X, w, nHeads) {
  const Q = linear(X, w.WQ);
  const K = linear(X, w.WK);
  const V = linear(X, w.WV);
  const { heads: Qh, dHead } = splitHeads(Q, nHeads);
  const { heads: Kh } = splitHeads(K, nHeads);
  const { heads: Vh } = splitHeads(V, nHeads);
  const attn = Qh.map((q, h) => scaledDotProductAttention(q, Kh[h], Vh[h], { causal: false }));
  const ctxHeads = attn.map((a) => a.output);
  const concat = concatHeads(ctxHeads);
  const proj = linear(concat, w.WO);
  const ffn = feedForward(proj, w.W1, w.b1, w.W2, w.b2);
  return { Q, K, V, Qh, Kh, Vh, dHead, attn, ctxHeads, concat, proj, ffn };
}

/* ---------------- LM head / logits / temperature / softmax ---------------- */

export function lmHead(X, Wlm) {
  return matmul(X, Wlm); // [seq,vocab]
}

export function applyTemperature(logitsVec, temperature) {
  return logitsVec.map((v) => v / temperature);
}

/* ---------------- Sampling ---------------- */

function renormalizeSample(cands, rng) {
  const s = cands.reduce((a, c) => a + c.prob, 0);
  const norm = cands.map((c) => ({ ...c, prob: c.prob / s }));
  let r = rng();
  for (const c of norm) { r -= c.prob; if (r <= 0) return { selectedId: c.id, candidates: norm }; }
  return { selectedId: norm[norm.length - 1].id, candidates: norm };
}

export function greedySample(probs) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  const candidates = probs.map((p, id) => ({ id, prob: p, kept: id === best }));
  return { selectedId: best, candidates };
}

export function topKSample(probs, k, rng) {
  const order = probs.map((p, id) => ({ id, prob: p })).sort((a, b) => b.prob - a.prob);
  const keep = new Set(order.slice(0, Math.min(k, order.length)).map((c) => c.id));
  const kept = order.filter((c) => keep.has(c.id));
  const picked = renormalizeSample(kept, rng);
  const candidates = probs.map((p, id) => ({ id, prob: p, kept: keep.has(id) }));
  return { selectedId: picked.selectedId, candidates, renormalized: picked.candidates };
}

export function topPSample(probs, p, rng) {
  const order = probs.map((pr, id) => ({ id, prob: pr })).sort((a, b) => b.prob - a.prob);
  let cum = 0;
  const keep = new Set();
  for (const c of order) { keep.add(c.id); cum += c.prob; if (cum >= p) break; }
  const kept = order.filter((c) => keep.has(c.id));
  const picked = renormalizeSample(kept, rng);
  const candidates = probs.map((pr, id) => ({ id, prob: pr, kept: keep.has(id) }));
  return { selectedId: picked.selectedId, candidates, renormalized: picked.candidates, cumulative: cum };
}

export function sampleNext(probs, method, { k = 5, p = 0.9, rng } = {}) {
  if (method === "greedy") return { ...greedySample(probs), method };
  if (method === "top-k") return { ...topKSample(probs, k, rng), method, k };
  return { ...topPSample(probs, p, rng), method, p };
}

/* ---------------- Full forward passes (single source of truth) ---------------- */

export function forwardEncoder(text, config, seed) {
  const vocab = buildVocab(config.vocabSize);
  const { tokens, ids } = tokenize(text, config.vocabSize);
  const weights = createWeights(config, seed);
  const w0 = weights.layers[0];
  const X = ids.length ? embedLookup(ids, weights.embedding) : [];
  const PE = ids.length ? createSinusoidalPositionEncoding(ids.length, config.dModel) : [];
  const Xpos = ids.length ? addMatrices(X, PE) : [];
  const enc = ids.length ? encoderAttention(Xpos, w0, config.nHeads) : null;
  return { vocab, tokens, ids, weights, X, PE, Xpos, enc, config, seed };
}

export function forwardDecoder(ids, tokens, config, seed) {
  const vocab = buildVocab(config.vocabSize);
  const weights = createWeights(config, seed);
  if (!ids.length) {
    return { vocab, tokens, ids, weights, X: [], PE: [], Xpos: [], blocks: [], finalLN: null, logits: [], config, seed };
  }
  const X = embedLookup(ids, weights.embedding);
  const PE = createSinusoidalPositionEncoding(ids.length, config.dModel);
  const Xpos = addMatrices(X, PE);
  const blocks = [];
  let h = Xpos;
  for (let l = 0; l < config.nLayers; l++) {
    const b = transformerBlock(h, weights.layers[l], config.nHeads, { causal: true });
    blocks.push(b);
    h = b.output;
  }
  const finalLN = layerNorm(h, weights.finalGamma, weights.finalBeta);
  const logits = lmHead(finalLN.output, weights.lmHead);
  return { vocab, tokens, ids, weights, X, PE, Xpos, blocks, finalLN, logits, config, seed };
}

/** Logits->temperature->softmax->sampling for the LAST position. */
export function decodeLastPosition(decoderTrace, { temperature, method, k, p, sampleSeed }) {
  const seq = decoderTrace.logits.length;
  const logitsLast = [...decoderTrace.logits[seq - 1]];
  const scaled = applyTemperature(logitsLast, temperature);
  const probs = softmaxVec(scaled);
  const rng = mulberry32(sampleSeed >>> 0);
  const sampling = sampleNext(probs, method, { k, p, rng });
  return { logitsLast, scaled, probs, sampling };
}

/* ---------------- Stats ---------------- */

export function statsOf(M) {
  const flat = M.flat();
  if (!flat.length) return { min: 0, max: 0, mean: 0 };
  let min = Infinity, max = -Infinity, s = 0;
  for (const v of flat) { if (v < min) min = v; if (v > max) max = v; s += v; }
  return { min, max, mean: s / flat.length };
}

export function vecStats(v) {
  return statsOf([v]);
}
