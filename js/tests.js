/* Unit tests for the mathematical engine. Run in-page or via node. */
import * as E from "./engine.js";

const approx = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;

export function runTests() {
  const results = [];
  const t = (name, fn) => {
    try { fn(); results.push({ name, pass: true }); }
    catch (e) { results.push({ name, pass: false, error: String(e?.message ?? e) }); }
  };
  const assert = (cond, msg) => { if (!cond) throw new Error(msg); };

  t("FNV-1a: known vector ('hello' -> 1335831723)", () => {
    assert(E.fnv1a("hello") === 1335831723, `got ${E.fnv1a("hello")}`);
  });
  t("FNV-1a: empty string -> offset basis", () => {
    assert(E.fnv1a("") === 0x811c9dc5, `got ${E.fnv1a("")}`);
  });
  t("Tokenization: 'The cat sat' -> 3 tokens, ids in range", () => {
    const { tokens, ids } = E.tokenize("The cat sat", 32);
    assert(tokens.join("|") === "The|cat|sat", JSON.stringify(tokens));
    assert(ids.length === 3 && ids.every((id) => id >= 0 && id < 32), JSON.stringify(ids));
  });
  t("Tokenization: deterministic (same input -> same ids)", () => {
    const a = E.tokenize("The cat sat", 32).ids;
    const b = E.tokenize("The cat sat", 32).ids;
    assert(JSON.stringify(a) === JSON.stringify(b), "not deterministic");
  });
  t("Softmax: rows sum to 1", () => {
    const m = E.softmax([[1, 2, 3], [-1, 0, 1]]);
    m.forEach((row) => assert(approx(row.reduce((a, b) => a + b, 0), 1, 1e-9), "row != 1"));
  });
  t("Causal mask: future positions get ~0 probability after softmax", () => {
    const scores = [[1, 1, 1], [1, 1, 1], [1, 1, 1]];
    const w = E.softmax(E.applyCausalMask(scores));
    assert(w[0][1] < 1e-6 && w[0][2] < 1e-6, JSON.stringify(w[0]));
    assert(w[1][2] < 1e-6 && w[1][0] > 0.4, JSON.stringify(w[1]));
    assert(w[2].every((x) => x > 0.3 && x < 0.35), JSON.stringify(w[2]));
  });
  t("Attention: output shape [seq,dHead]", () => {
    const cfg = { dModel: 8, dFF: 16, nHeads: 2, nLayers: 1, vocabSize: 32 };
    const tr = E.forwardEncoder("The cat sat", cfg, 42);
    const [seq, dm] = [tr.ids.length, cfg.dModel];
    assert(tr.enc.attn.length === 2, "head count");
    assert(tr.enc.attn[0].output.length === seq && tr.enc.attn[0].output[0].length === dm / 2, "out shape");
    assert(tr.enc.attn[0].weights.length === seq && tr.enc.attn[0].weights[0].length === seq, "weights shape");
  });
  t("Multi-head: split + concat restores [seq,dModel]", () => {
    const X = E.createMatrix(4, 8, (r, c) => r * 8 + c);
    const { heads } = E.splitHeads(X, 2);
    assert(heads.length === 2 && heads[0].length === 4 && heads[0][0].length === 4, "split shape");
    assert(JSON.stringify(E.concatHeads(heads)) === JSON.stringify(X), "concat mismatch");
  });
  t("LayerNorm: normalized rows have ~0 mean, ~1 variance", () => {
    const X = [[1, 2, 3, 4], [5, -2, 0, 3]];
    const g = [1, 1, 1, 1], b = [0, 0, 0, 0];
    const { normalized } = E.layerNorm(X, g, b);
    normalized.forEach((row) => {
      const mean = row.reduce((a, x) => a + x, 0) / row.length;
      const v = row.reduce((a, x) => a + (x - mean) ** 2, 0) / row.length;
      assert(approx(mean, 0, 1e-9), `mean ${mean}`);
      assert(approx(v, 1, 1e-3), `var ${v}`);
    });
  });
  t("GELU: known values", () => {
    assert(approx(E.gelu(0), 0, 1e-9), `gelu(0)=${E.gelu(0)}`);
    assert(approx(E.gelu(1), 0.8412, 1e-3), `gelu(1)=${E.gelu(1)}`);
    assert(approx(E.gelu(-1), -0.1588, 1e-3), `gelu(-1)=${E.gelu(-1)}`);
  });
  t("Temperature: applied before softmax (lower T sharpens)", () => {
    const logits = [2, 1, 0.5];
    const sharp = E.softmaxVec(E.applyTemperature(logits, 0.1));
    const flat = E.softmaxVec(E.applyTemperature(logits, 2.0));
    assert(sharp[0] > 0.99, JSON.stringify(sharp));
    assert(flat[0] < sharp[0] && flat[2] > sharp[2], JSON.stringify(flat));
  });
  t("Top-k: exactly k candidates eligible", () => {
    const probs = [0.5, 0.2, 0.15, 0.1, 0.05];
    const r = E.topKSample(probs, 3, E.mulberry32(1));
    assert(r.candidates.filter((c) => c.kept).length === 3, JSON.stringify(r.candidates));
  });
  t("Top-p: kept set reaches cumulative threshold", () => {
    const probs = [0.5, 0.2, 0.15, 0.1, 0.05];
    const r = E.topPSample(probs, 0.9, E.mulberry32(1));
    const kept = r.candidates.filter((c) => c.kept);
    const cum = kept.reduce((a, c) => a + c.prob, 0);
    assert(cum >= 0.9 - 1e-9, `cum=${cum}`);
    const sorted = kept.map((c) => c.prob).sort((a, b) => b - a);
    assert(cum - sorted[sorted.length - 1] < 0.9 + 1e-9, "not minimal set");
  });
  t("Greedy: selects argmax", () => {
    const r = E.greedySample([0.1, 0.7, 0.2]);
    assert(r.selectedId === 1, `got ${r.selectedId}`);
  });
  t("Generation: appending a token grows sequence by 1", () => {
    const cfg = { dModel: 8, dFF: 16, nHeads: 2, nLayers: 2, vocabSize: 32 };
    const a = E.forwardDecoder([1, 2], ["The", "cat"], cfg, 42);
    const b = E.forwardDecoder([1, 2, 9], ["The", "cat", "sat"], cfg, 42);
    assert(b.logits.length === a.logits.length + 1, "seq did not grow");
    assert(b.logits[0].length === 32, "logit width != vocab");
  });
  t("Determinism: same input+config+seed -> identical logits", () => {
    const cfg = { dModel: 8, dFF: 16, nHeads: 2, nLayers: 2, vocabSize: 32 };
    const a = E.forwardDecoder([1, 2], ["The", "cat"], cfg, 42);
    const b = E.forwardDecoder([1, 2], ["The", "cat"], cfg, 42);
    assert(JSON.stringify(a.logits) === JSON.stringify(b.logits), "logits differ");
  });
  t("Causal attention in decoder: no future leakage", () => {
    const cfg = { dModel: 8, dFF: 16, nHeads: 2, nLayers: 1, vocabSize: 32 };
    const tr = E.forwardDecoder([3, 7, 11], ["a", "b", "c"], cfg, 7);
    const w = tr.blocks[0].attn[0].weights;
    assert(w[0][1] < 1e-6 && w[0][2] < 1e-6, "row 0 leaks");
    assert(w[1][2] < 1e-6, "row 1 leaks");
    assert(approx(w[2].reduce((x, y) => x + y, 0), 1, 1e-9), "row 2 sum");
  });
  t("Invalid head config throws", () => {
    let threw = false;
    try { E.splitHeads([[1, 2, 3]], 2); } catch { threw = true; }
    assert(threw, "should throw when d_model % n_heads != 0");
  });

  return results;
}
