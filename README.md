# Inside a Tiny Transformer — Interactive LLM Visualizer

A web-based educational simulator that opens up a tiny Transformer and lets learners
watch data flow through it: **text → tokens → embeddings → Q/K/V → attention →
FFN → logits → probabilities → sampling → next token**.

No build step, no dependencies, no backend. Serve the folder statically and open
`index.html`. All computation runs in the browser from a pure, tested math engine.

## Quick start

```bash
cd llm-visualizer
python3 -m http.server 8080
# open http://localhost:8080
```

## Standalone build

`standalone.html` is a single self-contained file (CSS + JS inlined) that opens
directly in a browser over `file://` with no server needed.

```bash
npm install   # first time only — installs esbuild
npm run build # regenerate standalone.html
```

`build.js` patches `index.html` in-place: it replaces the `<link>` tag with an
inlined `<style>` block and the `<script type="module">` tag with the bundled JS.
Run it whenever you change `index.html`, `styles.css`, or any file under `js/`.

Run the engine tests in Node:

```bash
node -e "import('./js/tests.js').then(m => { const r = m.runTests(); console.log(r.filter(x=>x.pass).length + '/' + r.length + ' passing'); r.filter(x=>!x.pass).forEach(x => console.log('FAIL', x.name, x.error)); })"
```

Or click **Run tests** in the app header (same suite, in-page).

## Architecture

```
index.html          shell: header, config bar, 3-column layout, inspector, modal
styles.css          one quiet visual language (tokens, shapes, matrices, bars, equations)
js/
  engine.js         PURE math: tokenizer, embeddings, attention, norms, FFN,
                    blocks, LM head, sampling. No DOM. Independently testable.
  tests.js          unit tests for every operation in §38 of the spec.
  explanations.js   educational copy (What / Why / Shape / Equation / More).
  format.js         single implementation of every visual primitive:
                    matrix/heatmap, bars, token chips, shape badges, GELU plot.
  viewsEncoder.js   Mode A stage renderers (read S.encTrace only).
  viewsDecoder.js   Mode B stage renderers (read S.decTrace / S.decStep only).
  app.js            central state, navigation, config, inspector, generation loop.
```

Strict separation: **numerical model** (`engine.js`) → **application state**
(`app.js`) → **visualization** (`views*.js` + `format.js`) →
**educational content** (`explanations.js`).

## One source of truth

`app.js` holds two trace objects, recomputed from `(input, config, seed)`:

- `S.encTrace = forwardEncoder(text, config, seed)`
- `S.decTrace = forwardDecoder(ids, tokens, config, seed)` plus
  `S.decStep = decodeLastPosition(...)` for the last position.

Every visualization, tooltip, inspector cell, and "show me the math" box reads
from these traces. There are no separate "display numbers".

## Tensor shapes (defaults: d=8, ff=16, h=2, L=2, V=32)

| Stage | Shape |
|---|---|
| ids | `[seq]` |
| embedding / +PE | `[seq, d]` |
| W_Q/W_K/W_V, Q/K/V | `[d, d]`, `[seq, d]` |
| per-head Q/K/V | `[seq, d/h]` |
| scores / masked / weights | `[seq, seq]` |
| context / concat / proj | `[seq, d/h]`, `[seq, d]`, `[seq, d]` |
| FFN hidden / out | `[seq, d_ff]`, `[seq, d]` |
| logits | `[seq, V]` |
| last-row logits → probs | `[V]`, Σ=1 |

## Random seed strategy

- One master **seed** (header). Every table/weight draws from an independent
  `mulberry32(seed ⊕ FNV1a(streamName))` stream, so changing one dimension
  doesn't reshuffle unrelated tensors.
- Embeddings/weights: Gaussian × small scale (toy stand-ins for learned params).
- Sampling draw `n` uses `mulberry32(seed + n·1013904223)`, so generation is
  exactly reproducible unless the learner changes seed/method/temperature.

## Mathematical definitions

- Tokenizer: `id = FNV1a(token) mod V` (whitespace split).
- Positions: `PE(pos,2i)=sin(pos/10000^(2i/d))`, odd dims `cos`.
- Attention: `softmax(QKᵀ/√d_h)V`, causal variant masks `j>i` with −∞ pre-softmax.
- LayerNorm over features per token: `γ(x−μ)/σ+β`, γ=1, β=0.
- GELU (tanh approx), FFN `GELU(xW₁+b₁)W₂+b₂`, pre-LN blocks with 2 residuals.
- Sampling: greedy `argmax`, top-k (keep k), top-p (min set with cumsum ≥ p),
  renormalize, then draw. Temperature divides logits **before** softmax.

## Educational simplifications (labeled in-app)

Toy whitespace+FNV-1a tokenizer · tiny fixed vocabulary · seeded Gaussian toy
weights (real weights are **learned in training**) · small dims · sinusoidal
positions · inference only. Mode A intentionally shows the spec's
attention→FFN path without residuals/norms (production encoders include them;
the full pre-LN block is in Mode B).

## How to add a new pipeline stage

1. Append `{ id, nav, title, sub, shape }` to `ENCODER_STAGES` /
   `DECODER_STAGES` and a matching entry to the explanations array (same index).
2. Add a `case` in `renderEncoderStage` / `renderDecoderStage` reading only
   from the existing trace (add engine functions in `engine.js` if new math is
   needed — never compute inside the view).
3. Add a default inspector key in `STAGE_DEFAULT_TENSOR` and the tensor in
   `currentTensors()` if it introduces a new inspectable tensor.
4. Add engine tests in `tests.js` for any new operation.

## Testing strategy

`tests.js` covers tokenization determinism, FNV-1a vectors, softmax sums,
causal zero-leakage, attention/head shapes, LayerNorm moments, GELU values,
temperature-before-softmax, top-k/top-p set properties, greedy argmax,
sequence growth, full determinism, and invalid head configs. Same suite runs
in Node and in the browser.

---

## Disclaimer

The requirements, design, and architecture of this application were conceived and directed by the author. Generative AI tools were used to assist with parts of the implementation.
