# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development

No build step. Serve the directory statically and open `index.html`:

```bash
python3 -m http.server 8080
# open http://localhost:8080
```

Run the engine test suite in Node:

```bash
node -e "import('./js/tests.js').then(m => { const r = m.runTests(); console.log(r.filter(x=>x.pass).length + '/' + r.length + ' passing'); r.filter(x=>!x.pass).forEach(x => console.log('FAIL', x.name, x.error)); })"
```

## Architecture

Strict one-way data flow: **engine → state → views → explanations**

| File | Role |
|---|---|
| `js/engine.js` | Pure math — tokenizer, embeddings, attention, FFN, sampling. No DOM access. |
| `js/app.js` | Central state (`S`), navigation, config, inspector, generation loop. |
| `js/viewsEncoder.js` | Mode A (encoder) stage renderers — reads `S.encTrace` only. |
| `js/viewsDecoder.js` | Mode B (decoder) stage renderers — reads `S.decTrace` / `S.decStep` only. |
| `js/format.js` | All visual primitives: matrix/heatmap, bars, token chips, shape badges, GELU plot. |
| `js/explanations.js` | Educational copy (What / Why / Shape / Equation / More). |
| `js/tests.js` | Unit tests for every `engine.js` operation; runs in Node and in-browser. |

## Key invariants

- **Never compute in a view.** All numerical work goes in `engine.js`; view files only read from the trace objects.
- **One source of truth.** All visualization reads from `S.encTrace` or `S.decTrace`/`S.decStep`. These are recomputed from `(input, config, seed)` whenever any of those change.
- **Seeded determinism.** Each weight/table draws from `mulberry32(seed ⊕ FNV1a(streamName))` so dimensions are independently shuffled. Sampling draw `n` uses `mulberry32(seed + n·1013904223)`.

## Adding a pipeline stage

1. Append `{ id, nav, title, sub, shape }` to `ENCODER_STAGES` / `DECODER_STAGES` and a matching entry to the explanations array (same index).
2. Add a `case` in `renderEncoderStage` / `renderDecoderStage` reading only from the existing trace. New math belongs in `engine.js`.
3. Add a default inspector key in `STAGE_DEFAULT_TENSOR` and the tensor in `currentTensors()` if it introduces an inspectable tensor.
4. Add engine tests in `tests.js` for any new operation.
