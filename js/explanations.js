/* Educational copy: one compact panel per stage.
 * Structure: what / why / shape / equation (+ optional "tell me more").
 * Tone: precise, computational, non-anthropomorphic. */

export const ENCODER_EXPLANATIONS = [
  {
    what: "Text is split on whitespace, and each token is mapped to an integer ID with FNV-1a hashing.",
    why: "Models operate on numbers. IDs index into the embedding table and vocabulary.",
    shape: "text → [seq]",
    equation: "id = FNV-1a(token) mod vocab_size",
    more: "Toy tokenizer: whitespace split + FNV-1a hash. This is an educational simplification — real GPT-family tokenizers use learned subword schemes (e.g. BPE), not hashing. Hashing can collide; that is acceptable here because the goal is to visualize downstream math, not to build a real vocabulary.",
  },
  {
    what: "Each token ID is replaced by its row in the embedding table: a vector of d_model numbers.",
    why: "Vectors give the model a working representation of each token that later layers combine with context.",
    shape: "[seq] → [seq, d_model]",
    equation: "x_i = E[id_i],  E ∈ R^{vocab × d_model}",
    more: "Educational simplification: this simulator fills the embedding table with seeded Gaussian values so runs are reproducible. In real models these values are learned parameters from training — they are not sampled from a Gaussian at inference time.",
  },
  {
    what: "A sinusoidal position code is added to every embedding so position information enters the representation.",
    why: "Attention itself treats the sequence as a set; without position codes, order would be invisible.",
    shape: "[seq, d_model] + [seq, d_model] → [seq, d_model]",
    equation: "PE(pos,2i)=sin(pos/10000^{2i/d}), PE(pos,2i+1)=cos(pos/10000^{2i/d})",
    more: "This is the fixed sinusoidal scheme from Vaswani et al. (2017). Other schemes (learned positions, RoPE) exist; the simulator uses sinusoids because they are deterministic and easy to visualize.",
  },
  {
    what: "The same input is projected three ways by learned matrices into queries (Q), keys (K), and values (V).",
    why: "Queries ask, keys advertise, values carry content: comparing Q against K decides how V rows get mixed.",
    shape: "[seq, d_model] × [d_model, d_model] → [seq, d_model] (×3)",
    equation: "Q=XW_Q,  K=XW_K,  V=XW_V",
    more: "Each output row is a linear combination of the input row. The three projections let the model use the same token representation in three different roles at once.",
  },
  {
    what: "Q, K, V are reshaped from [seq, d_model] into [n_heads, seq, d_head], one slice per head.",
    why: "Heads run attention independently in subspaces, so different heads can track different relations.",
    shape: "[seq, d_model] → [n_heads, seq, d_head]",
    equation: "d_head = d_model / n_heads",
    more: "Splitting is a reshape, not a learned operation — no numbers change, only grouping. Each head below attends with its own Q/K/V slice.",
  },
  {
    what: "Each head scores queries against keys, scales, normalizes with softmax, and mixes value rows.",
    why: "This is how a token representation absorbs context from other positions, weighted by relevance scores.",
    shape: "[seq, d_head] × [d_head, seq] → [seq, seq] → × [seq, d_head]",
    equation: "Attention(Q,K,V) = softmax(QKᵀ/√d_head)V",
    more: "Scores are raw compatibilities; scaling by √d_head keeps softmax inputs in a stable range; weights (rows sum to 1) are the mixing coefficients; the output row is a weighted sum of value rows.",
  },
  {
    what: "Head outputs are concatenated back to d_model width, then projected once more by W_O.",
    why: "Concatenation reunites the subspaces; W_O lets the model blend per-head results into one representation.",
    shape: "[n_heads, seq, d_head] → [seq, d_model] → ×W_O → [seq, d_model]",
    equation: "out = concat(head_1..head_h) · W_O",
    more: "Concat and projection are distinct steps: concat only reorders values, while W_O is a learned mixing matrix.",
  },
  {
    what: "Each position passes independently through an expansion, GELU nonlinearity, and contraction back.",
    why: "Attention mixes across positions; the FFN transforms each position's vector with extra capacity.",
    shape: "[seq, d_model] → [seq, d_ff] → GELU → [seq, d_model]",
    equation: "FFN(x) = GELU(xW_1+b_1)W_2+b_2",
    more: "The same W_1/W_2 apply to every position (position-wise). GELU curves negative values smoothly toward zero instead of cutting them hard like ReLU.",
  },
];

export const DECODER_EXPLANATIONS = [
  {
    what: "The prompt is split on whitespace and hashed to token IDs (toy tokenizer).",
    why: "Generation starts from numbers: IDs index embeddings and define the growing context.",
    shape: "text → [seq]",
    equation: "id = FNV-1a(token) mod vocab_size",
    more: "Educational simplification: whitespace + FNV-1a hash. Real decoders use subword tokenizers. Detokenization maps sampled IDs back through the toy vocabulary list.",
  },
  {
    what: "IDs become embedding vectors; sinusoidal position codes are added.",
    why: "The block stack needs position-aware vectors before any attention happens.",
    shape: "[seq] → [seq, d_model] → +PE → [seq, d_model]",
    equation: "x_i = E[id_i] + PE(i)",
    more: "Same embedding and sinusoidal tables as encoder mode, rebuilt deterministically from the seed.",
  },
  {
    what: "Each token vector is normalized over its features (mean 0, variance 1), then scaled by γ and shifted by β.",
    why: "Pre-norm keeps activations in a stable range so attention and FFN blocks train and behave reliably.",
    shape: "[seq, d_model] → [seq, d_model]",
    equation: "LN(x) = γ·(x−μ)/σ + β",
    more: "LayerNorm normalizes over the feature dimension of each token independently — unlike BatchNorm, it never mixes across batch items or positions.",
  },
  {
    what: "Normalized states are projected to Q/K/V and split into per-head slices.",
    why: "Same Q/K/V mechanism as the encoder, now inside each decoder block.",
    shape: "[seq, d_model] → [n_heads, seq, d_head] (×3)",
    equation: "Q=XW_Q, K=XW_K, V=XW_V",
    more: "Weights differ per block but are fixed after initialization here; in real models they are learned during training.",
  },
  {
    what: "Scores for future positions are set to −∞ before softmax, so each token attends only to itself and the past.",
    why: "Generation must not peek at tokens it has not produced yet. This is what makes the model autoregressive.",
    shape: "[seq, seq] → masked → softmax → [seq, seq]",
    equation: "score_{i,j} = −∞  for j > i, before softmax",
    more: "The mask is applied after scaling and before softmax. Softmax turns −∞ into probability 0. Toggle the mask to see future columns collapse to zero.",
  },
  {
    what: "Head outputs are concatenated, projected by W_O, and added back to the block input (residual).",
    why: "Residuals preserve the original stream: attention only adds a refinement instead of replacing the state.",
    shape: "X + Attn(LN(X)) → [seq, d_model]",
    equation: "x' = x + W_O·concat(heads)",
    more: "Follow the two paths: the straight residual path and the attention path merge at '+'. This pattern repeats around the FFN too.",
  },
  {
    what: "A second norm + FFN refines each position, then a second residual adds it back: LN → GELU-MLP → add.",
    why: "Same role as encoder FFN: per-position capacity that attention (cross-position mixing) does not provide.",
    shape: "[seq, d_model] → [seq, d_ff] → [seq, d_model] → + residual",
    equation: "x'' = x' + FFN(LN(x'))",
    more: "One full pre-LN block is: x → +Attn(LN(x)) → +FFN(LN(·)). Stack N blocks by repeating this transformation.",
  },
  {
    what: "The stacked blocks run in order; expand one block to inspect its internals.",
    why: "Depth composes refinements: early blocks build local patterns, later blocks build on them.",
    shape: "[seq, d_model] → block ×N → [seq, d_model]",
    equation: "h_{l+1} = Block_l(h_l)",
    more: "Progressive disclosure: the overview shows only shapes per block. Select a block to reuse the same LN/attention/FFN views with that block's tensors.",
  },
  {
    what: "One final LayerNorm is applied to the last block's output.",
    why: "It standardizes the final hidden states before vocabulary scoring.",
    shape: "[seq, d_model] → [seq, d_model]",
    equation: "LN(x) = γ·(x−μ)/σ + β",
    more: "Same LayerNorm operation as inside blocks, with its own γ/β parameters.",
  },
  {
    what: "Final states are projected to one raw score (logit) per vocabulary token; the last row predicts the next token.",
    why: "Logits rank every possible continuation from the current context representation.",
    shape: "[seq, d_model] × [d_model, vocab] → [seq, vocab]",
    equation: "logits = h·W_LM",
    more: "Logits are unnormalized scores — not probabilities. Only the last sequence position is used for next-token prediction; earlier rows are shown for inspection.",
  },
  {
    what: "Last-row logits are divided by temperature, exponentiated, and normalized into probabilities.",
    why: "Temperature reshapes sharpness; softmax converts scores into a distribution that sums to 1.",
    shape: "[vocab] → /T → softmax → [vocab], Σ=1",
    equation: "p_i = e^{z_i/T} / Σ_j e^{z_j/T}",
    more: "Lower T sharpens (confident peaks), higher T flattens (more uniform). This changes shape only — never call it more or less intelligent.",
  },
  {
    what: "A token is drawn from the distribution (greedy / top-k / top-p), detokenized, and appended to the context.",
    why: "Sampling turns probabilities into a choice; appending makes generation autoregressive.",
    shape: "[vocab] → 1 id → text grows by 1 token",
    equation: "greedy: argmax p · top-k: keep k · top-p: keep cumsum ≥ p",
    more: "The same seed + method reproduces the same pick. Inspect any history step to see the full distribution that produced it.",
  },
];

export const ABOUT_TEXT = `This is a small educational Transformer simulator. Its tokenizer, vocabulary,
dimensions, and weights are intentionally simplified so the internal computation can
be visualized. It demonstrates the mechanics of Transformer inference (not training)
rather than reproducing a production GPT model.

Educational simplifications: whitespace+FNV-1a toy tokenizer · tiny fixed
vocabulary · seeded Gaussian toy weights (real weights are learned in training) ·
small d_model / layers / heads · sinusoidal positions.

Inference only: all weights are fixed. Attention is one component among
embeddings, norms, residuals, FFNs, and the LM head — not the whole model.`;
