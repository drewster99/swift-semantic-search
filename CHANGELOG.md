# Changelog

## [0.0.1] - 2026-06-17

First tagged release.

### Changed
- **Embedding pooling: mean → last-token (EOS).** Qwen3-Embedding is a *causal*
  model trained for last-token pooling — the sentence embedding is the hidden
  state of the final (EOS) token. The previous implementation mean-pooled the
  whole sequence (inherited from the `mlx-swift-examples` `embedder-tool`
  reference), which is wrong for this model: it compressed the cosine
  distribution into a narrow high band, badly hurting retrieval separability.

  `MLXEmbedderBackend` now gathers each sequence's final *real* token hidden
  state. Batching is preserved: a `.none` pooler returns the full per-token
  hidden states (Qwen3's `pooledOutput` is nil), and we select each row's last
  real token. MLXEmbedders' own `.last` strategy can't be used directly — it
  reads index `-1`, which is a PAD position under right-padding. Right-padding
  is safe for the forward pass because the model is causal: each EOS attends
  only to earlier real tokens, so trailing pads never affect it.

### Impact (measured on a 156-document / 143-query retrieval eval)
- Gold-vs-noise cosine **separability gap nearly doubled** (median 0.106 → 0.185).
- **Recall up ~13–20 points** across pools (e.g. prior-task recall 50.7% → 70.3%,
  memory recall 76.8% → 89.6%).
- **F1 up ~6–10 points**, and an absolute-similarity cutoff becomes *effective*
  (it was a no-op under mean pooling, since everything sat in one high band).
- Qwen3 query **instruction prefixes flipped from harmful to helpful**, which is
  the expected behavior once pooling matches how the model was trained.

### Migration note
This is an **inference-side change only** — same model, weights, tokenizer, and
1024-dimensional output. But last-token vectors are **not comparable** to the old
mean-pooled vectors, so **consumers must re-embed any persisted corpus** after
upgrading (the dimension is unchanged, so a dimension check will not detect the
difference).
