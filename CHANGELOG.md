# Changelog

## [0.1.0] - 2026-10-12

### Fixed
- **`unload()` during `prepare()` no longer corrupts engine state.** Each load carries a generation;
  a load superseded by `unload()` or a newer `prepare()` can no longer publish its result. Before,
  a load that finished after `unload()` re-marked the engine ready, and a superseded load that
  failed overwrote a newer in-flight load with `.failed`. A `prepare()` interrupted this way now
  throws `CancellationError`.
  The caller that started a load and every caller that joined it publish through one helper, so all
  of them finish with `.done` on success and throw `CancellationError` when the load was superseded.
- **`deleteDownloadedWeights()` refuses while a cancelled load is still winding down**
  (`cannotDeleteWhileLoaded`), instead of deleting files under it.
- **Weights already on disk load without the network.** Downloaded weights now load from their
  directory. Before, every `prepare()` re-listed the Hugging Face repository and failed on any
  network error other than "not connected" (captive portal, DNS failure, timeout, server error).
- **Interrupted downloads are no longer reported as present.** A model directory must hold
  `config.json`, `tokenizer.json`, `tokenizer_config.json` and a `.safetensors` file.
- **`prepare()` completes when the caller does not retain the engine.** It used to finish silently
  with no `.done` and no error.
- **Cancelling `embed(batch:)` stops between chunks** with `CancellationError`.
- **Progress no longer reports file counts as bytes.** `bytesDownloaded` / `bytesTotal` are now
  `nil` from `prepare()`: the download stack reports per-file progress only. Loads of weights
  already on disk report `.loadingWeights` instead of `.downloading`.
- **Produced vectors are validated**: wrong length throws `dimensionMismatch`, NaN/infinity throws
  `nonFiniteEmbedding`.
- **`VectorMath.match`** skips documents whose score is not finite (previously ranked as NaN),
  orders equal scores by ascending index, and states that a negative `top` traps.

### Changed
- The model cache directory is excluded from iCloud / device backups.
- `state` and `modelLocationOnDisk()` no longer create directories.
- `SemanticSearchError` is now public, so callers can match on its cases. New cases:
  `dimensionMismatch(expected:received:)`, `nonFiniteEmbedding`.
- `swift-transformers` (`Hub`, `Tokenizers`) is declared as a direct dependency, with the same
  version range `mlx-swift-lm` already requires.
- Documentation: README (requirements, errors, storage, tests, install version), doc comments,
  `.spi.yml` for Swift Package Index documentation.

## [0.0.8] - 2026-08-06

### Added
- **GPU memory introspection: `GPUMemoryReport`.** MLX allocates its tensors as Metal buffers,
  which the OS accounts as `IOAccelerator` memory — invisible to malloc-based tooling while the
  process footprint climbs into the tens of gigabytes. `GPUMemoryReport.current()` (also
  `SemanticSearchEngine.gpuMemoryReport()`) reports MLX's active / cache / peak bytes plus the
  Metal device's total allocation for the process, so hosts can see the gap between "what MLX is
  using" and "what MLX is hoarding". The number that grows without bound on embedding workloads is
  the **cache**: MLX recycles buffers by size, variable-length padding gives nearly every forward
  pass unique sizes, and the default cache cap (≈1.5× the GPU working-set size) permits tens of GB.
- **Per-batch growth tracing.** `MLXEmbedderBackend` logs a debug-level report after every forward
  pass and a notice-level line each time MLX's footprint grows another 512 MB (unified log,
  subsystem `com.nuclearcyborg.SemanticSearch`, category `GPUMemory`) — a runaway cache shows up
  as a ladder of notices instead of a mystery footprint.
- **Host policy knobs: `SemanticSearchEngine.setGPUCacheLimit(bytes:)` / `clearGPUCache()`.**
  Thin static wrappers over `MLX.Memory` so a host can cap the cache or return it to Metal after a
  bulk operation without importing MLX itself. The library deliberately sets no policy of its own —
  the right cap depends on what else the process does.

## [0.0.7] - 2026-07-26

### Fixed
- **Embeddings are now actually unit-length.** `extractVectors` widens to `Float32` *before* pooling
  and L2 normalization. The model computes in bfloat16 — 7 mantissa bits, ~0.4% relative precision —
  so dividing by the norm in that dtype rounded every component of the quotient, leaving vectors
  whose lengths scattered across ±0.4% instead of sitting at 1.0 (measured 0.995752–1.004187 over a
  223-vector corpus). That broke the contract this function exists to uphold, and broke it
  *systematically* rather than randomly: a caller using a dot product as a stand-in for cosine — the
  standard reason to normalize at all — gave a permanent ~0.4% boost to whichever vectors happened to
  land above 1.0, on every query, forever.
- **`prepare()` no longer force-unwraps.** A leftover debug `print` unwrapped
  `ModelLocator.downloadedLocation()`, which is `nil` whenever the model is **bundled** — `locate()`
  checks the bundle first, so nothing is ever downloaded. Shipping weights inside the host app (a
  documented, supported configuration via `bundleSubdirectory`) therefore trapped on every successful
  `prepare()`. A force unwrap traps rather than throws, so the enclosing `do`/`catch` never saw it.

### Changed
- Model load and warmup timings go to the unified log (subsystem `com.nuclearcyborg.SemanticSearch`,
  category `Engine`) instead of `print()`. A library should not write to its host's stdout.

### Notes
- **`EmbeddingModel.identifier` is deliberately unchanged; re-embedding is NOT required.** Vectors
  differ from 0.0.6 only in magnitude precision — direction is unchanged, because the components were
  already bfloat16 coming out of the model and only the division moved to Float32. Bumping the
  identifier would force every consumer to re-embed an entire corpus to correct a 0.4% magnitude
  error. Callers using a true cosine are unaffected either way; callers relying on the unit-length
  shortcut can rescale stored vectors on load, which is direction-preserving and recovers exactly the
  intended vector.

### Added
- `Examples/Benchmark/` — an embedding latency harness. Deliberately **not** a SwiftPM target, matching
  `Examples/CLIDemo` and `Examples/GUIDemo`: a published library should not carry a benchmark's
  dependencies. Measures latency against input size, batch-of-1 vs batch-of-2, sequential vs batched
  vs length-sorted corpus embedding, the MLX `cacheLimit`/`memoryLimit`/`wiredLimit` knobs, and an
  idle probe. See its README for how to run it.

## [0.0.3] - 2026-06-17

### Changed
- **Pooling is now a declared property of each model** — `EmbeddingModel.pooling: PoolingStrategy`
  (`.lastToken` / `.mean` / `.cls`), a **required** init parameter. The backend switches on it and
  **throws on a strategy it hasn't implemented** (during the warmup embed in `prepare()`) rather than
  silently producing a wrong vector. Previously last-token was hardcoded — correct for Qwen3 but a
  silent footgun for any non-causal model. Adding a new model now forces an explicit pooling decision
  at compile time. Qwen3 declares `.lastToken`; **output is identical to 0.0.2** (the identifier is
  unchanged, so no re-embedding is required).

### Added
- Load-phase timing: `prepare()` logs how long model download/load and warmup (first forward + Metal
  kernel compilation) each take.

## [0.0.2] - 2026-06-17

### Changed
- **`EmbeddingModel.identifier` now encodes the pooling scheme** (`…-lasttoken`). The identifier
  is the stable signature of the *embedding output* (model + quantization + pooling), meant to be
  persisted alongside stored vectors so a consumer can detect that a stored vector is stale and
  re-embed it. It is **not** a cache key (weights are located by `huggingFaceRepo` /
  `bundleSubdirectory`), so bumping it never forces a model re-download. Clarified the doc comment
  accordingly (the old "used as a directory name" note was stale).
  - This means 0.0.2's identifier differs from 0.0.1's even though both produce last-token vectors;
    consumers that began persisting the identifier should treat the bump as a one-time re-embed.

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
