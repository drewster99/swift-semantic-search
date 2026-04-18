# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`SemanticSearch` is a standalone Swift **package** (not an app) that wraps an MLX-backed embedding model behind a small public API. It targets macOS 14 / iOS 17 / visionOS 1 on Apple Silicon. There is no `.xcodeproj` — consumers add it as a SwiftPM dependency.

The parent `/Users/andrew/cursor/CLAUDE.md` instructs using `xcode-mcp-server` for Xcode projects. That does not apply here: this is SwiftPM. Use `swift build` / `swift test` from the command line, or `xcodebuild` when MLX's Metal shaders need to be compiled (see Testing).

## Commands

```sh
# Pure-logic tests (VectorMath) — runs anywhere, fast.
swift test --filter VectorMath

# Build the package.
swift build

# Integration tests need Xcode's build pipeline so MLX's metallib is produced.
# `swift test` alone cannot compile the Metal shaders.
xcodebuild test \
    -scheme SemanticSearch \
    -destination 'platform=macOS' \
    -only-testing:SemanticSearchTests/SemanticSearchIntegrationTests
```

First integration run downloads the default Qwen3 weights (~330 MB) into `Application Support/<bundleID>/SemanticSearch/models/`; subsequent runs reuse that cache.

## Architecture

`SemanticSearchEngine` is an `actor` that holds lifecycle state (`idle` → `loading` → `loaded` / `failed`) and serializes GPU access. Its only job is: resolve model location → load weights → run forward passes. It deliberately does **not** own a corpus — `VectorMath.match` ranks a query against caller-supplied document vectors.

Load flow (`SemanticSearchEngine.prepare()`):
1. `ModelLocator.locate()` checks, in order: `Bundle.main/<model.bundleSubdirectory>`, then the MLX cache directory, then absent. A location only counts as "present" when `config.json` exists inside it.
2. `MLXEmbedders.loadModelContainer` downloads (if absent) and loads weights, reporting progress through the `AsyncThrowingStream<PrepareProgress, Error>` returned to the caller.
3. A single `"warmup"` embedding runs so the first real call is hot.

`prepare()` is idempotent and coalesces concurrent callers onto the in-flight load task.

Embed flow (`MLXEmbedderBackend.embed(batch:)`):
- Tokenize → pad to the batch max (pad token: `[PAD]` → `eosTokenId` → `0`) → build attention mask excluding pad positions → forward → pooler (`normalize: true`, `applyLayerNorm: false`) → extract Float vectors.
- Output shape is handled for both 2D `(batch, dim)` and 3D `(batch, seq, dim)` (mean-pooled across seq). The 3D path is what the default Qwen3 repo hits since it ships no `1_Pooling/config.json`.
- Sentence-level L2 normalization is re-applied in `extractVectors` because the pooler's `normalize: true` normalizes per-token *before* our Swift-side mean-pool — averaging unit vectors does not preserve unit norm. This is load-bearing for the API contract.

Cache layout:
- `ModelLocator.canonicalDownloadDirectory()` asks `MLXEmbedders.ModelConfiguration.modelDirectory(hub:)` for the path instead of hand-rolling the HuggingFace layout. Don't mirror the layout manually — the library owns it.
- `deleteDownloadedWeights()` refuses while the model is loaded (raise `cannotDeleteWhileLoaded`; caller must `unload()` first). Bundled files are never deleted.

`VectorMath` is a stateless `enum` using `Accelerate`/`vDSP` for dot product, magnitude, L2 normalization, cosine similarity, and ranked `match(...)`. Vectors whose length differs from the query are silently skipped in `match`.

## Non-obvious constraints

- **mlx-swift-lm is pinned tightly** to `.upToNextMinor(from: "2.29.2")`. `main` rewrote `MLXEmbedders.loadModelContainer`'s signature (now takes `Downloader`/`TokenizerLoader` instead of `HubApi`). Do not bump to 2.30+ without migrating the call site in `SemanticSearchEngine.performLoad`.
- **Adding a new model** is one `static let` on `EmbeddingModel` — no backend changes needed, as long as the model is MLX-compatible and sits in a HuggingFace repo.
- **`identifier`** on `EmbeddingModel` is meant to be tagged onto stored vectors by callers so a model swap can trigger re-embed. Treat it as stable; renaming breaks downstream caches.
- The pooled-shape handling in `MLXEmbedderBackend.extractVectors` only knows 2D and 3D. A new pooling path that produces a different rank needs to extend that switch (and throw `unsupportedPoolingShape` otherwise).
