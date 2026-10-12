# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`SemanticSearch` is a standalone Swift **package** (not an app) that wraps an MLX-backed embedding model behind a small public API. It targets macOS 14 / iOS 17 / visionOS 1 on Apple Silicon. There is no `.xcodeproj` — consumers add it as a SwiftPM dependency.

The parent `/Users/andrew/cursor/CLAUDE.md` instructs using `drews-xcode-mcp` for Xcode projects. That does not apply here: this is SwiftPM. Use `swift build` / `swift test` from the command line, or `xcodebuild` when MLX's Metal shaders need to be compiled (see Testing).

## Commands

```sh
# Unit tests (no network, no GPU) — fast.
swift test --filter 'VectorMath|EngineLifecycle|EngineJoin|ModelLocator|BackendLoading'

# Build the package.
swift build

# Everything, including the Integration suite (Apple Silicon; network on first run).
# Verified to work under plain `swift test` with Swift 6.4 / Xcode 27.1. On older toolchains
# that do not produce MLX's metallib, use Xcode's build pipeline:
swift test
xcodebuild test \
    -scheme SemanticSearch \
    -destination 'platform=macOS' \
    -only-testing:SemanticSearchTests/SemanticSearchIntegrationTests
```

First integration run downloads the default Qwen3 weights (about 320 MB) into `Application Support/<bundleID>/SemanticSearch/models/`; subsequent runs reuse that cache.

## Architecture

`SemanticSearchEngine` is an `actor` that holds lifecycle state (`idle` → `loading` → `loaded` / `failed`) and serializes GPU access. Its only job is: resolve model location → load weights → run forward passes. It deliberately does **not** own a corpus — `VectorMath.match` ranks a query against caller-supplied document vectors. Each load carries a generation number; after every `await`, a load may only publish its result if it is still the current generation (`unload()` and newer loads supersede it). The starting caller and all joiners go through the single `publish(_:generation:)` helper; whichever resumes first writes the state, the rest observe it. Loading itself lives behind the internal `EmbeddingBackendLoader` protocol (`MLXEmbedderBackendLoader` in production) so lifecycle tests can inject a fake.

Load flow (`SemanticSearchEngine.prepare()`):
1. `ModelLocator.locate()` checks, in order: `Bundle.main/<model.bundleSubdirectory>`, then the MLX cache directory, then absent. A location only counts as "present" when it holds `config.json`, `tokenizer.json`, `tokenizer_config.json` and a `.safetensors` file. `locate()` creates nothing; `prepareCacheBaseURL()` creates the cache base and excludes it from backups.
2. `MLXEmbedders.loadModelContainer` loads weights on disk from their directory (no network), or downloads through the Hub id when absent, reporting progress through the `AsyncThrowingStream<PrepareProgress, Error>` returned to the caller.
3. A single `"warmup"` embedding runs so the first real call is hot.

`prepare()` is idempotent and coalesces concurrent callers onto the in-flight load task.

Embed flow (`MLXEmbedderBackend.embed(batch:)`):
- Tokenize (the Qwen3 tokenizer appends `<|endoftext|>`) → right-pad to the batch max with `eosTokenId` (no attention mask: Qwen3 is causal) → forward → pool per the model's `PoolingStrategy` (only `.lastToken` is implemented: gather each row's last real token) → widen to Float32 → L2-normalize → validate (dimension, finite) → extract Float vectors.
- `extractVectors` handles 2D `(batch, dim)` and 3D `(batch, seq, dim)` (mean-pooled across seq); `.lastToken` produces 2D.

Cache layout:
- `ModelLocator.canonicalDownloadDirectory()` asks `MLXEmbedders.ModelConfiguration.modelDirectory(hub:)` for the path instead of hand-rolling the HuggingFace layout. Don't mirror the layout manually — the library owns it.
- `deleteDownloadedWeights()` refuses while the model is loaded or loading, or while a load cancelled by `unload()` is still winding down (raise `cannotDeleteWhileLoaded`). Bundled files are never deleted.

`VectorMath` is a stateless `enum` using `Accelerate`/`vDSP` for dot product, magnitude, L2 normalization, cosine similarity, and ranked `match(...)`. `match` skips documents whose length differs from the query, zero-magnitude documents and non-finite scores; ties order by ascending index.

## Non-obvious constraints

- **mlx-swift-lm is pinned tightly** to the patched fork, `.upToNextMinor(from: "2.31.4")`. The 3.x line rewrote `MLXEmbedders.loadModelContainer`'s signature (takes `Downloader`/`TokenizerLoader` instead of `HubApi`). Do not move to 3.x without migrating the call site in `MLXEmbedderBackendLoader.loadBackend`. See ROADMAP.md.
- **Adding a new model** is one `static let` on `EmbeddingModel`, as long as the model is MLX-compatible, sits in a HuggingFace repo, and uses `.lastToken` pooling; other pooling strategies need a backend implementation first.
- **`identifier`** on `EmbeddingModel` is meant to be tagged onto stored vectors by callers so a model swap can trigger re-embed. Treat it as stable; renaming breaks downstream caches.
- The pooled-shape handling in `MLXEmbedderBackend.extractVectors` only knows 2D and 3D. A new pooling path that produces a different rank needs to extend that switch (and throw `unsupportedPoolingShape` otherwise).
