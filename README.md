# Drew's Swift SemanticSearch

A small Swift package for on-device semantic search on Apple platforms. Wraps an MLX-backed embedding model behind a tiny, easy-to-use API.

> **Heads-up:** in its current state, this package is built for **short, focused inputs** — sentences, short paragraphs, FAQ-style entries, or anything else that fits comfortably in a single embedding. There is no built-in document chunking, so feeding it long-form prose (multi-page documents, full chapters, anything tens of thousands of tokens) will produce one diluted "average" vector for the whole thing and silently lose anything past the model's context limit. See [Limitations and future improvements](#limitations-and-future-improvements) below.

## Why Build This?
Apple's on-device models didn't work well for semantic search.

## Features

- One-call `prepare()` that downloads (if needed), loads, and warms up an embedding model. Reports progress as an `AsyncThrowingStream`.
- Embed single strings or batches with one method. Output vectors are L2-normalized (unit length).
- Pure ranking helper: `VectorMath.match(query:against:top:)` ranks a query vector against a caller-owned set of document vectors, and `SemanticSearchEngine.match(query:against:top:)` embeds the query first. The package never owns your corpus.
- Bundled-or-downloaded model resolution: detects a model shipped inside the host app's bundle, or falls back to a cache directory it manages itself. Weights already on disk load without the network.
- Explicit `unload()` to release weights from memory and `deleteDownloadedWeights()` to remove them from disk.
- GPU memory introspection (`SemanticSearchEngine.gpuMemoryReport()`) and host-controlled MLX cache knobs (`setGPUCacheLimit(bytes:)`, `clearGPUCache()`).
- Default model: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` (1024 dimensions, last-token pooling). Custom models can be described with `EmbeddingModel(...)`; the backend currently implements `.lastToken` pooling only.

## Requirements

- macOS 14, iOS 17, visionOS 1 or newer. Apple platforms only: the package uses Accelerate, Metal and MLX, and does not build on Linux.
- Apple Silicon (the MLX backend requires Metal)
- A Swift 6.3 or newer toolchain. `Package.swift` declares tools version 6.1, but the MLX dependency requires 6.3.

### MLX dependency note

This package depends on patched forks, `drewster99/mlx-swift` and `drewster99/mlx-swift-lm`, which carry a fix for a Metal API Validation assertion (see [ROADMAP.md](ROADMAP.md)). SwiftPM identifies packages by the last component of their URL, so an app that also depends on `ml-explore/mlx-swift` or `ml-explore/mlx-swift-lm` gets a "conflicting identity" warning and only one copy is used. Tracked in [#1](https://github.com/drewster99/swift-semantic-search/issues/1).

## Installation

Add to your `Package.swift`:

```swift
.package(url: "https://github.com/drewster99/swift-semantic-search", from: "0.1.0")
```

And to your target:

```swift
.product(name: "SemanticSearch", package: "swift-semantic-search")
```

## Usage

```swift
import SemanticSearch

let engine = SemanticSearchEngine()  // uses the default model

// Download (if needed) + load + warm up. Progress is observable.
for try await progress in engine.prepare() {
    print("\(progress.phase) \(progress.fractionCompleted)")
}

// Embed text.
let queryVector = try await engine.embed("how to send a message")
let docVectors  = try await engine.embed(batch: documents.map(\.text))

// Rank a query against caller-owned document vectors.
let hits = VectorMath.match(query: queryVector, against: docVectors, top: 5)
for hit in hits {
    print("\(documents[hit.index].title): \(hit.score)")
}

// Free memory.
await engine.unload()

// Wipe the downloaded weights from disk (errors if currently loaded).
try await engine.deleteDownloadedWeights()
```

Store `engine.model.identifier` alongside your document vectors. If it differs on a later launch, the stored vectors came from a different model or pooling scheme and must be re-embedded.

## Errors

Engine methods throw `SemanticSearchError` for the package's own failures (`notReady` before `prepare()` completes, `cannotDeleteWhileLoaded`, `dimensionMismatch`, `nonFiniteEmbedding`, …). Errors from the download and model-loading stack are passed through unchanged. If `unload()` interrupts a `prepare()`, that `prepare()` stream throws `CancellationError`, as does an `embed(batch:)` whose task is cancelled between chunks.

`VectorMath.match` does not throw: it skips documents whose length differs from the query, documents with zero magnitude, and documents whose score is not finite. `top` must not be negative.

## Storage

Downloaded weights live under `Application Support/<bundle identifier>/SemanticSearch/models/` (processes without a bundle identifier, such as command-line tools, share `Application Support/SemanticSearch/SemanticSearch/models/`). The directory is excluded from iCloud and device backups because the weights are re-downloadable. A model directory only counts as present when it holds `config.json`, `tokenizer.json`, `tokenizer_config.json` and at least one `.safetensors` file, so an interrupted download is reported as `.notDownloaded`.

## Running the tests

The unit suites (`VectorMath`, `VectorMathRanking`, `EngineLifecycle`, `ModelLocator`,
`BackendLoading`) need neither the network nor a GPU:

```sh
swift test --filter 'VectorMath|EngineLifecycle|EngineJoin|ModelLocator|BackendLoading'
```

The end-to-end `Integration` suite loads the real MLX model and exercises
`prepare()` + `embed()` + `match()` against a curated corpus. It needs Apple Silicon,
and the network on first run. With a recent toolchain (verified with Swift 6.4 / Xcode 27.1)
plain `swift test` runs it. With older toolchains SwiftPM may not produce MLX's metallib; run
it through Xcode's build pipeline instead:

```sh
xcodebuild test \
    -scheme SemanticSearch \
    -destination 'platform=macOS' \
    -only-testing:SemanticSearchTests/SemanticSearchIntegrationTests
```

The first run downloads the default Qwen3 embedding weights (about 320 MB) into the
test host's Application Support cache; subsequent runs reuse the cache and
finish in a few seconds.

## Limitations and future improvements

### Document chunking (the big one)

This package takes a `String` and produces one vector. Today there is no chunking, no truncation, and no awareness of the model's context window. In `cli-demo` and `gui-demo` each whole file becomes one document → one vector. If a file exceeds the model's context (~32K tokens for Qwen3-Embedding-0.6B), behavior is at the mercy of the tokenizer + MLX model — typically silent truncation, so anything past the cutoff is invisible to retrieval.

Why "one big chunk" is the wrong default for long documents:

- A single embedding is a 1024-dim summary of *everything* the chunk contains. With 20K tokens of mixed content the vector becomes a generic-topic average that ranks high for nothing in particular. A query about a paragraph buried on page 30 will not find it.
- Truncation makes content past the model's limit invisible.

The standard playbook in semantic-search / RAG systems is to split long documents into smaller chunks before embedding, then store one vector per chunk with metadata pointing back at the source file:

| Strategy                                              | When                                                                                        |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Whole-document** (what we do today)                 | Short, focused inputs — FAQs, sentences, brief notes.                                       |
| **Fixed-size chunks, non-overlapping**                | Simple and cheap; risks splitting mid-sentence.                                             |
| **Fixed-size + overlap** (typical: 256–512 tokens, ~10–20% overlap) | Solid general-purpose default. The overlap recovers boundary cases.            |
| **Semantic chunking** (split on paragraph / section / heading) | Best fidelity for prose; needs a parser per format.                                |
| **Hierarchical** (sentence → paragraph → document)    | Highest quality. You store more vectors and index in two passes (coarse-to-fine retrieval). |

For a markdown corpus, the practical sweet spot is **paragraph-aware chunks of ~300–500 tokens with one paragraph of overlap**, indexing the chunk's parent file (and a character/line offset) in metadata so a hit can point back at *where* in the source it came from.

How this would land in the package: a `Chunker` utility (token-aware split with overlap) on the library side, with the embedding backend unchanged — chunking is a layer above one-string-in / one-vector-out. `gui-demo` would then split each file into chunks, embed all chunks, and a result row would surface the chunk text plus a "from filename.md" subtitle.

### Other rough edges

- The two demo targets use different on-disk caches because `Bundle.main.bundleIdentifier` differs (CLI tools have no bundle id; the SwiftUI app does). First launch of each will re-download the weights into its own cache directory.
- `cli-demo` and `gui-demo` are built with Xcode (the `Examples/SemanticSearchDemo.xcodeproj`), not via `swift run`; they are not SwiftPM targets.
- Embedding runs synchronously on a Swift Concurrency thread for the duration of each forward pass, and inputs are not truncated to the model's context window.

## License

MIT. See [LICENSE](LICENSE).
