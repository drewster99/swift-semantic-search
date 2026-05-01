# Drew's Swift SemanticSearch

A small Swift package for on-device semantic search on Apple platforms. Wraps an MLX-backed embedding model behind a tiny, easy-to-use API.

> **Heads-up:** in its current state, this package is built for **short, focused inputs** — sentences, short paragraphs, FAQ-style entries, or anything else that fits comfortably in a single embedding. There is no built-in document chunking, so feeding it long-form prose (multi-page documents, full chapters, anything tens of thousands of tokens) will produce one diluted "average" vector for the whole thing and silently lose anything past the model's context limit. See [Limitations and future improvements](#limitations-and-future-improvements) below.

## Why Build This?
Apple's models didn't work well for a semantic search purposes.

## Features

- One-call `prepare()` that downloads (if needed), loads, and warms up an embedding model. Reports progress as an `AsyncThrowingStream`.
- Embed single strings or batches with one method.
- Pure ranking helper: `match(query:against:top:)` ranks a query vector against a caller-owned set of document vectors. The package never owns your corpus.
- Bundled-or-downloaded model resolution: detects a model shipped inside the host app's bundle, or falls back to a cache directory it manages itself.
- Explicit `unload()` to release weights from memory and `deleteDownloadedWeights()` to remove them from disk.
- Default model: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`. Adding more models is one static on `EmbeddingModel`.

## Requirements

- macOS 14, iOS 17, visionOS 1 or newer
- Apple Silicon (the MLX backend requires Metal)

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

## Running the tests

The pure-logic `VectorMath` tests run anywhere:

```sh
swift test --filter VectorMath
```

The end-to-end `Integration` suite loads the real MLX model and exercises
`prepare()` + `embed()` + `match()` against a curated corpus. It must be run
through Xcode's build pipeline — `swift test` on its own cannot compile MLX's
Metal shaders, so the metallib isn't present at runtime.

Note: These tests will take several minutes:

```sh
xcodebuild test \
    -scheme SemanticSearch \
    -destination 'platform=macOS' \
    -only-testing:SemanticSearchTests/SemanticSearchIntegrationTests
```

The first run downloads the default Qwen3 embedding weights (~330 MB) into the
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
- `cli-demo` and `gui-demo` are built with Xcode (the `Examples/SemanticSearchDemo.xcodeproj`), not via `swift run`. Plain `swift build`/`swift run` cannot compile MLX's Metal shaders, so the metallib is missing at runtime — this is a SwiftPM-CLI vs. Xcode gap, not something this package can fix on its own.

## License

MIT. See [LICENSE](LICENSE).
