# SemanticSearch

A small Swift package for on-device semantic search on Apple platforms. Wraps an MLX-backed embedding model behind a tiny, easy-to-use API.

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
Metal shaders, so the metallib isn't present at runtime:

```sh
xcodebuild test \
    -scheme SemanticSearch \
    -destination 'platform=macOS' \
    -only-testing:SemanticSearchTests/SemanticSearchIntegrationTests
```

The first run downloads the default Qwen3 embedding weights (~330 MB) into the
test host's Application Support cache; subsequent runs reuse the cache and
finish in a few seconds.

## License

MIT. See [LICENSE](LICENSE).
