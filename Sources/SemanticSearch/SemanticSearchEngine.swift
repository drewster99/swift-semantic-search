import Foundation
import Hub
import MLXEmbedders

/// Tunable knobs for `SemanticSearchEngine`.
public struct EngineOptions: Sendable {
    /// Maximum number of strings to feed into a single MLX forward pass when
    /// `embed(batch:)` is called with more inputs than this. Defaults to 16,
    /// which mirrors the value used by the `embedder-tool` reference
    /// implementation but slightly higher to take advantage of M-series throughput.
    /// Larger batches are faster per-call but use more memory during the forward pass.
    public var maxBatchSize: Int

    public init(maxBatchSize: Int = 16) {
        self.maxBatchSize = maxBatchSize
    }
}

/// On-device semantic search engine. Holds an embedding model and exposes a tiny
/// API for embedding text and ranking results.
///
/// Lifecycle:
/// 1. Create an engine with `init(model:options:)`. Nothing loads yet.
/// 2. Call `prepare()` and consume the returned `AsyncThrowingStream<PrepareProgress, Error>`
///    to drive a progress UI. Downloads the model if needed, loads weights, runs warmup.
/// 3. Call `embed(_:)` / `embed(batch:)` / `match(...)`. The engine is an actor, so
///    concurrent calls serialize on the GPU automatically.
/// 4. Optionally call `unload()` to release weights from memory, or
///    `deleteDownloadedWeights()` to remove them from disk.
public actor SemanticSearchEngine {
    /// The model this engine was created with. Constant for the engine's lifetime.
    public nonisolated let model: EmbeddingModel

    /// Tunable knobs for batching, etc.
    public nonisolated let options: EngineOptions

    private let locator: ModelLocator
    private var internalState: InternalState = .idle

    private enum InternalState {
        case idle
        case loading(Task<MLXEmbedderBackend, Error>)
        case loaded(MLXEmbedderBackend)
        case failed(Error)
    }

    public init(model: EmbeddingModel = .default, options: EngineOptions = EngineOptions()) {
        self.model = model
        self.options = options
        self.locator = ModelLocator(model: model)
    }

    /// Current public state. Cheap — does not load anything.
    public var state: ModelState {
        switch internalState {
        case .idle:
            switch locator.locate() {
            case .bundled, .downloaded: return .availableOnDisk
            case .absent: return .notDownloaded
            }
        case .loading: return .loading
        case .loaded: return .ready
        case .failed(let error): return .failed(error)
        }
    }

    // MARK: - Prepare

    /// Downloads (if necessary), loads, and warms up the model. Returns an
    /// `AsyncThrowingStream` of progress events; iterate it to drive a progress UI.
    /// The stream finishes after the `.done` event is emitted.
    ///
    /// `prepare()` is idempotent: calling it on a loaded engine emits a single
    /// `.done` event. Calling it during an in-flight load joins the existing
    /// load and emits a single completion event when it finishes (granular
    /// progress is only seen by the first caller).
    public nonisolated func prepare() -> AsyncThrowingStream<PrepareProgress, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try await self.runPrepare(continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func runPrepare(
        continuation: AsyncThrowingStream<PrepareProgress, Error>.Continuation
    ) async throws {
        if case .loaded = internalState {
            continuation.yield(
                PrepareProgress(phase: .done, fractionCompleted: 1.0, message: "Already loaded")
            )
            return
        }

        if case .loading(let existingTask) = internalState {
            continuation.yield(
                PrepareProgress(
                    phase: .loadingWeights,
                    fractionCompleted: 0.5,
                    message: "Joining in-flight load"
                )
            )
            _ = try await existingTask.value
            continuation.yield(PrepareProgress(phase: .done, fractionCompleted: 1.0))
            return
        }

        let loadTask = Task<MLXEmbedderBackend, Error> { [self] in
            try await self.performLoad(continuation: continuation)
        }
        internalState = .loading(loadTask)

        do {
            let backend = try await loadTask.value
            internalState = .loaded(backend)
            continuation.yield(PrepareProgress(phase: .done, fractionCompleted: 1.0))
        } catch {
            internalState = .failed(error)
            throw error
        }
    }

    private func performLoad(
        continuation: AsyncThrowingStream<PrepareProgress, Error>.Continuation
    ) async throws -> MLXEmbedderBackend {
        continuation.yield(PrepareProgress(phase: .locating, fractionCompleted: 0.02))

        let location = locator.locate()

        let modelRepo = model.huggingFaceRepo
        let configuration: MLXEmbedders.ModelConfiguration = await MainActor.run {
            switch location {
            case .bundled(let url):
                return ModelConfiguration(directory: url)
            case .downloaded, .absent:
                return ModelConfiguration.configuration(id: modelRepo)
            }
        }

        let cacheBase = try locator.cacheBaseURL()
        let hub = HubApi(downloadBase: cacheBase)

        continuation.yield(PrepareProgress(phase: .downloading, fractionCompleted: 0.05))

        let container = try await MLXEmbedders.loadModelContainer(
            hub: hub,
            configuration: configuration,
            progressHandler: { progress in
                // Reserve 5% for locate, 85% for download/load, 10% for warmup.
                let scaled = 0.05 + 0.85 * progress.fractionCompleted
                continuation.yield(
                    PrepareProgress(
                        phase: .downloading,
                        fractionCompleted: scaled,
                        bytesDownloaded: progress.completedUnitCount,
                        bytesTotal: progress.totalUnitCount > 0 ? progress.totalUnitCount : nil
                    )
                )
            }
        )

        continuation.yield(PrepareProgress(phase: .warmingUp, fractionCompleted: 0.92))

        let backend = MLXEmbedderBackend(
            modelIdentifier: model.identifier,
            dimension: model.dimension,
            container: container
        )

        _ = try await backend.embed(batch: ["warmup"])

        return backend
    }

    // MARK: - Embed

    /// Embeds a single string. Returns an L2-normalized `[Float]` of length
    /// `model.dimension`.
    public func embed(_ text: String) async throws -> [Float] {
        let result = try await embed(batch: [text])
        guard let first = result.first else {
            throw SemanticSearchError.embeddingFailed("Backend returned no vectors for single embed")
        }
        return first
    }

    /// Embeds a batch of strings. Internally chunks at `options.maxBatchSize`,
    /// running one MLX forward pass per chunk. Returned vectors are in input order.
    public func embed(batch texts: [String]) async throws -> [[Float]] {
        let backend = try requireLoadedBackend()
        guard !texts.isEmpty else { return [] }

        let chunkSize = max(1, options.maxBatchSize)
        if texts.count <= chunkSize {
            return try await backend.embed(batch: texts)
        }

        var results: [[Float]] = []
        results.reserveCapacity(texts.count)
        var index = 0
        while index < texts.count {
            let end = min(index + chunkSize, texts.count)
            let chunk = Array(texts[index..<end])
            let chunkResults = try await backend.embed(batch: chunk)
            results.append(contentsOf: chunkResults)
            index = end
        }
        return results
    }

    /// Convenience: embed `query` and rank it against caller-owned document
    /// vectors via `VectorMath.match`. The package never owns your corpus —
    /// `documents` is whatever you pass in.
    public func match(
        query: String,
        against documents: [[Float]],
        top: Int? = nil
    ) async throws -> [(index: Int, score: Float)] {
        let queryVector = try await embed(query)
        return VectorMath.match(query: queryVector, against: documents, top: top)
    }

    // MARK: - Unload / Delete

    /// Releases the loaded model from memory. If a `prepare()` call is currently
    /// in flight, it is cancelled. After this call, `state` returns
    /// `.availableOnDisk` (or `.notDownloaded` if the cache was wiped externally).
    public func unload() {
        if case .loading(let task) = internalState {
            task.cancel()
        }
        internalState = .idle
    }

    /// Removes the downloaded model files from the cache directory. Bundled
    /// model files (shipped inside the host app's bundle) are never touched.
    /// Throws `SemanticSearchError.cannotDeleteWhileLoaded` if the model is
    /// currently loaded or loading — call `unload()` first.
    public func deleteDownloadedWeights() throws {
        switch internalState {
        case .loaded, .loading:
            throw SemanticSearchError.cannotDeleteWhileLoaded
        case .idle, .failed:
            try locator.deleteDownloaded()
        }
    }

    // MARK: - Internals

    private func requireLoadedBackend() throws -> MLXEmbedderBackend {
        if case .loaded(let backend) = internalState {
            return backend
        }
        throw SemanticSearchError.notReady
    }
}
