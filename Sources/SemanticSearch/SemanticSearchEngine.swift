import Foundation

/// Tunable knobs for `SemanticSearchEngine`.
public struct EngineOptions: Sendable {
    /// Maximum number of strings fed into a single MLX forward pass when `embed(batch:)` is
    /// called with more inputs than this. Larger batches are faster per string but use more
    /// memory during the forward pass, because every string in a batch is padded to the
    /// longest one. Values below 1 are treated as 1.
    public var maxBatchSize: Int

    /// Creates options. `maxBatchSize` defaults to 16.
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
///
/// The engine's own failures are thrown as `SemanticSearchError`. Also passed through
/// unchanged: errors from the download / model-loading stack, Foundation file-system errors
/// from creating the cache directory (`prepare()`) and removing it
/// (`deleteDownloadedWeights()`), and `CancellationError`.
public actor SemanticSearchEngine {
    /// The model this engine was created with. Constant for the engine's lifetime.
    public nonisolated let model: EmbeddingModel

    /// Tunable knobs for batching, etc.
    public nonisolated let options: EngineOptions

    private let locator: ModelLocator
    private let backendLoader: any EmbeddingBackendLoader
    private var internalState: InternalState = .idle

    /// Incremented for every load started. A load may only publish its result while it is still
    /// the current generation; `unload()` and later loads supersede it.
    private var loadGeneration: UInt64 = 0

    /// Load tasks that have started and not yet finished, including ones `unload()` cancelled.
    /// Cancellation is cooperative, so a cancelled load can keep touching the cache directory
    /// for a while; deleting weights under it is refused until it finishes.
    private var unfinishedLoadCount = 0

    /// `prepare()` callers currently waiting on a load another caller started. Lets tests
    /// release a load only after a joiner is attached to it.
    private(set) var joinedLoadWaiterCount = 0

    private enum InternalState {
        case idle
        case loading(Task<any EmbeddingBackend, Error>, generation: UInt64)
        case loaded(any EmbeddingBackend)
        case failed(Error)
    }

    /// Creates an engine for `model`. Nothing is loaded until `prepare()` is called.
    public init(model: EmbeddingModel = .default, options: EngineOptions = EngineOptions()) {
        self.init(model: model, options: options, locator: ModelLocator(model: model), backendLoader: MLXEmbedderBackendLoader())
    }

    init(
        model: EmbeddingModel,
        options: EngineOptions,
        locator: ModelLocator,
        backendLoader: any EmbeddingBackendLoader
    ) {
        self.model = model
        self.options = options
        self.locator = locator
        self.backendLoader = backendLoader
    }

    /// On-disk location of the model files, if the engine can find a complete copy.
    /// Returns the bundled URL if the model is shipped inside the host app,
    /// the cache URL once the model has been downloaded, or `nil` if no complete copy
    /// exists on disk yet. Cheap — checks the file system, loads and creates nothing.
    public nonisolated func modelLocationOnDisk() -> URL? {
        switch locator.locate() {
        case .bundled(let url), .downloaded(let url):
            return url
        case .absent:
            return nil
        }
    }

    /// Current public state. Cheap — checks the file system, loads and creates nothing.
    public var state: ModelState {
        switch internalState {
        case .idle:
            return locator.locate().isOnDisk ? .availableOnDisk : .notDownloaded
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
    ///
    /// The engine is kept alive until the stream finishes. If `unload()` is called before the
    /// load completes, the stream throws `CancellationError` and the engine stays unloaded.
    /// Ending iteration early does not cancel the load; call `unload()` for that.
    public nonisolated func prepare() -> AsyncThrowingStream<PrepareProgress, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
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
        switch internalState {
        case .loaded:
            continuation.yield(
                PrepareProgress(phase: .done, fractionCompleted: PrepareProgressFraction.done, message: "Already loaded")
            )
            return

        case .loading(let existingTask, let generation):
            continuation.yield(
                PrepareProgress(
                    phase: .loadingWeights,
                    fractionCompleted: PrepareProgressFraction.joinedInFlightLoad,
                    message: "Joining in-flight load"
                )
            )
            joinedLoadWaiterCount += 1
            let result = await existingTask.result
            joinedLoadWaiterCount -= 1
            try publish(result, generation: generation)
            continuation.yield(PrepareProgress(phase: .done, fractionCompleted: PrepareProgressFraction.done))
            return

        case .idle, .failed:
            break
        }

        loadGeneration &+= 1
        let generation = loadGeneration
        let model = self.model
        let locator = self.locator
        let backendLoader = self.backendLoader
        let loadTask = Task<any EmbeddingBackend, Error> {
            try await backendLoader.loadBackend(for: model, locator: locator, progress: continuation)
        }
        internalState = .loading(loadTask, generation: generation)
        unfinishedLoadCount += 1

        let result = await loadTask.result
        unfinishedLoadCount -= 1
        try publish(result, generation: generation)
        continuation.yield(PrepareProgress(phase: .done, fractionCompleted: PrepareProgressFraction.done))
    }

    /// Single point where a finished load's outcome reaches `internalState`, shared by the caller
    /// that started the load and every caller that joined it. Whichever of them resumes first
    /// publishes; the rest observe what was published. Actor reentrancy means `unload()` or a
    /// newer `prepare()` may have run during their await, in which case the result is discarded.
    ///
    /// Returns normally when the load's generation ended up loaded; rethrows the load's error
    /// when it ended up failed; throws `CancellationError` when it was superseded.
    private func publish(_ result: Result<any EmbeddingBackend, Error>, generation: UInt64) throws {
        if isCurrentLoad(generation) {
            switch result {
            case .success(let backend):
                internalState = .loaded(backend)
                return
            case .failure(let error):
                internalState = .failed(error)
                throw error
            }
        }
        guard generation == loadGeneration else {
            throw CancellationError()
        }
        switch internalState {
        case .loaded:
            return
        case .failed(let error):
            throw error
        case .idle, .loading:
            throw CancellationError()
        }
    }

    private func isCurrentLoad(_ generation: UInt64) -> Bool {
        if case .loading(_, let current) = internalState {
            return current == generation
        }
        return false
    }

    // MARK: - Embed

    /// Embeds a single string. Returns an L2-normalized `[Float]` of length
    /// `model.dimension`.
    ///
    /// Throws `SemanticSearchError.notReady` unless `prepare()` has completed.
    public func embed(_ text: String) async throws -> [Float] {
        let result = try await embed(batch: [text])
        guard let first = result.first else {
            throw SemanticSearchError.embeddingFailed("Backend returned no vectors for single embed")
        }
        return first
    }

    /// Embeds a batch of strings. Internally chunks at `options.maxBatchSize`,
    /// running one MLX forward pass per chunk. Returned vectors are in input order.
    ///
    /// Throws `SemanticSearchError.notReady` unless `prepare()` has completed. Cancellation is
    /// checked between chunks, so cancelling the calling task stops a large batch early with
    /// `CancellationError` (the chunk already running on the GPU finishes first).
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
            try Task.checkCancellation()
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
    ///
    /// - Parameters:
    ///   - query: The text to search for.
    ///   - documents: Caller-owned document vectors, normally produced by `embed(batch:)`.
    ///   - top: Maximum number of results; `nil` returns every scored document. Must not be negative.
    /// - Returns: `(index, score)` pairs sorted by cosine similarity, highest first. See
    ///   `VectorMath.match` for which documents are skipped.
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
    /// in flight, it is cancelled and its stream throws `CancellationError`; its result is
    /// discarded even if the underlying load finishes anyway. After this call, `state` returns
    /// `.availableOnDisk` (or `.notDownloaded` if no complete copy is on disk).
    public func unload() {
        if case .loading(let task, _) = internalState {
            task.cancel()
        }
        internalState = .idle
    }

    /// Removes the downloaded model files from the cache directory. Bundled
    /// model files (shipped inside the host app's bundle) are never touched.
    /// Throws `SemanticSearchError.cannotDeleteWhileLoaded` if the model is
    /// currently loaded or loading — call `unload()` first — or if a load cancelled by
    /// `unload()` has not finished winding down yet; retry once it has.
    public func deleteDownloadedWeights() throws {
        switch internalState {
        case .loaded, .loading:
            throw SemanticSearchError.cannotDeleteWhileLoaded
        case .idle, .failed:
            guard unfinishedLoadCount == 0 else {
                throw SemanticSearchError.cannotDeleteWhileLoaded
            }
            try locator.deleteDownloaded()
        }
    }

    // MARK: - Internals

    private func requireLoadedBackend() throws -> any EmbeddingBackend {
        if case .loaded(let backend) = internalState {
            return backend
        }
        throw SemanticSearchError.notReady
    }
}
