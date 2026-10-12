import Foundation
import Hub
import MLXEmbedders

/// Produces a ready-to-use `EmbeddingBackend` for a model. Separated from
/// `SemanticSearchEngine` so the engine's lifecycle (coalescing, unload, generations) can be
/// tested without loading real weights.
internal protocol EmbeddingBackendLoader: Sendable {
    /// Locates or downloads, loads, and warms up `model`, reporting progress on `progress`.
    func loadBackend(
        for model: EmbeddingModel,
        locator: ModelLocator,
        progress: AsyncThrowingStream<PrepareProgress, Error>.Continuation
    ) async throws -> any EmbeddingBackend
}

/// Fractions of `PrepareProgress.fractionCompleted` assigned to each load phase.
enum PrepareProgressFraction {
    static let locating = 0.02
    /// Start of the download / weight-load span.
    static let loadStart = 0.05
    /// Width of the download / weight-load span.
    static let loadSpan = 0.85
    static let warmingUp = 0.92
    static let done = 1.0
    /// Reported to callers that join a load already in flight; they get no granular progress.
    static let joinedInFlightLoad = 0.5
}

/// Loads models through `MLXEmbedders`.
internal struct MLXEmbedderBackendLoader: EmbeddingBackendLoader {
    func loadBackend(
        for model: EmbeddingModel,
        locator: ModelLocator,
        progress continuation: AsyncThrowingStream<PrepareProgress, Error>.Continuation
    ) async throws -> any EmbeddingBackend {
        continuation.yield(PrepareProgress(phase: .locating, fractionCompleted: PrepareProgressFraction.locating))

        let location = locator.locate()
        let registered = await MainActor.run {
            MLXEmbedders.ModelConfiguration.configuration(id: model.huggingFaceRepo)
        }
        let configuration = Self.configuration(for: location, registered: registered)

        // A bundled model never touches the cache, so it must not fail on cache-directory
        // creation or the backup-exclusion flag; the Hub only needs the path.
        let cacheBase: URL
        if case .bundled = location {
            cacheBase = locator.cacheBaseURL()
        } else {
            cacheBase = try locator.prepareCacheBaseURL()
        }
        let hub = HubApi(downloadBase: cacheBase)
        let loadPhase = Self.loadPhase(for: location)

        continuation.yield(PrepareProgress(phase: loadPhase, fractionCompleted: PrepareProgressFraction.loadStart))

        let loadStart = Date()
        let container = try await MLXEmbedders.loadModelContainer(
            hub: hub,
            configuration: configuration,
            progressHandler: { progress in
                // The Hub reports snapshot progress in FILES, not bytes, so no byte counts are
                // available here; `bytesDownloaded` / `bytesTotal` stay nil rather than lie.
                continuation.yield(
                    PrepareProgress(
                        phase: loadPhase,
                        fractionCompleted: PrepareProgressFraction.loadStart
                            + PrepareProgressFraction.loadSpan * progress.fractionCompleted
                    )
                )
            }
        )

        let loadSeconds = String(format: "%.1f", Date().timeIntervalSince(loadStart))
        let loadKind = location.isOnDisk ? "load (on disk)" : "download + load"
        engineLogger.notice("model \(loadKind, privacy: .public) took \(loadSeconds, privacy: .public)s")

        try Task.checkCancellation()
        continuation.yield(PrepareProgress(phase: .warmingUp, fractionCompleted: PrepareProgressFraction.warmingUp))

        let backend = MLXEmbedderBackend(
            modelIdentifier: model.identifier,
            dimension: model.dimension,
            container: container,
            pooling: model.pooling
        )

        let warmStart = Date()
        _ = try await backend.embed(batch: ["warmup"])
        let warmSeconds = String(format: "%.1f", Date().timeIntervalSince(warmStart))
        engineLogger.notice(
            "warmup (first forward + Metal kernel compile) took \(warmSeconds, privacy: .public)s")

        return backend
    }

    /// Chooses how `MLXEmbedders` should load the model. Weights already on disk load from
    /// their directory, which needs no network: going through the Hub id would re-list the
    /// repository on every launch and fail on any network error other than "not connected"
    /// (captive portal, DNS failure, timeout, server error) even though every file is local.
    ///
    /// The one exception is a registry entry that pulls its tokenizer from a different
    /// repository — that tokenizer is not in the model directory, so the Hub path is required.
    static func configuration(
        for location: ModelLocator.Location,
        registered: MLXEmbedders.ModelConfiguration
    ) -> MLXEmbedders.ModelConfiguration {
        switch location {
        case .bundled(let url):
            return MLXEmbedders.ModelConfiguration(directory: url)
        case .downloaded(let url):
            guard registered.tokenizerId == nil else { return registered }
            return MLXEmbedders.ModelConfiguration(
                directory: url, overrideTokenizer: registered.overrideTokenizer)
        case .absent:
            return registered
        }
    }

    /// `.downloading` only when a download is actually needed; otherwise the span is weight loading.
    static func loadPhase(for location: ModelLocator.Location) -> PrepareProgress.Phase {
        location.isOnDisk ? .loadingWeights : .downloading
    }
}

extension ModelLocator.Location {
    /// Whether the model is already complete on disk (bundled or downloaded).
    var isOnDisk: Bool {
        switch self {
        case .bundled, .downloaded: return true
        case .absent: return false
        }
    }
}
