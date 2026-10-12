import Foundation

/// High-level state of an `EmbeddingModel` from the engine's point of view.
public enum ModelState: Sendable {
    /// Neither bundled with the host app nor present in the cache directory.
    /// Calling `prepare()` will trigger a download.
    case notDownloaded

    /// Found on disk — either inside the host app's bundle or in the cache
    /// directory — but not yet loaded into memory.
    case availableOnDisk

    /// A `prepare()` call is currently in progress.
    case loading

    /// Loaded in memory and ready to embed.
    case ready

    /// The most recent `prepare()` call failed. Calling `prepare()` again
    /// will retry from a clean state.
    case failed(Error)
}

/// A single progress event emitted by `SemanticSearchEngine.prepare()`.
public struct PrepareProgress: Sendable {
    /// The stage of `prepare()` an event belongs to.
    public enum Phase: Sendable, Equatable {
        /// Checking the bundle and cache directory for an existing copy.
        case locating
        /// Downloading model weights from HuggingFace.
        case downloading
        /// Loading weights already on disk (bundled or previously downloaded) into the MLX
        /// runtime. Also reported to callers that joined an in-flight load.
        case loadingWeights
        /// Running a single warmup embedding so the first real call is hot.
        case warmingUp
        /// Model is loaded and ready. This is the last event before the stream finishes.
        case done
    }

    /// The stage this event belongs to.
    public let phase: Phase

    /// Estimated completion across the entire `prepare()` call, in `[0.0, 1.0]`.
    public let fractionCompleted: Double

    /// Bytes downloaded so far, when known. The current download stack reports progress per
    /// file rather than per byte, so `SemanticSearchEngine.prepare()` leaves this `nil`; use
    /// `fractionCompleted` for progress UI.
    public let bytesDownloaded: Int64?

    /// Total bytes expected, when known. `nil` from `SemanticSearchEngine.prepare()` for the
    /// same reason as `bytesDownloaded`.
    public let bytesTotal: Int64?

    /// Optional human-readable hint for UI.
    public let message: String?

    /// Creates a progress event.
    public init(
        phase: Phase,
        fractionCompleted: Double,
        bytesDownloaded: Int64? = nil,
        bytesTotal: Int64? = nil,
        message: String? = nil
    ) {
        self.phase = phase
        self.fractionCompleted = fractionCompleted
        self.bytesDownloaded = bytesDownloaded
        self.bytesTotal = bytesTotal
        self.message = message
    }
}
