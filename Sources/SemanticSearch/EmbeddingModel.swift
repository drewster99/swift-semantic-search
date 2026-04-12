import Foundation

/// Describes a single embedding model that `SemanticSearchEngine` knows how to load.
///
/// A model is identified by a stable string (`identifier`) used for cache directory
/// names and for tagging stored vectors so callers can detect a model swap and
/// re-embed accordingly. Models are sourced from a HuggingFace repository, but the
/// host app can also ship a model inside its bundle (under `bundleSubdirectory`)
/// to skip the first-launch download.
public struct EmbeddingModel: Sendable, Hashable {
    /// Stable, short identifier — used as a directory name and as the value tagged
    /// onto stored vectors so callers can detect a model swap.
    public let identifier: String

    /// Human-readable name for UI.
    public let displayName: String

    /// Output vector dimension.
    public let dimension: Int

    /// HuggingFace repository, e.g. `"mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"`.
    public let huggingFaceRepo: String

    /// Subdirectory inside `Bundle.main` where this model may be found pre-bundled.
    /// `SemanticSearchEngine` checks this location first; if present, no download
    /// is required and the model loads from the bundle.
    public let bundleSubdirectory: String

    public init(
        identifier: String,
        displayName: String,
        dimension: Int,
        huggingFaceRepo: String,
        bundleSubdirectory: String
    ) {
        self.identifier = identifier
        self.displayName = displayName
        self.dimension = dimension
        self.huggingFaceRepo = huggingFaceRepo
        self.bundleSubdirectory = bundleSubdirectory
    }
}

extension EmbeddingModel {
    /// Qwen3 Embedding 0.6B, 4-bit DWQ quantization. Runs on Apple Silicon via MLX.
    public static let qwen3Embedding_0_6B_4bitDWQ = EmbeddingModel(
        identifier: "qwen3-embedding-0.6b-4bit-dwq",
        displayName: "Qwen3 Embedding 0.6B (4-bit DWQ)",
        dimension: 1024,
        huggingFaceRepo: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        bundleSubdirectory: "Models/qwen3-embedding-0.6b-4bit-dwq"
    )

    /// The default model used by `SemanticSearchEngine()` when no model is specified.
    public static let `default`: EmbeddingModel = .qwen3Embedding_0_6B_4bitDWQ
}
