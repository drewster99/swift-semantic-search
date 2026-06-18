import Foundation

/// Describes a single embedding model that `SemanticSearchEngine` knows how to load.
///
/// A model carries a stable `identifier` (the embedding-output signature — for tagging stored
/// vectors so callers can detect a swap and re-embed) and the `pooling` strategy its architecture
/// requires. Models are sourced from a HuggingFace repository, but the host app can also ship a
/// model inside its bundle (under `bundleSubdirectory`) to skip the first-launch download.

/// How a model's per-token hidden states are reduced to one sentence embedding. This is a property
/// of the model's architecture — pooling the wrong way silently produces degraded embeddings — so
/// every `EmbeddingModel` must declare it explicitly, and the backend fails loudly on a strategy it
/// hasn't implemented rather than guessing.
public enum PoolingStrategy: String, Sendable, Hashable {
    /// Last real (EOS) token. Correct for causal models such as Qwen3-Embedding.
    case lastToken
    /// Mean over the real (non-pad) tokens. Typical for bidirectional models without a CLS head.
    case mean
    /// First (CLS) token. For BERT-style models trained with a CLS pooling head.
    case cls
}

public struct EmbeddingModel: Sendable, Hashable {
    /// Stable identifier for the *embedding output* — encodes the model, its quantization, AND
    /// the pooling scheme, so it changes whenever a produced vector would differ. NOT a cache key:
    /// weights are located by `huggingFaceRepo` / `bundleSubdirectory`, so bumping this never
    /// triggers a re-download. Persist it alongside stored vectors; a mismatch on load means those
    /// vectors are stale and must be re-embedded.
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

    /// How the model's per-token hidden states are pooled into one vector. Required — the backend
    /// cannot infer it safely, so adding a model forces this decision. See `PoolingStrategy`.
    public let pooling: PoolingStrategy

    public init(
        identifier: String,
        displayName: String,
        dimension: Int,
        huggingFaceRepo: String,
        bundleSubdirectory: String,
        pooling: PoolingStrategy
    ) {
        self.identifier = identifier
        self.displayName = displayName
        self.dimension = dimension
        self.huggingFaceRepo = huggingFaceRepo
        self.bundleSubdirectory = bundleSubdirectory
        self.pooling = pooling
    }
}

extension EmbeddingModel {
    /// Qwen3 Embedding 0.6B, 4-bit DWQ quantization. Runs on Apple Silicon via MLX.
    static let qwen3Embedding_0_6B_4bitDWQ = EmbeddingModel(
        identifier: "qwen3-embedding-0.6b-4bit-dwq-lasttoken",
        displayName: "Qwen3 Embedding 0.6B (4-bit DWQ)",
        dimension: 1024,
        huggingFaceRepo: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        bundleSubdirectory: "Models/qwen3-embedding-0.6b-4bit-dwq",
        pooling: .lastToken
    )

    /// The default model used by `SemanticSearchEngine()` when no model is specified.
    public static let `default`: EmbeddingModel = .qwen3Embedding_0_6B_4bitDWQ
}
