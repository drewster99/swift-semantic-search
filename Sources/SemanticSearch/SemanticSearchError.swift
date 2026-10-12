import Foundation

/// Errors thrown by `SemanticSearchEngine` and supporting types.
///
/// Switch over this with a `default` clause: new cases may be added in minor releases.
public enum SemanticSearchError: LocalizedError, Sendable {
    /// `embed(...)` or `match(...)` was called before `prepare()` completed.
    case notReady

    /// `deleteDownloadedWeights()` was called while the model is loaded or loading, or while a
    /// load cancelled by `unload()` is still winding down. Call `unload()` first; retry later
    /// in the winding-down case.
    case cannotDeleteWhileLoaded

    /// The MLX backend produced a vector count that didn't match the input batch.
    case unexpectedVectorCount(expected: Int, received: Int)

    /// Tokenization produced no tokens for one of the input strings.
    case emptyTokenization(input: String)

    /// The MLX backend returned a tensor with a shape we don't know how to extract.
    case unsupportedPoolingShape([Int])

    /// The backend produced vectors whose length differs from the model's declared
    /// `EmbeddingModel.dimension` — usually a wrong `dimension` on a custom model.
    case dimensionMismatch(expected: Int, received: Int)

    /// The backend produced a vector containing NaN or infinity.
    case nonFiniteEmbedding

    /// A generic embedding failure with a message describing what went wrong.
    case embeddingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notReady:
            return "SemanticSearchEngine is not ready. Call prepare() first."
        case .cannotDeleteWhileLoaded:
            return "Cannot delete model weights while the model is loaded. Call unload() first."
        case .unexpectedVectorCount(let expected, let received):
            return "MLX backend produced \(received) vectors but expected \(expected)."
        case .emptyTokenization(let input):
            let preview = input.prefix(60)
            return "Tokenization produced no tokens for input: \"\(preview)\""
        case .unsupportedPoolingShape(let shape):
            return "Pooling produced an unsupported tensor shape: \(shape)"
        case .dimensionMismatch(let expected, let received):
            return "Embedding has \(received) dimensions but the model declares \(expected)."
        case .nonFiniteEmbedding:
            return "Embedding contains NaN or infinite values."
        case .embeddingFailed(let message):
            return "Embedding failed: \(message)"
        }
    }
}
