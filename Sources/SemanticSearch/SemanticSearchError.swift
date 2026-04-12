import Foundation

/// Errors thrown by `SemanticSearchEngine` and supporting types.
public enum SemanticSearchError: LocalizedError {
    /// `embed(...)` or `prepare()` was called before the model finished loading.
    case notReady

    /// `deleteDownloadedWeights()` was called while the model is still loaded
    /// in memory. Call `unload()` first.
    case cannotDeleteWhileLoaded

    /// The MLX backend produced a vector count that didn't match the input batch.
    case unexpectedVectorCount(expected: Int, received: Int)

    /// Tokenization produced no tokens for one of the input strings.
    case emptyTokenization(input: String)

    /// The MLX backend returned a tensor with a shape we don't know how to extract.
    case unsupportedPoolingShape([Int])

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
        case .embeddingFailed(let message):
            return "Embedding failed: \(message)"
        }
    }
}
