import Foundation

/// Internal abstraction over a loaded embedding model. Lets us swap MLX for another
/// runtime later (CoreML, etc.) without changing `SemanticSearchEngine`'s public API.
internal protocol EmbeddingBackend: Sendable {
    /// Stable identifier of the model that produced this backend's vectors.
    var modelIdentifier: String { get }

    /// Output vector dimension.
    var dimension: Int { get }

    /// Embed a batch of strings in a single forward pass. The caller is responsible
    /// for chunking large batches into model-friendly sizes.
    func embed(batch texts: [String]) async throws -> [[Float]]
}
