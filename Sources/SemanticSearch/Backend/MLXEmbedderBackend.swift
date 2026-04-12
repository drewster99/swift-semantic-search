import Foundation
import MLX
import MLXEmbedders
import Tokenizers

/// MLX-backed embedding backend. Wraps a loaded `MLXEmbedders.ModelContainer`
/// and runs one tokenize → pad → forward → pool → extract pipeline per call.
///
/// The pipeline mirrors the canonical implementation in `mlx-swift-examples`'
/// `embedder-tool` so behavior matches the reference exactly.
internal final class MLXEmbedderBackend: EmbeddingBackend {
    let modelIdentifier: String
    let dimension: Int
    private let container: ModelContainer

    init(modelIdentifier: String, dimension: Int, container: ModelContainer) {
        self.modelIdentifier = modelIdentifier
        self.dimension = dimension
        self.container = container
    }

    func embed(batch texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        return try await container.perform { model, tokenizer, pooler in
            var encoded: [[Int]] = []
            encoded.reserveCapacity(texts.count)
            for text in texts {
                let tokens = tokenizer.encode(text: text, addSpecialTokens: true)
                if tokens.isEmpty {
                    throw SemanticSearchError.emptyTokenization(input: text)
                }
                encoded.append(tokens)
            }

            // Qwen3 is autoregressive and has no [PAD] token — fall back to EOS,
            // then to 0 as a last resort. The attention mask below excludes pad
            // positions so they don't contribute to pooled outputs.
            let padToken = tokenizer.convertTokenToId("[PAD]") ?? tokenizer.eosTokenId ?? 0

            let maxLength = encoded.map(\.count).max() ?? 0

            let padded = stacked(
                encoded.map { tokens in
                    MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
                }
            )
            let mask = (padded .!= padToken)
            let tokenTypes = MLXArray.zeros(like: padded)

            let outputs = model(
                padded,
                positionIds: nil,
                tokenTypeIds: tokenTypes,
                attentionMask: mask
            )

            let pooled = pooler(
                outputs,
                mask: mask,
                normalize: true,
                applyLayerNorm: false
            )
            pooled.eval()

            return try Self.extractVectors(from: pooled, expectedCount: encoded.count)
        }
    }

    /// Extracts `[Float]` vectors from a pooled MLX tensor. Handles both 2D
    /// `(batch, dim)` outputs (last-token / mean / cls poolers) and 3D
    /// `(batch, seq, dim)` outputs (per-token sequences), mean-pooling across the
    /// sequence axis in the 3D case.
    private static func extractVectors(from array: MLXArray, expectedCount: Int) throws -> [[Float]] {
        let shape = array.shape
        switch shape.count {
        case 2:
            let vectors = array.map { $0.asArray(Float.self) }
            guard vectors.count == expectedCount else {
                throw SemanticSearchError.unexpectedVectorCount(
                    expected: expectedCount,
                    received: vectors.count
                )
            }
            return vectors
        case 3:
            let reduced = mean(array, axis: 1)
            reduced.eval()
            let vectors = reduced.map { $0.asArray(Float.self) }
            guard vectors.count == expectedCount else {
                throw SemanticSearchError.unexpectedVectorCount(
                    expected: expectedCount,
                    received: vectors.count
                )
            }
            return vectors
        default:
            throw SemanticSearchError.unsupportedPoolingShape(shape)
        }
    }
}
