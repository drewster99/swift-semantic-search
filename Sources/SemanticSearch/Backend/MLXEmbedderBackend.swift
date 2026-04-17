import Foundation
import MLX
import MLXEmbedders
import MLXLinalg
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

    /// Extracts `[Float]` vectors from a pooled MLX tensor and L2-normalizes them
    /// at the sentence level so the API contract (unit-length sentence embeddings)
    /// holds regardless of the pooling strategy. Handles both 2D `(batch, dim)`
    /// outputs and 3D `(batch, seq, dim)` outputs — in the 3D case we mean-pool
    /// across the sequence axis first.
    ///
    /// The separate L2 pass matters because the MLX pooler's `normalize: true`
    /// flag is applied *before* our Swift-side mean pooling in the 3D case: each
    /// token is unit-normalized, but averaging them does not preserve unit norm.
    /// Models that ship without a `1_Pooling/config.json` (such as the default
    /// Qwen3 embedding repo) fall through to that 3D path.
    private static func extractVectors(from array: MLXArray, expectedCount: Int) throws -> [[Float]] {
        let shape = array.shape
        let pooled: MLXArray
        switch shape.count {
        case 2:
            pooled = array
        case 3:
            pooled = mean(array, axis: 1)
        default:
            throw SemanticSearchError.unsupportedPoolingShape(shape)
        }

        let normalized = pooled / MLX.maximum(norm(pooled, axis: -1, keepDims: true), MLXArray(1e-12))
        normalized.eval()

        let vectors = normalized.map { $0.asArray(Float.self) }
        guard vectors.count == expectedCount else {
            throw SemanticSearchError.unexpectedVectorCount(
                expected: expectedCount,
                received: vectors.count
            )
        }
        return vectors
    }
}
