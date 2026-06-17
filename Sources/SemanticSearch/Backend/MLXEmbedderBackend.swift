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

        return try await container.perform { model, tokenizer, _ in
            var encoded: [[Int]] = []
            encoded.reserveCapacity(texts.count)
            for text in texts {
                let tokens = tokenizer.encode(text: text, addSpecialTokens: true)
                if tokens.isEmpty {
                    throw SemanticSearchError.emptyTokenization(input: text)
                }
                encoded.append(tokens)
            }

            // Last-token (EOS) pooling for Qwen3, a CAUSAL model: the final real token has attended
            // to the whole sequence, so its hidden state IS the sentence embedding. (MLXEmbedders
            // mean-pools when a model ships no 1_Pooling/config.json — which Qwen3 doesn't — and that
            // compresses the cosine distribution into a narrow high band.)
            //
            // Batching is preserved: a `.none` pooler returns the full per-token hidden states as a
            // public MLXArray (Qwen3's `pooledOutput` is nil, so `.none` yields `hiddenStates`), and
            // we gather each row's last REAL token ourselves. MLXEmbedders' `.last` strategy can't be
            // used directly — it reads index -1, a PAD position under right-padding. Right-padding is
            // safe for the forward pass anyway: Qwen3 is causal (it builds its own causal mask and
            // ignores the passed attentionMask), so each EOS attends only to earlier real tokens and
            // trailing pads never affect it.
            let padToken = tokenizer.eosTokenId ?? 0
            let maxLength = encoded.map(\.count).max() ?? 0
            let padded = stacked(
                encoded.map { tokens in
                    MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
                }
            )
            let outputs = model(padded, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
            let hidden = Pooling(strategy: .none)(
                outputs, mask: nil, normalize: false, applyLayerNorm: false
            )
            let lastRows = encoded.indices.map { hidden[$0, encoded[$0].count - 1] }
            let pooled = stacked(lastRows)
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
