import Foundation
import MLX
import MLXEmbedders
import MLXLinalg
import Tokenizers
import os

/// MLX-backed embedding backend. Wraps a loaded `MLXEmbedders.ModelContainer`
/// and runs one tokenize → pad → forward → pool → extract pipeline per call.
///
/// Pooling follows the model's declared `PoolingStrategy` (Qwen3 is `.lastToken`); a strategy the
/// backend hasn't implemented throws rather than silently producing a wrong vector.
internal final class MLXEmbedderBackend: EmbeddingBackend {
    let modelIdentifier: String
    let dimension: Int
    private let container: ModelContainer
    private let pooling: PoolingStrategy

    /// MLX footprint (active + cache) at the last notice-level log, so growth is reported in
    /// steps instead of once per batch. A lock rather than a plain var because `embed` is
    /// nonisolated — serialization through the owning actor is a caller convention this file
    /// shouldn't bet its memory safety on.
    private let lastNoticedMLXFootprintBytes = OSAllocatedUnfairLock(initialState: 0)
    /// Log at notice level every time MLX's footprint has grown this much since the last notice.
    /// Debug-level lines fire on every batch regardless.
    private static let footprintNoticeStepBytes = 512 * 1024 * 1024

    init(modelIdentifier: String, dimension: Int, container: ModelContainer, pooling: PoolingStrategy) {
        self.modelIdentifier = modelIdentifier
        self.dimension = dimension
        self.container = container
        self.pooling = pooling
    }

    func embed(batch texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        defer { logMemoryAfterBatch(batchSize: texts.count) }
        let pooling = self.pooling
        let expectedDimension = self.dimension

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

            // Right-pad to the longest sequence. The pad value is arbitrary: the only model we
            // support (Qwen3) is causal and builds its own causal mask, ignoring `attentionMask`.
            let padToken = tokenizer.eosTokenId ?? 0
            let maxLength = encoded.map(\.count).max() ?? 0
            let padded = stacked(
                encoded.map { tokens in
                    MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
                }
            )
            let outputs = model(padded, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)

            let pooled: MLXArray
            switch pooling {
            case .lastToken:
                // Causal model (Qwen3): the final real token has attended to the whole sequence, so
                // its hidden state IS the sentence embedding. A `.none` pooler returns the full
                // per-token hidden states as a public MLXArray (Qwen3's `pooledOutput` is nil), and we
                // gather each row's last REAL token. MLXEmbedders' `.last` can't be used directly: it
                // reads index -1, a PAD position under right-padding. Right-padding is safe because a
                // causal model never attends forward, so trailing pads can't affect the EOS state.
                let hidden = Pooling(strategy: .none)(
                    outputs, mask: nil, normalize: false, applyLayerNorm: false
                )
                let lastRows = encoded.indices.map { hidden[$0, encoded[$0].count - 1] }
                pooled = stacked(lastRows)
            case .mean, .cls:
                // Declared by the model but not implemented here. Fail loudly instead of guessing —
                // a bidirectional / CLS model needs a real (attention-masked) implementation + tests.
                throw SemanticSearchError.embeddingFailed(
                    "Pooling strategy '\(pooling.rawValue)' is not implemented in MLXEmbedderBackend; "
                    + "implement it before using a model that declares it")
            }
            pooled.eval()

            let vectors = try Self.extractVectors(from: pooled, expectedCount: encoded.count)
            try Self.validate(vectors, expectedDimension: expectedDimension)
            return vectors
        }
    }

    /// Traces GPU memory after a forward pass: a debug line per batch, and a notice line each
    /// time MLX's footprint has grown `footprintNoticeStepBytes` past the last notice — so a
    /// runaway cache shows up as a ladder of notices in the unified log (subsystem
    /// `com.nuclearcyborg.SemanticSearch`, category `GPUMemory`) rather than one line per batch
    /// drowning everything else out.
    private func logMemoryAfterBatch(batchSize: Int) {
        let report = GPUMemoryReport.current()
        gpuMemoryLogger.debug(
            "after batch of \(batchSize, privacy: .public): \(report.description, privacy: .public)")
        let shouldNotice = lastNoticedMLXFootprintBytes.withLock { lastNoticed in
            guard report.mlxFootprintBytes - lastNoticed >= Self.footprintNoticeStepBytes else {
                return false
            }
            lastNoticed = report.mlxFootprintBytes
            return true
        }
        if shouldNotice {
            gpuMemoryLogger.notice("MLX footprint grew: \(report.description, privacy: .public)")
        }
    }

    /// Checks the API contract on every produced vector: length equals the model's declared
    /// dimension and every component is finite. A wrong `dimension` on a custom model, or a
    /// numerical blow-up in the forward pass, would otherwise reach callers as vectors that
    /// `VectorMath.match` silently skips.
    static func validate(_ vectors: [[Float]], expectedDimension: Int) throws {
        for vector in vectors {
            guard vector.count == expectedDimension else {
                throw SemanticSearchError.dimensionMismatch(expected: expectedDimension, received: vector.count)
            }
            guard vector.allSatisfy(\.isFinite) else {
                throw SemanticSearchError.nonFiniteEmbedding
            }
        }
    }

    /// Extracts `[Float]` vectors from a pooled MLX tensor and L2-normalizes them
    /// at the sentence level so the API contract (unit-length sentence embeddings)
    /// holds regardless of the pooling strategy. `.lastToken` pooling produces 2D
    /// `(batch, dim)` output; the 3D `(batch, seq, dim)` branch mean-pools across the
    /// sequence axis and exists for a future pooling path that returns per-token states.
    private static func extractVectors(from array: MLXArray, expectedCount: Int) throws -> [[Float]] {
        // Widen to Float32 BEFORE pooling and normalizing. The model computes in bfloat16, which
        // keeps Float32's exponent range but has only 7 mantissa bits — roughly 0.4% relative
        // precision. Dividing by the norm in that dtype rounds every component of the quotient, so
        // the vectors come out with lengths scattered across ~±0.4% instead of 1.0. That breaks the
        // unit-length contract this function exists to uphold, and it breaks it SYSTEMATICALLY: a
        // caller using a dot product as a stand-in for cosine (the standard reason to normalize)
        // gives a permanent ~0.4% boost to whichever vectors happen to land above 1.0, on every
        // query, forever. Widening first costs one extra kernel over a (batch x dimension) array.
        let widened = array.asType(.float32)
        let shape = widened.shape
        let pooled: MLXArray
        switch shape.count {
        case 2:
            pooled = widened
        case 3:
            pooled = mean(widened, axis: 1)
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
