import Foundation
import Accelerate

/// Stateless vector math primitives. Operates on `Float` because that's what the
/// MLX backend produces natively.
public enum VectorMath {
    /// Dot product of two equal-length vectors. O(n) via vDSP. Traps if the lengths differ.
    public static func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have equal length")
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    /// L2 magnitude (Euclidean norm) of a vector.
    public static func magnitude(_ v: [Float]) -> Float {
        var sumOfSquares: Float = 0
        vDSP_svesq(v, 1, &sumOfSquares, vDSP_Length(v.count))
        return sqrtf(sumOfSquares)
    }

    /// Returns a new vector scaled to unit magnitude. Returns the input unchanged
    /// if its magnitude is zero.
    public static func l2Normalized(_ v: [Float]) -> [Float] {
        let mag = magnitude(v)
        guard mag > 0 else { return v }
        var scale: Float = 1.0 / mag
        var result = [Float](repeating: 0, count: v.count)
        vDSP_vsmul(v, 1, &scale, &result, 1, vDSP_Length(v.count))
        return result
    }

    /// Cosine similarity of two equal-length vectors. Returns 0 if either has zero magnitude.
    /// Range: `[-1, 1]`. Traps if the lengths differ.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have equal length")
        let magA = magnitude(a)
        let magB = magnitude(b)
        guard magA > 0, magB > 0 else { return 0 }
        return dotProduct(a, b) / (magA * magB)
    }

    /// Ranks `documents` against `query` by cosine similarity. Returns pairs of
    /// `(originalIndex, score)` sorted by score descending; equal scores are ordered by
    /// ascending index, so results are deterministic.
    ///
    /// Not every document is scored. Skipped without error:
    /// - documents whose length differs from the query's (for example, vectors stored by a
    ///   different model — compare `EmbeddingModel.identifier` before calling);
    /// - documents with zero magnitude;
    /// - documents whose score is not finite (NaN or infinite components).
    ///
    /// A query with zero or non-finite magnitude returns an empty array.
    ///
    /// - Parameter query: the query vector.
    /// - Parameter documents: caller-owned document vectors. The package never owns your corpus.
    /// - Parameter top: optional cap. `nil` returns every scored document. Must not be
    ///   negative; a negative value traps.
    public static func match(
        query: [Float],
        against documents: [[Float]],
        top: Int? = nil
    ) -> [(index: Int, score: Float)] {
        if let top {
            precondition(top >= 0, "top must not be negative")
        }
        let queryMag = magnitude(query)
        guard queryMag > 0, queryMag.isFinite else { return [] }

        var scored: [(index: Int, score: Float)] = []
        scored.reserveCapacity(documents.count)
        for (i, doc) in documents.enumerated() {
            guard doc.count == query.count else { continue }
            let docMag = magnitude(doc)
            guard docMag > 0 else { continue }
            let score = dotProduct(query, doc) / (queryMag * docMag)
            guard score.isFinite else { continue }
            scored.append((i, score))
        }
        scored.sort { lhs, rhs in
            lhs.score != rhs.score ? lhs.score > rhs.score : lhs.index < rhs.index
        }

        if let top {
            return Array(scored.prefix(top))
        }
        return scored
    }
}
