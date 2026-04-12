import Foundation
import Testing
@testable import SemanticSearch

@Suite("VectorMath")
struct VectorMathTests {
    @Test("dot product of orthogonal vectors is zero")
    func dotOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]
        #expect(VectorMath.dotProduct(a, b) == 0)
    }

    @Test("dot product of identical unit vectors is one")
    func dotIdentity() {
        let a: [Float] = [1, 0, 0]
        #expect(VectorMath.dotProduct(a, a) == 1)
    }

    @Test("magnitude of unit vector is one")
    func magnitudeUnit() {
        let v: [Float] = [0, 1, 0]
        #expect(VectorMath.magnitude(v) == 1)
    }

    @Test("l2Normalized scales to unit magnitude")
    func l2Normalize() {
        let v: [Float] = [3, 4, 0]
        let normalized = VectorMath.l2Normalized(v)
        let mag = VectorMath.magnitude(normalized)
        #expect(abs(mag - 1.0) < 1e-6)
    }

    @Test("cosine similarity of identical vectors is one")
    func cosineIdentical() {
        let v: [Float] = [3, 4, 0]
        let sim = VectorMath.cosineSimilarity(v, v)
        #expect(abs(sim - 1.0) < 1e-6)
    }

    @Test("cosine similarity of opposite vectors is negative one")
    func cosineOpposite() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [-1, 0, 0]
        #expect(VectorMath.cosineSimilarity(a, b) == -1)
    }

    @Test("cosine similarity ignores magnitude")
    func cosineMagnitudeInvariant() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [5, 0, 0]
        #expect(VectorMath.cosineSimilarity(a, b) == 1)
    }

    @Test("match returns documents in score order")
    func matchOrdering() {
        let query: [Float] = [1, 0, 0]
        let documents: [[Float]] = [
            [0, 1, 0],   // orthogonal — score 0
            [1, 0, 0],   // identical — score 1
            [0.5, 0.5, 0]  // 45° — score ~0.707
        ]
        let results = VectorMath.match(query: query, against: documents)
        #expect(results.count == 3)
        #expect(results[0].index == 1)
        #expect(results[1].index == 2)
        #expect(results[2].index == 0)
    }

    @Test("match with top=2 returns only the top two")
    func matchTopK() {
        let query: [Float] = [1, 0, 0]
        let documents: [[Float]] = [
            [0, 1, 0],
            [1, 0, 0],
            [0.5, 0.5, 0]
        ]
        let results = VectorMath.match(query: query, against: documents, top: 2)
        #expect(results.count == 2)
        #expect(results[0].index == 1)
        #expect(results[1].index == 2)
    }

    @Test("match skips documents with mismatched length")
    func matchSkipsMismatched() {
        let query: [Float] = [1, 0, 0]
        let documents: [[Float]] = [
            [1, 0, 0, 0],  // wrong length — skipped
            [1, 0, 0]      // matches
        ]
        let results = VectorMath.match(query: query, against: documents)
        #expect(results.count == 1)
        #expect(results[0].index == 1)
    }

    @Test("match skips zero-magnitude documents")
    func matchSkipsZero() {
        let query: [Float] = [1, 0, 0]
        let documents: [[Float]] = [
            [0, 0, 0],
            [1, 0, 0]
        ]
        let results = VectorMath.match(query: query, against: documents)
        #expect(results.count == 1)
        #expect(results[0].index == 1)
    }

    @Test("match with zero-magnitude query returns empty")
    func matchZeroQuery() {
        let query: [Float] = [0, 0, 0]
        let documents: [[Float]] = [[1, 0, 0]]
        let results = VectorMath.match(query: query, against: documents)
        #expect(results.isEmpty)
    }
}
