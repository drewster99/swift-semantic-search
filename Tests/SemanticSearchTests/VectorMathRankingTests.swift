import Foundation
import Testing
@testable import SemanticSearch

@Suite("VectorMathRanking")
struct VectorMathRankingTests {
    @Test("a document whose score is not finite is not ranked")
    func nonFiniteScoreIsSkipped() {
        let query: [Float] = [1, 0, 0]
        let documents: [[Float]] = [
            [.infinity, 0, 0],  // magnitude is infinite, so the cosine is NaN
            [1, 0, 0]
        ]
        let results = VectorMath.match(query: query, against: documents)
        #expect(results.map(\.index) == [1])
        #expect(results.allSatisfy { $0.score.isFinite })
    }

    @Test("equal scores are ordered by ascending document index")
    func tiesBreakByIndex() {
        let query: [Float] = [1, 0, 0]
        let documents: [[Float]] = [
            [0, 1, 0],
            [2, 0, 0],
            [1, 0, 0],
            [0, 3, 0],
            [5, 0, 0]
        ]
        let results = VectorMath.match(query: query, against: documents)
        #expect(results.map(\.index) == [1, 2, 4, 0, 3])
    }

    @Test("top of zero returns no results")
    func topZero() {
        let results = VectorMath.match(query: [1, 0], against: [[1, 0]], top: 0)
        #expect(results.isEmpty)
    }
}
