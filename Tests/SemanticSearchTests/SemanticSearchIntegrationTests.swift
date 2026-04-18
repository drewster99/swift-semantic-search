import Foundation
import Testing
@testable import SemanticSearch

/// Curated corpus used by the integration suite. Shipped as a bundle resource so
/// the test can be read from disk once and reused across every test in the suite.
private struct Corpus: Decodable, Sendable {
    struct Document: Decodable, Sendable {
        let id: String
        let topic: String
        let text: String
    }
    struct Query: Decodable, Sendable {
        let text: String
        let expectedId: String
    }
    let documents: [Document]
    let queries: [Query]

    static func load() throws -> Corpus {
        guard let url = Bundle.module.url(forResource: "corpus", withExtension: "json") else {
            throw SemanticSearchError.embeddingFailed("Missing corpus.json fixture in test bundle")
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Corpus.self, from: data)
    }
}

/// End-to-end tests that exercise the real MLX-backed embedding pipeline. The first
/// run downloads the Qwen3 embedding weights (~500MB) into the user's cache; later
/// runs hit the cache and finish in a few seconds on Apple Silicon.
///
/// Tagged so the suite can be included or skipped with a filter if needed:
///   swift test --filter Integration
@Suite("Integration", .serialized)
struct SemanticSearchIntegrationTests {
    /// A prepared engine plus the embedded corpus, shared across the suite's tests.
    /// Stored in a `Task.once`-style computed property so `prepare()` only runs once
    /// per `swift test` invocation.
    private static let shared: Task<PreparedCorpus, Error> = Task {
        let corpus = try Corpus.load()
        let engine = SemanticSearchEngine()

        for try await _ in engine.prepare() { /* drain progress */ }

        let docVectors = try await engine.embed(batch: corpus.documents.map(\.text))
        return PreparedCorpus(engine: engine, corpus: corpus, documentVectors: docVectors)
    }

    private struct PreparedCorpus: Sendable {
        let engine: SemanticSearchEngine
        let corpus: Corpus
        let documentVectors: [[Float]]
    }

    // MARK: - Tests

    @Test("model prepares, loads, and becomes ready")
    func enginePrepares() async throws {
        let prepared = try await Self.shared.value
        let state = await prepared.engine.state
        if case .ready = state { return }
        Issue.record("Engine state should be .ready after prepare(), got \(state)")
    }

    @Test("embedding a single string returns a unit-length vector of the model dimension")
    func singleEmbeddingShape() async throws {
        let prepared = try await Self.shared.value
        let vector = try await prepared.engine.embed("a short sentence about semantic search")
        #expect(vector.count == prepared.engine.model.dimension)
        let magnitude = VectorMath.magnitude(vector)
        #expect(abs(magnitude - 1.0) < 1e-3, "expected L2-normalized vector, magnitude was \(magnitude)")
    }

    @Test("batch and single embedding agree for the same input")
    func batchMatchesSingle() async throws {
        let prepared = try await Self.shared.value
        let text = "the quick brown fox jumps over the lazy dog"
        let single = try await prepared.engine.embed(text)
        let batch = try await prepared.engine.embed(batch: [text])
        try #require(batch.count == 1)
        let cosine = VectorMath.cosineSimilarity(single, batch[0])
        #expect(cosine > 0.999, "single vs batch embedding of the same text should be nearly identical, got cosine=\(cosine)")
    }

    @Test("top-1 match for each query is its expected document")
    func top1MatchesExpectedDocument() async throws {
        let prepared = try await Self.shared.value
        var misses: [(query: String, expected: String, got: String, score: Float)] = []

        for query in prepared.corpus.queries {
            let queryVector = try await prepared.engine.embed(query.text)
            let results = VectorMath.match(query: queryVector, against: prepared.documentVectors, top: 1)
            try #require(!results.isEmpty, "query \"\(query.text)\" returned no results")
            let hit = results[0]
            let hitID = prepared.corpus.documents[hit.index].id
            if hitID != query.expectedId {
                misses.append((query.text, query.expectedId, hitID, hit.score))
            }
        }

        if !misses.isEmpty {
            let description = misses
                .map { "  \"\($0.query)\" → got \($0.got) (score \($0.score)), expected \($0.expected)" }
                .joined(separator: "\n")
            Issue.record("Top-1 mismatches:\n\(description)")
        }
    }

    @Test("queries rank their expected document above an obvious distractor from a different topic")
    func expectedBeatsCrossTopicDistractor() async throws {
        let prepared = try await Self.shared.value
        let docs = prepared.corpus.documents

        for query in prepared.corpus.queries {
            guard let expectedIndex = docs.firstIndex(where: { $0.id == query.expectedId }) else {
                Issue.record("Fixture references unknown id \(query.expectedId)")
                continue
            }
            let expectedTopic = docs[expectedIndex].topic
            guard let distractorIndex = docs.firstIndex(where: { $0.topic != expectedTopic }) else {
                continue
            }

            let queryVector = try await prepared.engine.embed(query.text)
            let expectedScore = VectorMath.cosineSimilarity(queryVector, prepared.documentVectors[expectedIndex])
            let distractorScore = VectorMath.cosineSimilarity(queryVector, prepared.documentVectors[distractorIndex])
            #expect(
                expectedScore > distractorScore,
                "query \"\(query.text)\" scored \(expectedScore) against expected \(query.expectedId) but \(distractorScore) against cross-topic \(docs[distractorIndex].id)"
            )
        }
    }

    @Test("embedding the same corpus twice is deterministic")
    func embeddingsAreStable() async throws {
        let prepared = try await Self.shared.value
        let redo = try await prepared.engine.embed(batch: prepared.corpus.documents.map(\.text))
        try #require(redo.count == prepared.documentVectors.count)
        for (index, (a, b)) in zip(prepared.documentVectors, redo).enumerated() {
            let cosine = VectorMath.cosineSimilarity(a, b)
            #expect(cosine > 0.999, "document \(prepared.corpus.documents[index].id) drifted between runs, cosine=\(cosine)")
        }
    }
}
