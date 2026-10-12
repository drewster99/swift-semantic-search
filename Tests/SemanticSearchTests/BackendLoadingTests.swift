import Foundation
import MLXEmbedders
import Testing
@testable import SemanticSearch

@Suite("BackendLoading")
struct BackendLoadingTests {
    private static let downloadedURL = URL(fileURLWithPath: "/tmp/SemanticSearchTests/downloaded-model", isDirectory: true)

    @Test("weights already downloaded load from their directory, not through the Hub")
    func downloadedWeightsLoadFromDirectory() {
        let registered = ModelConfiguration(id: "org/model")
        let configuration = MLXEmbedderBackendLoader.configuration(
            for: .downloaded(Self.downloadedURL), registered: registered)
        guard case .directory(let url) = configuration.id else {
            Issue.record("downloaded weights would be loaded through the Hub id, which needs the network")
            return
        }
        #expect(url == Self.downloadedURL)
    }

    @Test("a registry entry with a separate tokenizer repository keeps the Hub path")
    func separateTokenizerRepositoryUsesHub() {
        let registered = ModelConfiguration(id: "org/model", tokenizerId: "org/tokenizer")
        let configuration = MLXEmbedderBackendLoader.configuration(
            for: .downloaded(Self.downloadedURL), registered: registered)
        guard case .id(let id) = configuration.id else {
            Issue.record("expected the Hub id path")
            return
        }
        #expect(id == "org/model")
    }

    @Test("absent weights download through the Hub")
    func absentWeightsUseHub() {
        let registered = ModelConfiguration(id: "org/model")
        let configuration = MLXEmbedderBackendLoader.configuration(for: .absent, registered: registered)
        guard case .id = configuration.id else {
            Issue.record("expected the Hub id path")
            return
        }
    }

    @Test("loading weights already on disk is reported as loadingWeights, not downloading")
    func onDiskLoadPhase() {
        #expect(MLXEmbedderBackendLoader.loadPhase(for: .downloaded(Self.downloadedURL)) == .loadingWeights)
        #expect(MLXEmbedderBackendLoader.loadPhase(for: .bundled(Self.downloadedURL)) == .loadingWeights)
        #expect(MLXEmbedderBackendLoader.loadPhase(for: .absent) == .downloading)
    }

    @Test("vectors of the wrong dimension are rejected")
    func wrongDimensionIsRejected() {
        #expect {
            try MLXEmbedderBackend.validate([[1, 0, 0]], expectedDimension: 4)
        } throws: { error in
            guard case SemanticSearchError.dimensionMismatch(expected: 4, received: 3) = error else { return false }
            return true
        }
    }

    @Test("vectors containing NaN or infinity are rejected")
    func nonFiniteIsRejected() {
        for bad: Float in [.nan, .infinity, -.infinity] {
            #expect {
                try MLXEmbedderBackend.validate([[1, 0], [0, bad]], expectedDimension: 2)
            } throws: { error in
                guard case SemanticSearchError.nonFiniteEmbedding = error else { return false }
                return true
            }
        }
    }

    @Test("well-formed vectors pass validation")
    func validVectorsPass() throws {
        try MLXEmbedderBackend.validate([[1, 0], [0, 1]], expectedDimension: 2)
    }
}
