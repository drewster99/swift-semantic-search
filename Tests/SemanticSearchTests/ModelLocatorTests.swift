import Foundation
import Testing
@testable import SemanticSearch

@Suite("ModelLocator")
struct ModelLocatorTests {
    private static func write(_ names: [String], into directory: URL) throws {
        for name in names {
            try Data("{}".utf8).write(to: directory.appendingPathComponent(name))
        }
    }

    @Test("a directory holding only config.json is not a complete model")
    func partialDownloadIsNotComplete() throws {
        let directory = try TestFixtures.makeTemporaryDirectory()
        try Self.write(["config.json"], into: directory)
        #expect(ModelLocator.validatedModelDirectory(directory) == nil)
    }

    @Test("a directory missing the weights file is not a complete model")
    func missingWeightsIsNotComplete() throws {
        let directory = try TestFixtures.makeTemporaryDirectory()
        try Self.write(["config.json", "tokenizer.json", "tokenizer_config.json"], into: directory)
        #expect(ModelLocator.validatedModelDirectory(directory) == nil)
    }

    @Test("a directory with config, tokenizer and weights is a complete model")
    func completeModelIsFound() throws {
        let directory = try TestFixtures.makeTemporaryDirectory()
        try Self.write(["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"], into: directory)
        #expect(ModelLocator.validatedModelDirectory(directory) == directory)
    }

    @Test("locating a model creates no directories")
    func locateHasNoSideEffects() throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let locator = ModelLocator(model: TestFixtures.model, applicationSupportRoot: root)
        guard case .absent = locator.locate() else {
            Issue.record("expected .absent for a model that is nowhere on disk")
            return
        }
        let created = try FileManager.default.contentsOfDirectory(atPath: root.path)
        #expect(created.isEmpty, "locate() created \(created)")
    }

    @Test("the download cache is excluded from backups")
    func cacheIsExcludedFromBackup() throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let locator = ModelLocator(model: TestFixtures.model, applicationSupportRoot: root)
        let cacheBase = try locator.prepareCacheBaseURL()
        let values = try cacheBase.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(values.isExcludedFromBackup == true)
    }
}
