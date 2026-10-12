import Foundation
import Testing
@testable import SemanticSearch

/// Regression tests for callers that join a load already in flight.
@Suite("EngineJoin")
struct EngineJoinTests {
    @Test("two prepare() calls sharing one successful load both finish with .done")
    func joinerOfSuccessfulLoadFinishes() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let gate = LoadGate()
        let engine = TestFixtures.makeEngine(
            loader: GatedBackendLoader(gate: gate, backend: FakeEmbeddingBackend(dimension: TestFixtures.dimension)),
            applicationSupportRoot: root
        )
        let owner = Task { await drain(engine.prepare()) }
        await gate.waitUntilStarted(1)
        let joiner = Task { await drain(engine.prepare()) }
        // The joiner must be suspended on the in-flight load before it is released.
        while await engine.joinedLoadWaiterCount == 0 {
            await Task.yield()
        }

        await gate.release(loadNumber: 0, with: .succeed)

        for (name, outcome) in [("owner", await owner.value), ("joiner", await joiner.value)] {
            if case .finished(let phases) = outcome {
                #expect(phases.last == .done, "\(name) ended without .done: \(phases)")
            } else {
                Issue.record("\(name) prepare() threw: \(outcome)")
            }
        }
        if case .ready = await engine.state {
        } else {
            Issue.record("engine should be ready, is \(await engine.state)")
        }
    }

    @Test("a joiner of a load superseded by unload() throws CancellationError, not the load's error")
    func supersededJoinerThrowsCancellation() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let gate = LoadGate()
        let engine = TestFixtures.makeEngine(
            loader: GatedBackendLoader(gate: gate, backend: FakeEmbeddingBackend(dimension: TestFixtures.dimension)),
            applicationSupportRoot: root
        )
        let owner = Task { await drain(engine.prepare()) }
        await gate.waitUntilStarted(1)
        let joiner = Task { await drain(engine.prepare()) }
        while await engine.joinedLoadWaiterCount == 0 {
            await Task.yield()
        }

        await engine.unload()
        await gate.release(loadNumber: 0, with: .fail)

        for (name, outcome) in [("owner", await owner.value), ("joiner", await joiner.value)] {
            if case .threw(let error) = outcome {
                #expect(error is CancellationError, "\(name) threw \(error) instead of CancellationError")
            } else {
                Issue.record("\(name) prepare() should throw CancellationError, got \(outcome)")
            }
        }
    }
}
