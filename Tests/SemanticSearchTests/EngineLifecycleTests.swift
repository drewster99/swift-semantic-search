import Foundation
import Testing
@testable import SemanticSearch

/// Lifecycle regression tests. They inject a fake backend loader, so they need neither the
/// network nor Metal.
@Suite("EngineLifecycle")
struct EngineLifecycleTests {
    private static func isCancellation(_ outcome: PrepareStreamOutcome) -> Bool {
        if case .threw(let error) = outcome { return error is CancellationError }
        return false
    }

    private static func isReady(_ state: ModelState) -> Bool {
        if case .ready = state { return true }
        return false
    }

    @Test("a load that finishes after unload() does not mark the engine ready")
    func unloadDuringPrepareDiscardsLateResult() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let gate = LoadGate()
        let engine = TestFixtures.makeEngine(
            loader: GatedBackendLoader(gate: gate, backend: FakeEmbeddingBackend(dimension: TestFixtures.dimension)),
            applicationSupportRoot: root
        )
        let consumer = Task { await drain(engine.prepare()) }
        await gate.waitUntilStarted(1)

        await engine.unload()
        await gate.release(loadNumber: 0, with: .succeed)
        let outcome = await consumer.value

        #expect(Self.isCancellation(outcome), "prepare() interrupted by unload() should throw CancellationError, got \(outcome)")
        let state = await engine.state
        #expect(!Self.isReady(state), "unload() was undone by a late load result; state is \(state)")
    }

    @Test("a superseded load that fails does not overwrite a newer in-flight load")
    func supersededFailureDoesNotClobberNewerLoad() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let gate = LoadGate()
        let engine = TestFixtures.makeEngine(
            loader: GatedBackendLoader(gate: gate, backend: FakeEmbeddingBackend(dimension: TestFixtures.dimension)),
            applicationSupportRoot: root
        )
        let firstConsumer = Task { await drain(engine.prepare()) }
        await gate.waitUntilStarted(1)
        await engine.unload()

        let secondConsumer = Task { await drain(engine.prepare()) }
        await gate.waitUntilStarted(2)

        await gate.release(loadNumber: 0, with: .fail)
        _ = await firstConsumer.value
        let stateWhileSecondLoads = await engine.state
        if case .loading = stateWhileSecondLoads {
        } else {
            Issue.record("superseded load overwrote the in-flight load's state with \(stateWhileSecondLoads)")
        }

        await gate.release(loadNumber: 1, with: .succeed)
        let secondOutcome = await secondConsumer.value
        if case .finished(let phases) = secondOutcome {
            #expect(phases.last == .done)
        } else {
            Issue.record("second prepare() should finish with .done, got \(secondOutcome)")
        }
        #expect(Self.isReady(await engine.state))
    }

    @Test("deleting weights is refused while a cancelled load is still running")
    func deleteRefusedWhileCancelledLoadWindsDown() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let gate = LoadGate()
        let engine = TestFixtures.makeEngine(
            loader: GatedBackendLoader(gate: gate, backend: FakeEmbeddingBackend(dimension: TestFixtures.dimension)),
            applicationSupportRoot: root
        )
        let consumer = Task { await drain(engine.prepare()) }
        await gate.waitUntilStarted(1)
        await engine.unload()

        await #expect {
            try await engine.deleteDownloadedWeights()
        } throws: { error in
            guard case SemanticSearchError.cannotDeleteWhileLoaded = error else { return false }
            return true
        }

        await gate.release(loadNumber: 0, with: .succeed)
        _ = await consumer.value
        try await engine.deleteDownloadedWeights()
    }

    @Test("prepare() completes even when the caller does not retain the engine")
    func prepareKeepsEngineAlive() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let loader = ImmediateBackendLoader(backend: FakeEmbeddingBackend(dimension: TestFixtures.dimension))
        let stream = TestFixtures.makeEngine(loader: loader, applicationSupportRoot: root).prepare()
        let outcome = await drain(stream)
        guard case .finished(let phases) = outcome else {
            Issue.record("prepare() threw: \(outcome)")
            return
        }
        #expect(phases.last == .done, "stream ended without .done; phases: \(phases)")
    }

    @Test("cancelling a chunked embed(batch:) stops before the remaining chunks")
    func embedBatchHonorsCancellationBetweenChunks() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let backend = FakeEmbeddingBackend(dimension: TestFixtures.dimension, cancelsCallerOnFirstCall: true)
        let engine = TestFixtures.makeEngine(
            loader: ImmediateBackendLoader(backend: backend),
            applicationSupportRoot: root,
            maxBatchSize: 1
        )
        _ = await drain(engine.prepare())

        let texts = Array(repeating: "text", count: 8)
        let embedding = Task { try await engine.embed(batch: texts) }
        let result = await embedding.result

        switch result {
        case .success(let vectors):
            Issue.record("cancelled embed(batch:) ran to completion and returned \(vectors.count) vectors")
        case .failure(let error):
            #expect(error is CancellationError, "expected CancellationError, got \(error)")
        }
        #expect(backend.embedCallCount == 1, "backend ran \(backend.embedCallCount) chunks after cancellation")
    }

    @Test("embedding before prepare() throws the public notReady error")
    func embedBeforePrepareThrowsNotReady() async throws {
        let root = try TestFixtures.makeTemporaryDirectory()
        let engine = TestFixtures.makeEngine(
            loader: ImmediateBackendLoader(backend: FakeEmbeddingBackend(dimension: TestFixtures.dimension)),
            applicationSupportRoot: root
        )
        await #expect {
            try await engine.embed("text")
        } throws: { error in
            guard case SemanticSearchError.notReady = error else { return false }
            return true
        }
    }
}
