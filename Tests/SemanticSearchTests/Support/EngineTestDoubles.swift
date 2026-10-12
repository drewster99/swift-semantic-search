import Foundation
import os
@testable import SemanticSearch

/// Error a test load fails with.
struct TestLoadError: Error, Equatable {}

/// Outcome a test releases a gated load with.
enum GatedLoadOutcome: Sendable {
    case succeed
    case fail
}

/// Backend that returns deterministic vectors without touching MLX, and counts its calls.
final class FakeEmbeddingBackend: EmbeddingBackend {
    let modelIdentifier = "fake-model"
    let dimension: Int
    /// When true, the first `embed` call cancels the task that is running it, simulating a
    /// caller cancelling a bulk embed while a chunk is on the GPU.
    private let cancelsCallerOnFirstCall: Bool
    private let callCount = OSAllocatedUnfairLock(initialState: 0)

    init(dimension: Int, cancelsCallerOnFirstCall: Bool = false) {
        self.dimension = dimension
        self.cancelsCallerOnFirstCall = cancelsCallerOnFirstCall
    }

    var embedCallCount: Int { callCount.withLock { $0 } }

    func embed(batch texts: [String]) async throws -> [[Float]] {
        let callNumber = callCount.withLock { count in
            count += 1
            return count
        }
        if cancelsCallerOnFirstCall, callNumber == 1 {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        return texts.map { _ in [Float](repeating: 0, count: dimension - 1) + [1] }
    }
}

/// Loader that returns a fixed backend immediately.
struct ImmediateBackendLoader: EmbeddingBackendLoader {
    let backend: FakeEmbeddingBackend

    func loadBackend(
        for model: EmbeddingModel,
        locator: ModelLocator,
        progress: AsyncThrowingStream<PrepareProgress, Error>.Continuation
    ) async throws -> any EmbeddingBackend {
        backend
    }
}

/// Lets a test hold each load at its start and decide when, and how, it finishes. Loads ignore
/// task cancellation, like a real download or weight load that is past its cancellation points.
actor LoadGate {
    private var startedCount = 0
    private var startWaiters: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []
    private var heldLoads: [Int: CheckedContinuation<GatedLoadOutcome, Never>] = [:]
    private var earlyReleases: [Int: GatedLoadOutcome] = [:]

    /// Called by a load: registers it and suspends until `release(loadNumber:with:)`.
    func holdLoad() async -> GatedLoadOutcome {
        let loadNumber = startedCount
        startedCount += 1
        let ready = startWaiters.filter { $0.count <= startedCount }
        startWaiters.removeAll { $0.count <= startedCount }
        for waiter in ready {
            waiter.continuation.resume()
        }
        if let outcome = earlyReleases.removeValue(forKey: loadNumber) {
            return outcome
        }
        return await withCheckedContinuation { heldLoads[loadNumber] = $0 }
    }

    /// Suspends until at least `count` loads have started.
    func waitUntilStarted(_ count: Int) async {
        if startedCount >= count { return }
        await withCheckedContinuation { startWaiters.append((count, $0)) }
    }

    /// Finishes load number `loadNumber` (0-based, in start order).
    func release(loadNumber: Int, with outcome: GatedLoadOutcome) {
        if let held = heldLoads.removeValue(forKey: loadNumber) {
            held.resume(returning: outcome)
        } else {
            earlyReleases[loadNumber] = outcome
        }
    }
}

struct GatedBackendLoader: EmbeddingBackendLoader {
    let gate: LoadGate
    let backend: FakeEmbeddingBackend

    func loadBackend(
        for model: EmbeddingModel,
        locator: ModelLocator,
        progress: AsyncThrowingStream<PrepareProgress, Error>.Continuation
    ) async throws -> any EmbeddingBackend {
        switch await gate.holdLoad() {
        case .succeed: return backend
        case .fail: throw TestLoadError()
        }
    }
}

/// How a `prepare()` stream ended.
enum PrepareStreamOutcome: Sendable {
    case finished(phases: [PrepareProgress.Phase])
    case threw(any Error)
}

/// Drains a `prepare()` stream, recording phases and how it ended.
func drain(_ stream: AsyncThrowingStream<PrepareProgress, Error>) async -> PrepareStreamOutcome {
    var phases: [PrepareProgress.Phase] = []
    do {
        for try await event in stream {
            phases.append(event.phase)
        }
        return .finished(phases: phases)
    } catch {
        return .threw(error)
    }
}

enum TestFixtures {
    static let dimension = 4

    /// A model that is neither bundled nor downloaded.
    static let model = EmbeddingModel(
        identifier: "test-model",
        displayName: "Test Model",
        dimension: dimension,
        huggingFaceRepo: "semantic-search-tests/nonexistent-model",
        bundleSubdirectory: "SemanticSearchTests/nonexistent-model",
        pooling: .lastToken
    )

    /// A fresh, empty directory under the temporary directory.
    static func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("SemanticSearchTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func makeEngine(
        loader: any EmbeddingBackendLoader,
        applicationSupportRoot: URL,
        maxBatchSize: Int = 16
    ) -> SemanticSearchEngine {
        SemanticSearchEngine(
            model: model,
            options: EngineOptions(maxBatchSize: maxBatchSize),
            locator: ModelLocator(model: model, applicationSupportRoot: applicationSupportRoot),
            backendLoader: loader
        )
    }
}
