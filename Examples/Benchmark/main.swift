import Foundation
import MLX
import SemanticSearch

// Latency characterization for the MLX embedding path. See
// Tests/SemanticSearchTests/EmbeddingLatencyBenchmarkTests.swift for the full rationale; this is
// the same measurement as an executable, because SwiftPM's test runner does not resolve the MLX
// metallib resource bundle while a plain executable (which sits beside `mlx-swift_Cmlx.bundle` in
// the products directory) does.
//
// Reads real text from a JSON file whose path comes from SEMANTIC_SEARCH_BENCH_CORPUS:
//
//     {"samples": [{"label": "...", "kind": "...", "words": 42, "text": "..."}]}
//
// The corpus is deliberately not committed — this is a public repository and realistic inputs are
// a host application's private content.
//
//     SEMANTIC_SEARCH_BENCH_CORPUS=/path/to/corpus.json swift run SemanticSearchBench

struct Sample: Decodable, Sendable {
    let label: String
    let kind: String
    let words: Int
    let text: String
}

struct BenchmarkCorpus: Decodable, Sendable {
    let samples: [Sample]
}

let repetitions = 8

func milliseconds(_ body: () async throws -> Void) async rethrows -> Double {
    let elapsed = try await ContinuousClock().measure { try await body() }
    let components = elapsed.components
    return Double(components.seconds) * 1000 + Double(components.attoseconds) / 1e15
}

func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2)
        ? (sorted[middle - 1] + sorted[middle]) / 2
        : sorted[middle]
}

func pad(_ value: Double, _ width: Int = 8) -> String {
    String(format: "%\(width).1f", value)
}

func label(_ text: String, _ width: Int = 18) -> String {
    text.count >= width ? String(text.prefix(width)) : text.padding(toLength: width, withPad: " ", startingAt: 0)
}

guard let corpusPath = ProcessInfo.processInfo.environment["SEMANTIC_SEARCH_BENCH_CORPUS"] else {
    FileHandle.standardError.write(Data("SEMANTIC_SEARCH_BENCH_CORPUS is not set\n".utf8))
    exit(1)
}
let corpus = try JSONDecoder().decode(
    BenchmarkCorpus.self,
    from: Data(contentsOf: URL(fileURLWithPath: corpusPath))
)

/// Which experiments to run, e.g. `swift run SemanticSearchBench 1 3`. Defaults to all but the
/// idle probe, which costs two minutes of wall clock on its own.
let selected: Set<Int> = {
    let requested = CommandLine.arguments.dropFirst().compactMap(Int.init)
    return requested.isEmpty ? [1, 2, 3] : Set(requested)
}()

setvbuf(stdout, nil, _IOLBF, 0)

let engine = SemanticSearchEngine()
let prepareStart = ContinuousClock().now
for try await _ in engine.prepare() { /* drain progress */ }
print("prepare() completed in \(ContinuousClock().now - prepareStart)")

// MARK: - Experiment 1: latency vs size, repeated back-to-back

if selected.contains(1) {
print("")
print("=== embed(single string) — \(repetitions) back-to-back reps per sample ===")
print("  sample              words   chars |    rep1 |     min   median      max  (reps 2+) | rep1/steady")
print("  ------------------ ------  ------ | ------- | -------  -------  ------- | -----------")

for sample in corpus.samples {
    var timings: [Double] = []
    for _ in 0 ..< repetitions {
        timings.append(try await milliseconds { _ = try await engine.embed(sample.text) })
    }
    let first = timings[0]
    let rest = Array(timings.dropFirst())
    let steady = median(rest)
    print(
        "  \(label(sample.label)) \(String(format: "%6d", sample.words))  \(String(format: "%6d", sample.text.count)) "
        + "| \(pad(first, 7)) "
        + "| \(pad(rest.min() ?? 0, 7))  \(pad(steady, 7))  \(pad(rest.max() ?? 0, 7)) "
        + "| \(String(format: "%.2fx", steady > 0 ? first / steady : 0))"
    )
}
}

// MARK: - Experiment 2: the two-pool search shape

let memoryInstruction = "Return related memories"
let taskInstruction = "Given a software engineering task, retrieve earlier tasks that are "
    + "related, similar, or could inform how to carry it out."

if selected.contains(2) {
print("")
print("=== batch=1 vs batch=2 (two-pool search shape), median of \(repetitions) reps ===")
print("  sample              words | batch=1  batch=2 |   delta   ratio")
print("  ------------------ ------ | -------  ------- | -------  -----")

for sample in corpus.samples {
    let pair = ["\(memoryInstruction)\n\(sample.text)", "\(taskInstruction)\n\(sample.text)"]
    var single: [Double] = []
    var double: [Double] = []
    for _ in 0 ..< repetitions {
        single.append(try await milliseconds { _ = try await engine.embed(sample.text) })
        double.append(try await milliseconds { _ = try await engine.embed(batch: pair) })
    }
    let one = median(single)
    let two = median(double)
    print(
        "  \(label(sample.label)) \(String(format: "%6d", sample.words)) "
        + "| \(pad(one, 7))  \(pad(two, 7)) "
        + "| \(pad(two - one, 7))  \(String(format: "%.2fx", two / max(one, 0.001)))"
    )
}
}

// MARK: - Experiment 3: batching the corpus (the reembed path)

if selected.contains(3), let biggest = corpus.samples.max(by: { $0.words < $1.words }) {
    let corpusTexts = corpus.samples.map(\.text)
    var sequential = 0.0
    for text in corpusTexts {
        sequential += try await milliseconds { _ = try await engine.embed(text) }
    }
    let batched = try await milliseconds { _ = try await engine.embed(batch: corpusTexts) }
    print("")
    print("=== \(corpusTexts.count) documents: one call each vs one batched call ===")
    print("  sequential (\(corpusTexts.count) calls) : \(pad(sequential)) ms")
    print("  batched (1 call)      : \(pad(batched)) ms")
    print("  speedup               : \(String(format: "%.2fx", sequential / max(batched, 0.001)))")
    print("  (largest sample: \(biggest.words) words)")
}

// MARK: - Experiment 5: does length-sorting the batch recover the padding loss?

// `embed(batch:)` chunks at `maxBatchSize` IN INPUT ORDER, and the backend right-pads each chunk
// to its longest member. On a heterogeneous corpus that means a single long document inflates
// every short one it happens to share a chunk with. Sorting by length first makes each chunk
// homogeneous, so padding is bounded by the spread WITHIN a chunk rather than across the corpus.
if selected.contains(5) {
    let texts = corpus.samples.map(\.text)
    let sorted = texts.sorted { $0.count < $1.count }

    var sequential = 0.0
    for text in texts {
        sequential += try await milliseconds { _ = try await engine.embed(text) }
    }
    let unsorted = try await milliseconds { _ = try await engine.embed(batch: texts) }
    let sortedBatch = try await milliseconds { _ = try await engine.embed(batch: sorted) }

    // Padded work each strategy actually pushes through the model, in characters, as a proxy for
    // token positions. Sequential pays exactly the real length; a batch pays chunkSize x longest.
    func paddedChars(_ list: [String], chunk: Int) -> Int {
        stride(from: 0, to: list.count, by: chunk).reduce(0) { total, start in
            let slice = list[start ..< min(start + chunk, list.count)]
            return total + slice.count * (slice.map(\.count).max() ?? 0)
        }
    }
    let real = texts.reduce(0) { $0 + $1.count }

    print("")
    print("=== \(texts.count) real documents: sequential vs unsorted batch vs length-sorted batch ===")
    print("  sequential (\(texts.count) calls) : \(pad(sequential)) ms   padded chars \(real)")
    print("  batch, input order      : \(pad(unsorted)) ms   padded chars \(paddedChars(texts, chunk: 16))")
    print("  batch, length-sorted    : \(pad(sortedBatch)) ms   padded chars \(paddedChars(sorted, chunk: 16))")
    print("  sorted vs unsorted      : \(String(format: "%.2fx", unsorted / max(sortedBatch, 0.001))) faster")
    print("  sorted vs sequential    : \(String(format: "%.2fx", sequential / max(sortedBatch, 0.001))) faster")
}

// MARK: - Experiment 6: does the MLX buffer-cache limit matter for this workload?

// MLX keeps FREED Metal buffers in a pool (`MetalAllocator::max_pool_size_`) so the next forward
// pass reuses them instead of asking Metal for new allocations. `GPU.cacheLimit` caps that pool and
// defaults to the memory limit, i.e. effectively unbounded. This measures whether the pool is doing
// anything for us: same work at the default limit, with the cache disabled, and with a small cap.
if selected.contains(6) {
    guard let sample = corpus.samples.max(by: { $0.words < $1.words }) else { exit(0) }

    func scale(_ bytes: Int) -> String { String(format: "%.1f MB", Double(bytes) / 1_048_576) }

    print("")
    print("=== MLX memory configuration ===")
    print("  memoryLimit : \(scale(GPU.memoryLimit))")
    print("  cacheLimit  : \(scale(GPU.cacheLimit))   (defaults to the memory limit)")

    let defaultLimit = GPU.cacheLimit
    let configurations: [(name: String, limit: Int)] = [
        ("default (unbounded)", defaultLimit),
        ("cacheLimit = 0", 0),
        ("cacheLimit = 2 MB", 2 * 1_048_576),
        ("cacheLimit = 256 MB", 256 * 1_048_576),
        ("back to default", defaultLimit)
    ]

    print("")
    print("=== \(sample.words)-word sample, \(repetitions) reps per configuration ===")
    for configuration in configurations {
        GPU.set(cacheLimit: configuration.limit)
        GPU.clearCache()
        var timings: [Double] = []
        for _ in 0 ..< repetitions {
            timings.append(try await milliseconds { _ = try await engine.embed(sample.text) })
        }
        let snapshot = GPU.snapshot()
        print(
            "  \(label(configuration.name, 22)) median \(pad(median(timings), 8)) ms   min \(pad(timings.min() ?? 0, 8)) ms"
            + "   active \(scale(snapshot.activeMemory))  cache \(scale(snapshot.cacheMemory))"
        )
    }
}

// MARK: - Experiment 7: all three MLX memory knobs

// MLX exposes three independent limits, and none of them is set anywhere in our stack:
//
//  - cacheLimit  — caps the pool of FREED Metal buffers kept for reuse. Defaults to memoryLimit.
//  - memoryLimit — allocation ceiling; malloc waits on scheduled tasks past it, then errors or
//                  swaps depending on `relaxed`. Defaults to min(1.5 x maxWorkingSet, 0.95 x RAM).
//  - wiredLimit  — size of the Metal residency set, i.e. how much MLX asks the OS to keep wired
//                  and non-evictable. Defaults to ZERO (`size_t wired_limit_{0}`), so nothing the
//                  allocator holds — including the model weights — is pinned. This is the only one
//                  of the three that could plausibly affect page residency between calls.
//
// Timing is inlined rather than routed through `milliseconds` so the closures stay on one
// isolation domain inside `withWiredLimit`.
if selected.contains(7) {
    guard let sample = corpus.samples.max(by: { $0.words < $1.words }) else { exit(0) }
    let megabyte = 1_048_576

    func scale(_ bytes: Int) -> String { String(format: "%.1f MB", Double(bytes) / 1_048_576) }
    func report(_ name: String, _ timings: [Double]) {
        let snapshot = GPU.snapshot()
        print(
            "  \(label(name, 26)) median \(pad(median(timings), 8)) ms   min \(pad(timings.min() ?? 0, 8)) ms"
            + "   active \(scale(snapshot.activeMemory))  cache \(scale(snapshot.cacheMemory))"
        )
    }

    let info = GPU.deviceInfo()
    let maxWorkingSet = Int(info.maxRecommendedWorkingSetSize)
    let defaultCache = GPU.cacheLimit
    let defaultMemory = GPU.memoryLimit

    print("")
    print("=== device ===")
    print("  architecture               : \(info.architecture)")
    print("  system memory              : \(scale(info.memorySize))")
    print("  maxRecommendedWorkingSet   : \(scale(maxWorkingSet))")
    print("  memoryLimit (default)      : \(scale(defaultMemory))")
    print("  cacheLimit  (default)      : \(scale(defaultCache))")
    print("  wiredLimit  (default)      : 0.0 MB  (nothing pinned)")

    print("")
    print("=== \(sample.words)-word sample, \(repetitions) reps per configuration ===")

    // --- cacheLimit ---
    for limit in [defaultCache, 0, 256 * megabyte] {
        GPU.set(cacheLimit: limit)
        var timings: [Double] = []
        for _ in 0 ..< repetitions {
            let start = ContinuousClock().now
            _ = try await engine.embed(sample.text)
            let d = (ContinuousClock().now - start).components
            timings.append(Double(d.seconds) * 1000 + Double(d.attoseconds) / 1e15)
        }
        report("cacheLimit \(scale(limit))", timings)
    }
    GPU.set(cacheLimit: defaultCache)

    // --- memoryLimit --- raising it above the default is meaningless (it already exceeds RAM),
    // so this tests whether TIGHTENING it to something a desktop app might choose costs anything.
    for limit in [defaultMemory, 4096 * megabyte, 1024 * megabyte] {
        GPU.set(memoryLimit: limit, relaxed: true)
        var timings: [Double] = []
        for _ in 0 ..< repetitions {
            let start = ContinuousClock().now
            _ = try await engine.embed(sample.text)
            let d = (ContinuousClock().now - start).components
            timings.append(Double(d.seconds) * 1000 + Double(d.attoseconds) / 1e15)
        }
        report("memoryLimit \(scale(limit))", timings)
    }
    GPU.set(memoryLimit: defaultMemory, relaxed: true)

    // --- wiredLimit --- must stay <= maxRecommendedWorkingSetSize or mlx throws.
    for limit in [0, 512 * megabyte, min(2048 * megabyte, maxWorkingSet), maxWorkingSet] {
        let timings = try await GPU.withWiredLimit(limit) { () async throws -> [Double] in
            var collected: [Double] = []
            for _ in 0 ..< repetitions {
                let start = ContinuousClock().now
                _ = try await engine.embed(sample.text)
                let d = (ContinuousClock().now - start).components
                collected.append(Double(d.seconds) * 1000 + Double(d.attoseconds) / 1e15)
            }
            return collected
        }
        report("wiredLimit \(scale(limit))", timings)
    }
}

// MARK: - Experiment 4: idle probe

if selected.contains(4), let sample = corpus.samples.max(by: { $0.words < $1.words }) {
    let idleSeconds = 120
    var warm: [Double] = []
    for _ in 0 ..< 5 {
        warm.append(try await milliseconds { _ = try await engine.embed(sample.text) })
    }
    try await Task.sleep(for: .seconds(idleSeconds))
    let afterIdle = try await milliseconds { _ = try await engine.embed(sample.text) }
    let recovered = try await milliseconds { _ = try await engine.embed(sample.text) }

    print("")
    print("=== idle probe (\(sample.label), \(sample.words) words, \(idleSeconds)s gap) ===")
    print("  steady-state median : \(pad(median(warm))) ms")
    print("  first after idle    : \(pad(afterIdle)) ms")
    print("  immediately after   : \(pad(recovered)) ms")
}

print("")
