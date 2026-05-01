import Foundation
import SemanticSearch
import SwiftUI

/// Single source of truth for the GUI demo. Owns the engine, the loaded
/// corpus, the precomputed document vectors, the current query, and the
/// ranked results.
@Observable
@MainActor
final class DemoState {
    // MARK: - Public observable state

    /// High-level lifecycle state shown in the UI.
    enum Phase: Equatable {
        case idle
        case preparingModel(fraction: Double, message: String)
        case ready
        case embeddingCorpus(progress: String)
        case failed(message: String)
    }

    var phase: Phase = .idle

    /// Where the current corpus came from (for the source label in the UI).
    enum SourceKind: Equatable {
        case none
        case demo
        case folder(URL, fileCount: Int)
    }

    var sourceKind: SourceKind = .none
    var documents: [Document] = []
    private var documentVectors: [[Float]] = []

    /// What the user typed in the search field. Live-bound from the UI.
    var query: String = ""

    /// Ranked results for the current query.
    struct Hit: Identifiable {
        let id: String
        let document: Document
        let score: Float
    }
    var results: [Hit] = []
    var selectedHitID: String?
    var lastSearchTimingMS: Double?

    /// On-disk model location, set after prepare completes. Useful in the UI.
    var modelLocation: URL?

    // MARK: - Private

    private let engine: SemanticSearchEngine
    private let topK: Int = 20
    private var searchTask: Task<Void, Never>?

    // MARK: - Init

    init() {
        self.engine = SemanticSearchEngine()
    }

    // MARK: - Lifecycle

    func prepareModelIfNeeded() async {
        if case .ready = phase { return }
        if case .embeddingCorpus = phase { return }

        phase = .preparingModel(fraction: 0, message: "Locating…")
        do {
            for try await progress in engine.prepare() {
                phase = .preparingModel(
                    fraction: progress.fractionCompleted,
                    message: friendlyPhase(progress)
                )
            }
            modelLocation = engine.modelLocationOnDisk()
            phase = .ready

            // Auto-load the demo corpus on first launch so the user has
            // something to query immediately.
            if documents.isEmpty {
                await useDemoCorpus()
            }
        } catch {
            phase = .failed(message: "Couldn't load model: \(error.localizedDescription)")
        }
    }

    func useDemoCorpus() async {
        let docs = DemoCorpus.documents.map(Document.from)
        await ingest(docs, source: .demo)
    }

    func chooseFolder() async {
        let panel = NSOpenPanel()
        panel.title = "Choose a folder of .txt or .md files"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false

        guard panel.runModal() == .OK, let url = panel.url else { return }

        let docs: [Document]
        do {
            docs = try collectTextFiles(under: url)
        } catch {
            phase = .failed(message: "Couldn't read folder: \(error.localizedDescription)")
            return
        }

        if docs.isEmpty {
            phase = .failed(message: "No .txt or .md files found under \(url.lastPathComponent).")
            return
        }

        await ingest(docs, source: .folder(url, fileCount: docs.count))
    }

    func resetModel() async {
        searchTask?.cancel()
        searchTask = nil
        results = []
        documents = []
        documentVectors = []
        sourceKind = .none

        await engine.unload()
        do {
            try await engine.deleteDownloadedWeights()
        } catch {
            phase = .failed(message: "Couldn't delete weights: \(error.localizedDescription)")
            return
        }
        phase = .idle
        await prepareModelIfNeeded()
    }

    /// Re-runs the search with the current query. Called whenever `query` changes
    /// (debounced) or after a corpus swap.
    func search() async {
        searchTask?.cancel()
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !documentVectors.isEmpty else {
            results = []
            lastSearchTimingMS = nil
            return
        }

        let task = Task<Void, Never> { @MainActor in
            // Tiny debounce so typing doesn't trigger an embed per keystroke.
            // `try?` is intentional here: Task.sleep throws on cancellation,
            // and the very next line checks Task.isCancelled and bails — the
            // thrown error carries no information we don't already check for.
            try? await Task.sleep(for: .milliseconds(120))
            if Task.isCancelled { return }

            let start = ContinuousClock.now
            do {
                let queryVector = try await engine.embed(trimmed)
                if Task.isCancelled { return }

                let ranked = VectorMath.match(
                    query: queryVector,
                    against: documentVectors,
                    top: topK
                )
                let elapsed = ContinuousClock.now - start
                let elapsedMS = Double(elapsed.components.attoseconds) / 1e15
                    + Double(elapsed.components.seconds) * 1000

                let newResults = ranked.map { hit in
                    Hit(
                        id: documents[hit.index].id,
                        document: documents[hit.index],
                        score: hit.score
                    )
                }
                results = newResults
                lastSearchTimingMS = elapsedMS

                // Keep selection if it still points at a doc in the new results,
                // otherwise jump to the top hit so the detail pane is never empty
                // while results exist.
                if let current = selectedHitID,
                   newResults.contains(where: { $0.id == current }) {
                    // keep selection
                } else {
                    selectedHitID = newResults.first?.id
                }
            } catch is CancellationError {
                // Replaced by another search.
            } catch {
                phase = .failed(message: "Search failed: \(error.localizedDescription)")
            }
        }
        searchTask = task
    }

    // MARK: - Helpers

    private func ingest(_ docs: [Document], source: SourceKind) async {
        guard case .ready = phase else {
            // If we're not ready, ingest will run after prepare via prepareModelIfNeeded.
            documents = docs
            sourceKind = source
            return
        }

        phase = .embeddingCorpus(progress: "Embedding \(docs.count) document\(docs.count == 1 ? "" : "s")…")
        do {
            let texts = docs.map(\.text)
            let vectors = try await engine.embed(batch: texts)
            documents = docs
            documentVectors = vectors
            sourceKind = source
            results = []
            selectedHitID = nil
            phase = .ready
            await search()
        } catch {
            phase = .failed(message: "Couldn't embed corpus: \(error.localizedDescription)")
        }
    }

    private func collectTextFiles(under root: URL) throws -> [Document] {
        let fm = FileManager.default
        var docs: [Document] = []
        let allowedExtensions: Set<String> = ["txt", "md", "markdown"]

        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        for case let url as URL in enumerator {
            let ext = url.pathExtension.lowercased()
            guard allowedExtensions.contains(ext) else { continue }
            do {
                let doc = try Document.from(fileAt: url)
                if !doc.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    docs.append(doc)
                }
            } catch {
                // Skip files we can't read; don't fail the whole import.
                continue
            }
        }
        return docs.sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
    }

    private func friendlyPhase(_ progress: PrepareProgress) -> String {
        switch progress.phase {
        case .locating:        return "Locating model…"
        case .downloading:
            if let bytes = progress.bytesDownloaded, let total = progress.bytesTotal, total > 0 {
                let mb = Double(bytes) / 1_048_576
                let totalMB = Double(total) / 1_048_576
                return String(format: "Downloading model %.1f / %.1f MB", mb, totalMB)
            }
            return "Downloading model…"
        case .loadingWeights:  return "Loading weights…"
        case .warmingUp:       return "Warming up…"
        case .done:            return "Ready"
        }
    }
}
