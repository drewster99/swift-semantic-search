import Foundation
import SemanticSearch

// MARK: - Argument parsing

private struct Arguments {
    var clean: Bool = false
    var useDemoCorpus: Bool = false
    /// Direct corpus files passed positionally on the command line. Each file
    /// is read in full and becomes one document.
    var corpusFiles: [String] = []
    /// File-list paths passed via `--corpus-file-list`. Each list file is read
    /// line-by-line; every non-blank, non-comment line names a corpus file
    /// that is itself read in full and becomes one document.
    var corpusFileLists: [String] = []
    var topK: Int = 5
}

private enum ArgError: Error, CustomStringConvertible {
    case unknownFlag(String)
    case missingValue(String)
    case invalidTopK(String)

    var description: String {
        switch self {
        case .unknownFlag(let f):    return "Unknown flag: \(f)"
        case .missingValue(let f):   return "Flag \(f) needs a value"
        case .invalidTopK(let v):    return "--top expects a positive integer, got \(v)"
        }
    }
}

private func parseArguments(_ argv: [String]) throws -> Arguments {
    var args = Arguments()
    var index = 0
    while index < argv.count {
        let token = argv[index]
        switch token {
        case "--clean":
            args.clean = true
        case "--demo-corpus":
            args.useDemoCorpus = true
        case "--corpus-file-list":
            guard index + 1 < argv.count else { throw ArgError.missingValue(token) }
            args.corpusFileLists.append(argv[index + 1])
            index += 1
        case "--top":
            guard index + 1 < argv.count else { throw ArgError.missingValue(token) }
            guard let n = Int(argv[index + 1]), n > 0 else {
                throw ArgError.invalidTopK(argv[index + 1])
            }
            args.topK = n
            index += 1
        case "-h", "--help":
            // Falls through to "no args" path below.
            return Arguments()
        default:
            if token.hasPrefix("--") {
                throw ArgError.unknownFlag(token)
            }
            args.corpusFiles.append(token)
        }
        index += 1
    }
    return args
}

private func printUsage() {
    let exe = "cli-demo"
    let demoPath = "Examples/CLIDemo/DemoCorpus.swift"
    let text = """
        cli-demo — interactive semantic search REPL backed by SemanticSearch.

        USAGE
          \(exe) --demo-corpus
          \(exe) <file> [<file> ...]
          \(exe) --corpus-file-list <listfile> [--corpus-file-list <listfile> ...]
          \(exe) --clean
          \(exe) [options] [--top N] ...

        OPTIONS
          --demo-corpus              Use the built-in 24-document corpus
                                     (\(demoPath)). Spans cooking, software,
                                     astronomy, music, fitness, and gardening;
                                     includes sample queries to try.
          <path>                     One or more corpus files passed
                                     positionally. Each file is read in full
                                     and becomes one document.
          --corpus-file-list <path>  Path to a list file. Each non-blank,
                                     non-comment line in the list file is the
                                     path to a corpus file. Each referenced
                                     corpus file is read in full and becomes
                                     one document. Repeatable.
          --top N                    Number of results to display per query
                                     (default 5).
          --clean                    Delete the downloaded embedding model
                                     from the cache directory and exit.
                                     Bundled models are not touched.
          -h, --help                 Show this message.

        Run with no arguments to see this help. On first run, the model
        weights (~330 MB) are downloaded into the user's Application Support
        cache; subsequent runs reuse them.
        """
    print(text)
}

// MARK: - Corpus loading

private struct CorpusDoc {
    let id: String
    let topic: String?
    let text: String
}

private func loadCorpus(from arguments: Arguments) throws -> [CorpusDoc] {
    var docs: [CorpusDoc] = []

    if arguments.useDemoCorpus {
        for d in DemoCorpus.documents {
            docs.append(CorpusDoc(id: d.id, topic: d.topic, text: d.text))
        }
    }

    // Direct corpus files: each file is one document.
    for path in arguments.corpusFiles {
        docs.append(try loadCorpusFile(at: path))
    }

    // List files: each line is a path to a corpus file (one document each).
    for listPath in arguments.corpusFileLists {
        let listURL = URL(fileURLWithPath: listPath)
        let raw = try String(contentsOf: listURL, encoding: .utf8)
        let listDirectory = listURL.deletingLastPathComponent()

        for line in raw.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }

            // Relative paths in the list file resolve relative to the list
            // file's directory, which matches users' intuition when the list
            // sits next to the corpus.
            let referenced: URL
            if (trimmed as NSString).isAbsolutePath {
                referenced = URL(fileURLWithPath: trimmed)
            } else {
                referenced = listDirectory.appendingPathComponent(trimmed)
            }
            docs.append(try loadCorpusFile(at: referenced.path))
        }
    }

    return docs
}

/// Reads a corpus file in full and returns it as a single document. The id is
/// the file's basename so REPL output stays readable; topic is `nil` because
/// arbitrary user files don't carry topic metadata.
private func loadCorpusFile(at path: String) throws -> CorpusDoc {
    let url = URL(fileURLWithPath: path)
    let raw = try String(contentsOf: url, encoding: .utf8)
    return CorpusDoc(
        id: url.lastPathComponent,
        topic: nil,
        text: raw
    )
}

// MARK: - Progress rendering

private func formatBytes(_ bytes: Int64) -> String {
    let mb = Double(bytes) / 1_048_576.0
    return String(format: "%.1f MB", mb)
}

private func renderProgressBar(fraction: Double, width: Int = 30) -> String {
    let clamped = max(0.0, min(1.0, fraction))
    let filled = Int((Double(width) * clamped).rounded())
    let bar = String(repeating: "#", count: filled) + String(repeating: "-", count: width - filled)
    return "[\(bar)]"
}

private func printProgress(_ progress: PrepareProgress) {
    let phase: String
    switch progress.phase {
    case .locating:        phase = "locating     "
    case .downloading:     phase = "downloading  "
    case .loadingWeights:  phase = "loading      "
    case .warmingUp:       phase = "warming up   "
    case .done:            phase = "done         "
    }

    var suffix = ""
    if let downloaded = progress.bytesDownloaded {
        if let total = progress.bytesTotal, total > 0 {
            suffix = " \(formatBytes(downloaded)) / \(formatBytes(total))"
        } else {
            suffix = " \(formatBytes(downloaded))"
        }
    }

    let bar = renderProgressBar(fraction: progress.fractionCompleted)
    let percent = String(format: "%5.1f%%", progress.fractionCompleted * 100)
    // Carriage return so each line overwrites the previous in the terminal.
    let line = "\r\(phase) \(bar) \(percent)\(suffix)"
    FileHandle.standardOutput.write(Data(line.utf8))
    if progress.phase == .done {
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}

// MARK: - REPL

private func formatScore(_ score: Float) -> String {
    String(format: "%.4f", score)
}

private func formatMs(_ seconds: Double) -> String {
    String(format: "%6.1f ms", seconds * 1000)
}

private func runREPL(
    engine: SemanticSearchEngine,
    docs: [CorpusDoc],
    docVectors: [[Float]],
    topK: Int
) async {
    print("")
    print("Type a query and press return. Empty line or :quit to exit. :help for commands.")
    print("")

    while true {
        FileHandle.standardOutput.write(Data("> ".utf8))
        guard let raw = readLine(strippingNewline: true) else {
            print("")
            break
        }
        let query = raw.trimmingCharacters(in: .whitespaces)
        if query.isEmpty || query == ":quit" || query == ":q" || query == ":exit" {
            break
        }
        if query == ":help" || query == ":?" {
            print("Commands: :quit  :help")
            continue
        }

        let start = ContinuousClock.now
        do {
            let queryVector = try await engine.embed(query)
            let embedElapsed = ContinuousClock.now - start

            let rankStart = ContinuousClock.now
            let ranked = VectorMath.match(query: queryVector, against: docVectors, top: topK)
            let rankElapsed = ContinuousClock.now - rankStart
            let totalElapsed = ContinuousClock.now - start

            let embedSec = Double(embedElapsed.components.attoseconds) / 1e18
                + Double(embedElapsed.components.seconds)
            let rankSec = Double(rankElapsed.components.attoseconds) / 1e18
                + Double(rankElapsed.components.seconds)
            let totalSec = Double(totalElapsed.components.attoseconds) / 1e18
                + Double(totalElapsed.components.seconds)

            print("")
            for (rank, hit) in ranked.enumerated() {
                let doc = docs[hit.index]
                let header: String
                if let topic = doc.topic {
                    header = "  \(rank + 1). [\(formatScore(hit.score))]  \(doc.id)  (\(topic))"
                } else {
                    header = "  \(rank + 1). [\(formatScore(hit.score))]  \(doc.id)"
                }
                print(header)
                print("       \(doc.text)")
            }
            print("")
            print(
                "  embed: \(formatMs(embedSec))   rank \(docVectors.count) docs: \(formatMs(rankSec))   total: \(formatMs(totalSec))"
            )
            print("")
        } catch {
            print("error: \(error)")
        }
    }
}

// MARK: - Entry point

@main
struct CLIDemoApp {
    static func main() async {
        let argv = Array(CommandLine.arguments.dropFirst())

        if argv.isEmpty {
            printUsage()
            return
        }

        let arguments: Arguments
        do {
            arguments = try parseArguments(argv)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
            printUsage()
            exit(2)
        }

        // --clean: delete downloaded weights and exit. Engine must be unloaded
        // for delete to succeed; a fresh engine starts unloaded.
        if arguments.clean {
            let engine = SemanticSearchEngine()
            do {
                try await engine.deleteDownloadedWeights()
                print("Deleted downloaded model weights for \(engine.model.displayName).")
            } catch {
                FileHandle.standardError.write(Data("error: failed to delete weights: \(error)\n".utf8))
                exit(1)
            }
            return
        }

        if !arguments.useDemoCorpus && arguments.corpusFiles.isEmpty && arguments.corpusFileLists.isEmpty {
            printUsage()
            return
        }

        // Build corpus before touching the model so bad paths fail fast.
        let docs: [CorpusDoc]
        do {
            docs = try loadCorpus(from: arguments)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
        guard !docs.isEmpty else {
            FileHandle.standardError.write(Data("error: no documents loaded.\n".utf8))
            exit(1)
        }

        if arguments.useDemoCorpus {
            print(DemoCorpus.blurb)
            print("")
        }
        print("Loaded \(docs.count) document\(docs.count == 1 ? "" : "s").")

        // Prepare the engine with a progress bar.
        let engine = SemanticSearchEngine()
        print("Preparing model: \(engine.model.displayName)")
        do {
            for try await progress in engine.prepare() {
                printProgress(progress)
            }
        } catch {
            FileHandle.standardError.write(Data("\nerror: prepare failed: \(error)\n".utf8))
            exit(1)
        }

        // Report where the loaded weights live.
        if let url = engine.modelLocationOnDisk() {
            print("Model on disk: \(url.path)")
        }

        // Embed the corpus, with a one-line timing report.
        print("Embedding corpus...")
        let embedStart = ContinuousClock.now
        let docVectors: [[Float]]
        do {
            docVectors = try await engine.embed(batch: docs.map(\.text))
        } catch {
            FileHandle.standardError.write(Data("error: failed to embed corpus: \(error)\n".utf8))
            exit(1)
        }
        let embedElapsed = ContinuousClock.now - embedStart
        let embedSec = Double(embedElapsed.components.attoseconds) / 1e18
            + Double(embedElapsed.components.seconds)
        print(String(
            format: "Embedded %d documents in %.2f s (%.1f ms/doc).",
            docs.count, embedSec, embedSec * 1000.0 / Double(max(docs.count, 1))
        ))

        await runREPL(engine: engine, docs: docs, docVectors: docVectors, topK: arguments.topK)
        print("Goodbye.")
    }
}
