import Foundation

/// One document the engine has embedded and can rank against. Holds the
/// display metadata plus an opaque source for "where it came from" so the UI
/// can show a sensible row and a sensible detail pane.
struct Document: Identifiable, Hashable {
    enum Source: Hashable {
        case demo(topic: String)
        case file(URL)
    }

    let id: String
    let title: String
    let text: String
    let source: Source

    var topicLabel: String? {
        if case .demo(let topic) = source { return topic }
        return nil
    }

    var fileURL: URL? {
        if case .file(let url) = source { return url }
        return nil
    }

    /// First ~140 chars of the document text on a single line.
    var snippet: String {
        let trimmed = text.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= 140 { return trimmed }
        let cutoff = trimmed.index(trimmed.startIndex, offsetBy: 140)
        return String(trimmed[..<cutoff]) + "…"
    }
}

extension Document {
    static func from(_ doc: DemoCorpus.Document) -> Document {
        Document(
            id: doc.id,
            title: doc.id,
            text: doc.text,
            source: .demo(topic: doc.topic)
        )
    }

    static func from(fileAt url: URL) throws -> Document {
        let raw = try String(contentsOf: url, encoding: .utf8)
        return Document(
            id: url.path,
            title: url.lastPathComponent,
            text: raw,
            source: .file(url)
        )
    }
}
