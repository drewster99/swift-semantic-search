import SwiftUI
import SemanticSearch

struct ContentView: View {
    @Environment(DemoState.self) private var state

    var body: some View {
        @Bindable var state = state

        NavigationSplitView {
            ResultsList(state: state)
                .frame(minWidth: 380, idealWidth: 460)
        } detail: {
            DetailPane(state: state)
                .frame(minWidth: 420)
        }
        .navigationTitle("Semantic Search")
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                SourcePickerMenu(state: state)
            }
            ToolbarItem(placement: .principal) {
                SearchField(query: $state.query, isReady: state.isReady)
                    .frame(minWidth: 280, idealWidth: 420)
                    .onChange(of: state.query) { _, _ in
                        Task { await state.search() }
                    }
            }
            ToolbarItem(placement: .primaryAction) {
                StatusPill(state: state)
            }
        }
        .overlay {
            LoadingOverlay(state: state)
        }
        .frame(minWidth: 920, minHeight: 580)
    }
}

// MARK: - Source picker

private struct SourcePickerMenu: View {
    let state: DemoState

    var body: some View {
        Menu {
            Button {
                Task { await state.useDemoCorpus() }
            } label: {
                Label("Built-in demo corpus", systemImage: "books.vertical")
            }
            Button {
                Task { await state.chooseFolder() }
            } label: {
                Label("Choose folder of .txt / .md…", systemImage: "folder")
            }
            Divider()
            Button(role: .destructive) {
                Task { await state.resetModel() }
            } label: {
                Label("Delete downloaded model", systemImage: "trash")
            }
        } label: {
            switch state.sourceKind {
            case .none:
                Label("Source", systemImage: "tray")
            case .demo:
                Label("Demo corpus", systemImage: "books.vertical")
            case .folder(let url, let count):
                Label("\(url.lastPathComponent) — \(count) docs", systemImage: "folder")
            }
        }
        .menuStyle(.borderlessButton)
    }
}

// MARK: - Search field

private struct SearchField: View {
    @Binding var query: String
    let isReady: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search semantically…", text: $query)
                .textFieldStyle(.plain)
                .disabled(!isReady)
            if !query.isEmpty {
                Button {
                    query = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(.separator, lineWidth: 0.5)
        )
    }
}

// MARK: - Status pill

private struct StatusPill: View {
    let state: DemoState

    var body: some View {
        HStack(spacing: 6) {
            switch state.phase {
            case .idle:
                Image(systemName: "circle.dashed")
                    .foregroundStyle(.secondary)
                Text("Idle").foregroundStyle(.secondary)
            case .preparingModel(_, let message):
                ProgressView().controlSize(.small)
                Text(message).foregroundStyle(.secondary).lineLimit(1)
            case .embeddingCorpus(let message):
                ProgressView().controlSize(.small)
                Text(message).foregroundStyle(.secondary).lineLimit(1)
            case .ready:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                if let ms = state.lastSearchTimingMS {
                    Text(String(format: "%.0f ms", ms))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                } else {
                    Text("Ready").foregroundStyle(.secondary)
                }
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(message).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .font(.system(.caption, design: .default).monospacedDigit())
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.regularMaterial, in: Capsule())
    }
}

// MARK: - Results list

private struct ResultsList: View {
    @Bindable var state: DemoState

    var body: some View {
        Group {
            if state.results.isEmpty {
                EmptyStateView(state: state)
            } else {
                List(state.results, selection: $state.selectedHitID) { hit in
                    ResultRow(hit: hit)
                        .tag(hit.id)
                }
            }
        }
    }
}

private struct ResultRow: View {
    let hit: DemoState.Hit
    private static let scoreFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.minimumFractionDigits = 3
        f.maximumFractionDigits = 3
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(hit.document.title)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Text(Self.scoreFormatter.string(from: NSNumber(value: hit.score)) ?? "—")
                    .font(.system(.subheadline, design: .monospaced))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            if let topic = hit.document.topicLabel {
                Text(topic.uppercased())
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 5).padding(.vertical, 2)
                    .background(.secondary.opacity(0.12), in: Capsule())
            }
            SimilarityBar(score: hit.score)
                .frame(height: 4)
            Text(hit.document.snippet)
                .font(.system(.callout))
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .padding(.vertical, 4)
    }
}

private struct SimilarityBar: View {
    let score: Float

    var body: some View {
        GeometryReader { geo in
            let clamped = max(0, min(1, CGFloat(score)))
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(.tertiary.opacity(0.4))
                RoundedRectangle(cornerRadius: 2)
                    .fill(barColor(score: score).gradient)
                    .frame(width: geo.size.width * clamped)
            }
        }
    }

    private func barColor(score: Float) -> Color {
        switch score {
        case ..<0.3:  return .secondary
        case ..<0.5:  return .yellow
        case ..<0.7:  return .orange
        default:      return .green
        }
    }
}

// MARK: - Detail pane

private struct DetailPane: View {
    let state: DemoState

    private var selectedHit: DemoState.Hit? {
        guard let id = state.selectedHitID else { return nil }
        return state.results.first { $0.id == id }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let hit = selectedHit {
                    Text(hit.document.title)
                        .font(.title2.weight(.semibold))
                    HStack(spacing: 12) {
                        Label(String(format: "score %.4f", hit.score), systemImage: "scope")
                            .font(.system(.subheadline).monospacedDigit())
                        if let topic = hit.document.topicLabel {
                            Label(topic, systemImage: "tag")
                                .font(.subheadline)
                        }
                        if let url = hit.document.fileURL {
                            Label(url.path, systemImage: "doc")
                                .font(.system(.caption, design: .monospaced))
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .help(url.path)
                        }
                    }
                    .foregroundStyle(.secondary)
                    Divider()
                    Text(hit.document.text)
                        .font(.system(.body))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    DetailEmptyState(documentCount: state.documents.count)
                }
                Spacer(minLength: 0)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(.background)
    }
}

private struct DetailEmptyState: View {
    let documentCount: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: "magnifyingglass.circle")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text(documentCount == 0
                 ? "Pick a corpus to begin."
                 : "Type a query to rank \(documentCount) document\(documentCount == 1 ? "" : "s").")
                .font(.title3)
                .foregroundStyle(.secondary)
            Text(DemoCorpus.blurb)
                .font(.callout)
                .foregroundStyle(.tertiary)
                .frame(maxWidth: 480, alignment: .leading)
        }
    }
}

// MARK: - Empty state for the results list

private struct EmptyStateView: View {
    let state: DemoState

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch state.sourceKind {
            case .none:
                Text("No corpus loaded")
                    .font(.headline)
                Text("Pick **Built-in demo corpus** or **Choose folder…** from the toolbar.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            case .demo:
                Text("Built-in corpus loaded")
                    .font(.headline)
                Text("Try queries like:")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                ExampleQuery(text: "what causes the northern lights")
                ExampleQuery(text: "stopping data races between threads in Swift")
                ExampleQuery(text: "interval workout for max aerobic capacity")
                ExampleQuery(text: "ratio of browns to greens for healthy compost")
            case .folder(let url, let count):
                Text("Folder loaded")
                    .font(.headline)
                Text("\(count) document\(count == 1 ? "" : "s") from `\(url.lastPathComponent)`")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Text("Type a query in the search field above to rank them.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct ExampleQuery: View {
    let text: String

    var body: some View {
        HStack {
            Image(systemName: "arrow.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Text(text)
                .font(.system(.callout, design: .default).italic())
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Loading overlay

private struct LoadingOverlay: View {
    let state: DemoState

    var body: some View {
        switch state.phase {
        case .preparingModel(let fraction, let message):
            ZStack {
                Color.black.opacity(0.18).ignoresSafeArea()
                VStack(spacing: 16) {
                    ProgressView(value: fraction, total: 1.0)
                        .progressViewStyle(.linear)
                        .frame(width: 320)
                    Text(message)
                        .font(.callout)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .padding(28)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
            .transition(.opacity)
        default:
            EmptyView()
        }
    }
}

// MARK: - Convenience

extension DemoState {
    var isReady: Bool {
        if case .ready = phase { return true }
        return false
    }
}
