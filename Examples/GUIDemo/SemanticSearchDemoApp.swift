import SwiftUI

@main
struct SemanticSearchDemoApp: App {
    @State private var demoState = DemoState()

    var body: some Scene {
        WindowGroup("Semantic Search") {
            ContentView()
                .environment(demoState)
                .task {
                    await demoState.prepareModelIfNeeded()
                }
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) { }
            CommandGroup(after: .toolbar) {
                Button("Choose Corpus Folder…") {
                    Task { await demoState.chooseFolder() }
                }
                .keyboardShortcut("o", modifiers: [.command])

                Button("Use Built-in Corpus") {
                    Task { await demoState.useDemoCorpus() }
                }
                .keyboardShortcut("d", modifiers: [.command])

                Divider()

                Button("Reset Model") {
                    Task { await demoState.resetModel() }
                }
            }
        }
    }
}
