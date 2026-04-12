import Foundation
import Hub
import MLXEmbedders

/// Resolves where the weight files for a given `EmbeddingModel` live, if anywhere.
///
/// Search order:
/// 1. **Bundle.** A subdirectory under `Bundle.main` matching `model.bundleSubdirectory`.
///    The host app can ship a model inline this way and skip the first-launch download.
///    A bundled location only counts as "present" when its `config.json` exists.
/// 2. **Cache directory.** Wherever `MLXEmbedders.ModelConfiguration.modelDirectory(hub:)`
///    says the library will store this model when downloaded, anchored at our own cache
///    base under `Application Support/<bundleID>/SemanticSearch/models/`. Counted as
///    "present" only when its `config.json` exists.
/// 3. **Absent.** Neither location holds the model; `prepare()` will need to download.
internal struct ModelLocator: Sendable {
    let model: EmbeddingModel

    enum Location: Sendable {
        case bundled(URL)
        case downloaded(URL)
        case absent
    }

    func locate() -> Location {
        if let url = bundleLocation() {
            return .bundled(url)
        }
        if let url = downloadedLocation() {
            return .downloaded(url)
        }
        return .absent
    }

    /// Root directory under which all SemanticSearch caches live for the host app.
    /// Created on demand.
    func cacheBaseURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let bundleID = Bundle.main.bundleIdentifier ?? "SemanticSearch"
        let dir = appSupport
            .appendingPathComponent(bundleID, isDirectory: true)
            .appendingPathComponent("SemanticSearch", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// The canonical local directory where MLXEmbedders will store this model after
    /// downloading. Computed by asking the library itself rather than hand-rolling the
    /// HuggingFace cache layout — that way we follow whatever scheme the library uses
    /// (snapshots, content-addressed, etc.) without having to mirror it.
    func canonicalDownloadDirectory() throws -> URL {
        let hub = try HubApi(downloadBase: cacheBaseURL())
        let configuration = MLXEmbedders.ModelConfiguration(id: model.huggingFaceRepo)
        return configuration.modelDirectory(hub: hub)
    }

    /// Returns a bundle URL only when the directory exists AND contains a `config.json`.
    func bundleLocation() -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else { return nil }
        let candidate = resourceURL.appendingPathComponent(model.bundleSubdirectory, isDirectory: true)
        return Self.validatedModelDirectory(candidate)
    }

    /// Returns the cache URL only when the directory exists AND contains a `config.json`.
    func downloadedLocation() -> URL? {
        guard let dir = try? canonicalDownloadDirectory() else { return nil }
        return Self.validatedModelDirectory(dir)
    }

    /// Removes the cache directory for this model. Bundled files are never touched.
    func deleteDownloaded() throws {
        let dir = try canonicalDownloadDirectory()
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }

    /// Returns `url` if it points at a directory that contains the canonical
    /// `config.json` marker file. The marker is what `MLXEmbedders.loadSynchronous`
    /// reads first when loading a model, so its presence is the strongest signal
    /// that the directory holds a real, complete model.
    private static func validatedModelDirectory(_ url: URL) -> URL? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return nil
        }
        let configFile = url.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configFile.path) else {
            return nil
        }
        return url
    }
}
