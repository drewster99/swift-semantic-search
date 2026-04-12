import Foundation

/// Resolves where the weight files for a given `EmbeddingModel` live, if anywhere.
///
/// Search order:
/// 1. **Bundle.** A subdirectory under `Bundle.main` matching `model.bundleSubdirectory`.
///    The host app can ship a model inline this way and skip the first-launch download.
/// 2. **Cache directory.** `Application Support/<bundleID>/SemanticSearch/models/<repo>/`,
///    populated on first download.
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

    /// Directory used for the HuggingFace download cache for THIS model. Existence
    /// of this directory does not guarantee a complete download — `downloadedLocation()`
    /// also requires it to be non-empty.
    func cacheDirectory() throws -> URL {
        // The HuggingFace Hub layout is `<base>/models/<repo>/...`. We pass our base
        // to `HubApi(downloadBase:)` and let the Hub manage the inner structure.
        try cacheBaseURL()
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(model.huggingFaceRepo, isDirectory: true)
    }

    /// Returns a bundle URL only when the directory exists AND contains files.
    func bundleLocation() -> URL? {
        guard let url = Bundle.main.url(
            forResource: model.bundleSubdirectory,
            withExtension: nil
        ) else {
            return nil
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return nil
        }
        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: url.path),
              !contents.isEmpty else {
            return nil
        }
        return url
    }

    /// Returns the cache URL only when the directory exists AND is non-empty.
    func downloadedLocation() -> URL? {
        guard let dir = try? cacheDirectory() else { return nil }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return nil
        }
        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: dir.path),
              !contents.isEmpty else {
            return nil
        }
        return dir
    }

    /// Removes the cache directory for this model. Bundled files are never touched.
    func deleteDownloaded() throws {
        let dir = try cacheDirectory()
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }
}
