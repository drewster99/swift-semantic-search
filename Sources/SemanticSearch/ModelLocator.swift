import Foundation
import Hub
import MLXEmbedders

/// Resolves where the weight files for a given `EmbeddingModel` live, if anywhere.
///
/// Search order:
/// 1. **Bundle.** A subdirectory under `Bundle.main` matching `model.bundleSubdirectory`.
///    The host app can ship a model inline this way and skip the first-launch download.
/// 2. **Cache directory.** Wherever `MLXEmbedders.ModelConfiguration.modelDirectory(hub:)`
///    says the library will store this model when downloaded, anchored at our own cache
///    base under `Application Support/<bundleID>/SemanticSearch/models/`.
/// 3. **Absent.** Neither location holds a complete model; `prepare()` will need to download.
///
/// A location only counts as present when it holds every file a local load needs (see
/// `requiredModelFiles`), so an interrupted download is reported as absent rather than as a
/// model that will fail to load.
internal struct ModelLocator: Sendable {
    let model: EmbeddingModel

    /// The per-user Application Support directory the cache base is anchored at. Injectable so
    /// tests can exercise directory creation and backup exclusion without touching the real one.
    let applicationSupportRoot: URL

    enum Location: Sendable {
        case bundled(URL)
        case downloaded(URL)
        case absent
    }

    /// Files that must exist in a model directory for `MLXEmbedders` to load it without the
    /// network: the model config, the tokenizer (both halves), and at least one weights file
    /// (checked separately because the weight file names vary by model).
    static let requiredModelFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    /// Extension of the weight files; at least one must be present.
    static let weightsFileExtension = "safetensors"

    /// Used when the host process has no bundle identifier (command-line tools). Every such
    /// process shares this one cache directory.
    static let fallbackCacheNamespace = "SemanticSearch"

    init(model: EmbeddingModel, applicationSupportRoot: URL = .applicationSupportDirectory) {
        self.model = model
        self.applicationSupportRoot = applicationSupportRoot
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

    /// Root directory under which all SemanticSearch caches live for the host app. Pure path
    /// computation — nothing is created, so `locate()` has no side effects.
    func cacheBaseURL() -> URL {
        let namespace = Bundle.main.bundleIdentifier ?? Self.fallbackCacheNamespace
        return applicationSupportRoot
            .appendingPathComponent(namespace, isDirectory: true)
            .appendingPathComponent("SemanticSearch", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
    }

    /// Creates the cache base (if needed) and excludes it from iCloud / device backups. The
    /// weights are large and re-downloadable, which Apple's data-storage guidelines say must not
    /// be backed up. Called before a download, never from `locate()`.
    func prepareCacheBaseURL() throws -> URL {
        var directory = cacheBaseURL()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try directory.setResourceValues(values)
        return directory
    }

    /// The canonical local directory where MLXEmbedders will store this model after
    /// downloading. Computed by asking the library itself rather than hand-rolling the
    /// HuggingFace cache layout — that way we follow whatever scheme the library uses
    /// without having to mirror it.
    func canonicalDownloadDirectory() -> URL {
        let hub = HubApi(downloadBase: cacheBaseURL())
        let configuration = MLXEmbedders.ModelConfiguration(id: model.huggingFaceRepo)
        return configuration.modelDirectory(hub: hub)
    }

    /// Returns the bundle URL only when it holds a complete model.
    func bundleLocation() -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else { return nil }
        let candidate = resourceURL.appendingPathComponent(model.bundleSubdirectory, isDirectory: true)
        return Self.validatedModelDirectory(candidate)
    }

    /// Returns the cache URL only when it holds a complete model.
    func downloadedLocation() -> URL? {
        Self.validatedModelDirectory(canonicalDownloadDirectory())
    }

    /// Removes the cache directory for this model. Bundled files are never touched.
    func deleteDownloaded() throws {
        let dir = canonicalDownloadDirectory()
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }

    /// Returns `url` if it is a directory holding every file in `requiredModelFiles` plus at
    /// least one weights file; `nil` otherwise. A directory listing failure is treated as
    /// "not a complete model" because the caller would only be able to download or report absent.
    static func validatedModelDirectory(_ url: URL) -> URL? {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return nil
        }
        for name in requiredModelFiles where !fileManager.fileExists(atPath: url.appendingPathComponent(name).path) {
            return nil
        }
        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        } catch {
            locatorLogger.error(
                "cannot list model directory \(url.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
            return nil
        }
        guard contents.contains(where: { $0.pathExtension == weightsFileExtension }) else {
            return nil
        }
        return url
    }
}
