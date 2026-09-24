import Foundation

/// Pinned, checksum-verified download of `FluidInference/cua-s1-4b-coreml`.
///
/// `Resources/cua-s1-4b-manifest.json` records every file's size and SHA-256 at one Hub revision
/// (regenerate with `Tools/pin_cua_s1_4b.py`). Only the files one configuration needs are fetched:
/// the shared tokenizer and embedding table, the requested decoder bucket(s), and for multimodal the
/// vision tower. Files land in `~/Library/Application Support/FluidUse/Models/cua-s1-4b-coreml`.
public enum CuaS1FourBModelStore {
    public static let repository = "FluidInference/cua-s1-4b-coreml"
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Manifest: Decodable {
        let repository: String
        let revision: String
        let files: [PublishedCoreMLModelStore.Manifest.File]
    }

    static func manifest() throws -> Manifest {
        guard
            let url = Bundle.module.url(
                forResource: "cua-s1-4b-manifest", withExtension: "json", subdirectory: "Resources")
        else { throw CuaS1FourBError.invalidAsset("cua-s1-4b-manifest.json is not bundled") }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        guard manifest.repository == repository else {
            throw CuaS1FourBError.invalidAsset("manifest pins \(manifest.repository), expected \(repository)")
        }
        return manifest
    }

    /// Path prefixes one configuration needs.
    static func requiredPrefixes(_ configuration: CuaS1FourBManager.Configuration) -> [String] {
        var prefixes = ["tokenizer.json", "embeddings.f16", "LICENSE", "NOTICE"]
        let suffix = configuration.variant.isEmpty ? "" : "-\(configuration.variant)"
        for length in configuration.lengths {
            prefixes.append("\(configuration.modality.rawValue)/L\(length)\(suffix)/")
        }
        if configuration.modality == .multimodal { prefixes.append("multimodal/vision/") }
        return prefixes
    }

    /// Ensure the files for `configuration` exist under `cacheDirectory/cua-s1-4b-coreml`, downloading
    /// missing or mismatched ones. Returns the repository directory.
    public static func ensure(
        configuration: CuaS1FourBManager.Configuration, cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        let manifest = try manifest()
        let prefixes = requiredPrefixes(configuration)
        let files = manifest.files.filter { file in prefixes.contains { file.path.hasPrefix($0) } }
        for prefix in prefixes where prefix.hasSuffix("/") && !files.contains(where: { $0.path.hasPrefix(prefix) }) {
            throw CuaS1FourBError.invalidAsset("\(prefix) is not in the pinned \(repository) revision")
        }
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("cua-s1-4b-coreml", isDirectory: true)
        let manager = FileManager.default
        for file in files {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(file.path)
            if try PublishedCoreMLModelStore.matches(destination, file) { continue }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            let escaped = file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            guard
                let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(manifest.revision)/\(escaped)")
            else { throw CuaS1FourBError.invalidAsset("invalid download URL for \(file.path)") }
            progress?(file.path, 0)
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw CuaS1FourBError.invalidAsset(
                    "download of \(file.path) failed (\((response as? HTTPURLResponse)?.statusCode ?? -1))")
            }
            guard try PublishedCoreMLModelStore.matches(temporary, file) else {
                throw CuaS1FourBError.invalidAsset("size or checksum mismatch for \(file.path)")
            }
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(file.path, file.size)
        }
        return directory
    }
}

extension CuaS1FourBManager {
    /// Download (pinned, verified) and load one configuration from the FluidUse model cache.
    public static func load(
        configuration: Configuration = Configuration(), cacheDirectory: URL? = nil,
        progress: CuaS1FourBModelStore.Progress? = nil
    ) async throws -> CuaS1FourBManager {
        let directory = try await CuaS1FourBModelStore.ensure(
            configuration: configuration, cacheDirectory: cacheDirectory, progress: progress)
        return try await load(from: directory, configuration: configuration)
    }
}
