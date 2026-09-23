import CryptoKit
import Foundation

/// Downloads pinned, checksum-verified snapshots of the Python-assisted Core ML repositories and builds
/// their locked Python environments.
///
/// `published-coreml-manifest.json` records every file's size and SHA-256 at one Hub revision per model
/// (regenerate it with `Tools/pin_published_coreml.py`). Only the packages for the requested precision are
/// fetched; the runtime source, tokenizer, lock files, and licenses always are.
public enum PublishedCoreMLModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Manifest: Decodable {
        struct File: Decodable {
            let path: String
            let size: Int64
            let sha256: String
        }

        let repository: String
        let revision: String
        let files: [File]
    }

    static func manifest(for model: PublishedCoreMLModel) throws -> Manifest {
        guard model.isDownloadable else {
            throw PublishedCoreMLError.missingAsset(
                "\(model.rawValue) has no redistributable weights; pass a local conversion directory")
        }
        guard
            let url = Bundle.module.url(
                forResource: "published-coreml-manifest", withExtension: "json", subdirectory: "Resources")
        else { throw PublishedCoreMLError.missingAsset("published-coreml-manifest.json") }
        let manifests = try JSONDecoder().decode([String: Manifest].self, from: Data(contentsOf: url))
        guard let manifest = manifests[model.rawValue], manifest.repository == model.repository else {
            throw PublishedCoreMLError.missingAsset("No pinned manifest for \(model.repository)")
        }
        return manifest
    }

    /// Pinned Hub revision of a downloadable model.
    public static func revision(for model: PublishedCoreMLModel) throws -> String {
        try manifest(for: model).revision
    }

    /// Files `ensure` materializes: everything outside `.mlpackage` directories plus the selected packages.
    static func selectedFiles(
        for model: PublishedCoreMLModel, precision: String
    ) throws -> [Manifest.File] {
        let packages = try model.requiredPackages(precision: precision).map { $0 + "/" }
        let files = try manifest(for: model).files.filter { file in
            guard file.path.contains(".mlpackage/") else { return true }
            return packages.contains { file.path.hasPrefix($0) }
        }
        for package in packages where !files.contains(where: { $0.path.hasPrefix(package) }) {
            throw PublishedCoreMLError.missingAsset("\(package) is not in the pinned \(model.repository) revision")
        }
        return files
    }

    /// Ensure the pinned snapshot for `precision` exists under `cacheDirectory/<repository name>`,
    /// downloading missing or mismatched files. Returns the repository directory.
    public static func ensure(
        model: PublishedCoreMLModel, precision: String? = nil, cacheDirectory: URL? = nil,
        progress: Progress? = nil
    ) async throws -> URL {
        let manifest = try manifest(for: model)
        let files = try selectedFiles(for: model, precision: precision ?? model.precisions[0])
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("\(model.rawValue)-coreml", isDirectory: true)
        let manager = FileManager.default
        for file in files {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(file.path)
            if try matches(destination, file) { continue }
            try manager.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            let escaped = file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            guard
                let url = URL(
                    string: "https://huggingface.co/\(manifest.repository)/resolve/\(manifest.revision)/\(escaped)")
            else { throw PublishedCoreMLError.missingAsset("Invalid download URL for \(file.path)") }
            progress?(file.path, 0)
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw PublishedCoreMLError.missingAsset(
                    "Download of \(file.path) failed (\((response as? HTTPURLResponse)?.statusCode ?? -1))")
            }
            guard try matches(temporary, file) else {
                throw PublishedCoreMLError.missingAsset("Size or checksum mismatch for \(file.path)")
            }
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(file.path, file.size)
        }
        return directory
    }

    static func matches(_ url: URL, _ file: Manifest.File) throws -> Bool {
        guard
            let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber,
            size.int64Value == file.size
        else { return false }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 4_194_304), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined() == file.sha256
    }

    /// Build the repository's locked Python environment with `uv sync --frozen --no-dev` and return its
    /// interpreter. `uv` installs Python 3.12 itself when needed; the first sync downloads PyTorch.
    public static func prepareEnvironment(
        for model: PublishedCoreMLModel, in directory: URL, uv: URL? = nil
    ) async throws -> URL {
        guard let project = model.projectDirectory else {
            throw PublishedCoreMLError.missingAsset("\(model.rawValue) has no published Python project")
        }
        let projectURL = project.isEmpty ? directory : directory.appendingPathComponent(project, isDirectory: true)
        for name in ["pyproject.toml", "uv.lock"] {
            guard FileManager.default.fileExists(atPath: projectURL.appendingPathComponent(name).path) else {
                throw PublishedCoreMLError.missingAsset("\(project.isEmpty ? "" : project + "/")\(name)")
            }
        }
        guard let executable = uv ?? locateUV() else {
            throw PublishedCoreMLError.missingAsset("uv executable (install from https://docs.astral.sh/uv/)")
        }
        let status = try await PublishedProcess.run(
            executable, arguments: ["sync", "--frozen", "--no-dev", "--project", projectURL.path])
        guard status.code == 0 else {
            throw PublishedCoreMLError.runtime("uv sync exited with \(status.code): \(status.standardError)")
        }
        let python = projectURL.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw PublishedCoreMLError.missingAsset(python.path)
        }
        return python
    }

    static func locateUV() -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let path = ProcessInfo.processInfo.environment["PATH"]?.split(separator: ":").map(String.init) ?? []
        let candidates =
            path + ["/opt/homebrew/bin", "/usr/local/bin", "\(home)/.local/bin", "\(home)/.cargo/bin"]
        return candidates.lazy.map { URL(fileURLWithPath: $0).appendingPathComponent("uv") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }
}
