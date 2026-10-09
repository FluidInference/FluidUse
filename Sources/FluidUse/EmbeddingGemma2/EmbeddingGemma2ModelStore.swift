@preconcurrency import CoreML
import CryptoKit
import Foundation

/// Downloads the pinned EmbeddingGemma 2 Core ML packages, token table, and tokenizer from Hugging Face into the
/// FluidUse cache, verifies every file, and compiles each package once (`<name>.mlmodelc` beside it). Text and audio
/// are separate bundles, each pinned to the revision that added it, so adding audio never recompiles the text model.
public enum EmbeddingGemma2ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    public static let repository = "FluidInference/embeddinggemma-2-coreml"
    static let revision = "70fcf18f32c8cffc173fea982d5652d0bd1de8ea"
    static let directoryName = "embeddinggemma-2-coreml"

    private struct Asset {
        let path: String
        let sha256: String
    }

    static let audioRevision = "c0fefd0ab4c1140ab35fb582ddbe51b4f93205e1"

    private static let audioAssets = [
        Asset(
            path: "EmbeddingGemma2Audio.mlpackage/Manifest.json",
            sha256: "65c8e9811516e7916a3d742261c7d5a02d4e43c113b437330ce86f98b3da3514"),
        Asset(
            path: "EmbeddingGemma2Audio.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "d40896e5c88d62b02d15cc975f27e254c28a5774a1702e09bf5483071623548d"),
        Asset(
            path: "EmbeddingGemma2Audio.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "41d7b552caabe376e377dcbe1c8c5f9879853bc4f384a161bb1f886f19e84b56"),
    ]

    private static let assets = [
        Asset(path: "config.json", sha256: "4ff891a28a5d55562b55ff3aa4ef362968d378629f9453f337cb70a9e1b45f61"),
        Asset(path: "tokenizer.json", sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
        Asset(path: "embeddings.bf16", sha256: "56ebbdbfd706c827f3fae1d97aa39135c7ddc0f1e6d9b12f8565f84bbfbda877"),
        Asset(
            path: "EmbeddingGemma2Text.mlpackage/Manifest.json",
            sha256: "83eace80b69b46f8415af620ba1b40d7c490d00f7c00ba9dedd2e29da6e9ac19"),
        Asset(
            path: "EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0df9aca02807cf490db6d4f1c68607a8c7f486ad23b376d1968fd3be5eb14677"),
        Asset(
            path: "EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "75b97ab036ca87eae60560c4034a63d604201498dc4b6f6971dd0b1e4558d61f"),
    ]

    /// Ensures the text model, token table and tokenizer exist, match their checksums, and are compiled; returns the
    /// directory.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        try await ensure(
            assets: assets, revision: revision, package: "EmbeddingGemma2Text", verifiedName: ".verified-\(revision)",
            cacheDirectory: cacheDirectory, progress: progress)
    }

    /// Ensures the audio model (`EmbeddingGemma2Audio`) exists, matches its checksums, and is compiled; returns the
    /// directory. The text bundle is needed too (`ensure`): audio tokens are embedded by the text model.
    public static func ensureAudio(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        try await ensure(
            assets: audioAssets, revision: audioRevision, package: "EmbeddingGemma2Audio",
            verifiedName: ".verified-audio-\(audioRevision)", cacheDirectory: cacheDirectory, progress: progress)
    }

    private static func ensure(
        assets: [Asset], revision: String, package: String, verifiedName: String, cacheDirectory: URL?,
        progress: Progress?
    ) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent(directoryName)
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        // Hashing hundreds of MB takes a moment; skip it once this revision has been verified and nothing is missing.
        let verified = directory.appendingPathComponent(verifiedName)
        let compiled = directory.appendingPathComponent("\(package).mlmodelc")
        let stamp = compiled.appendingPathComponent("fluiduse-revision")
        let allPresent = assets.allSatisfy {
            manager.fileExists(atPath: directory.appendingPathComponent($0.path).path)
        }
        if allPresent, manager.fileExists(atPath: verified.path),
            (try? String(contentsOf: stamp, encoding: .utf8)) == revision
        {
            return directory
        }
        try? manager.removeItem(at: verified)
        for asset in assets {
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw EmbeddingGemma2Error.invalidAsset("Invalid Hugging Face asset URL")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw EmbeddingGemma2Error.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw EmbeddingGemma2Error.invalidAsset(
                    "Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actual)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(asset.path, size)
        }
        // Recompile unless the compiled model carries this revision's stamp (a revision bump or an interrupted
        // compile leaves an old or missing stamp). Staged rename, so a reader never sees a half-written bundle.
        if (try? String(contentsOf: stamp, encoding: .utf8)) != revision {
            progress?("\(package).mlmodelc", 0)
            let temporary = try await MLModel.compileModel(at: directory.appendingPathComponent("\(package).mlpackage"))
            try revision.write(
                to: temporary.appendingPathComponent("fluiduse-revision"), atomically: true, encoding: .utf8)
            let staged = directory.appendingPathComponent("\(package).\(UUID().uuidString).mlmodelc")
            try manager.moveItem(at: temporary, to: staged)
            let retired = directory.appendingPathComponent("\(package).\(UUID().uuidString).old")
            if manager.fileExists(atPath: compiled.path) { try manager.moveItem(at: compiled, to: retired) }
            try manager.moveItem(at: staged, to: compiled)
            try? manager.removeItem(at: retired)
        }
        manager.createFile(atPath: verified.path, contents: Data())
        return directory
    }

    private static func checksum(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
