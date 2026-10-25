import CryptoKit
import Foundation

/// Downloads the pinned, checksummed d1-omni-600M Core ML snapshot
/// ([FluidInference/d1-omni-600m-coreml](https://huggingface.co/FluidInference/d1-omni-600m-coreml)) into the
/// FluidUse cache, laid out as `D1OmniManager.load(from:)` expects (~735 MB).
public enum D1OmniModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    static let repository = "FluidInference/d1-omni-600m-coreml"
    static let revision = "8ae382b1f2e5b9b7e33008515a40df4ecdfcebdb"
    static let folder = "d1-omni-600m-coreml"

    static let assets: [(path: String, sha256: String)] = [
        ("config.json", "dfccae241822f81752426629c88c8b8d7efdd03dd19418b6645c604fad43d619"),
        ("tokenizer.json", "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711"),
        ("d1-omni-text.mlpackage/Manifest.json", "9f86e7c8f5a1e65d3656d050be5ba0c5d4842eaaba3de255029522ee82c3acab"),
        (
            "d1-omni-text.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            "dcb6ee0574310a9d308a46722b201e28b18ff69acdcc21d19143f1cdc5dab8cf"
        ),
        (
            "d1-omni-text.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            "488fa28c27d64363f3c94ba6a99a2e34987b572785af64072d5045b84e80bbf9"
        ),
    ]

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent(folder, isDirectory: true)
        let manager = FileManager.default
        let verified = directory.appendingPathComponent(".verified-\(revision)")
        if manager.fileExists(atPath: verified.path),
            assets.allSatisfy({ manager.fileExists(atPath: directory.appendingPathComponent($0.path).path) })
        {
            return directory
        }
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for asset in assets {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 { continue }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw D1OmniError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw D1OmniError.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw D1OmniError.invalidAsset(
                    "Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actual)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            if asset.path.hasPrefix("d1-omni-text.mlpackage/") {  // a stale compiled model must not outlive its package
                try? manager.removeItem(at: directory.appendingPathComponent("d1-omni-text.mlmodelc"))
            }
            progress?(asset.path, size)
        }
        try Data().write(to: verified)
        return directory
    }

    private static func checksum(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 4_194_304), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
