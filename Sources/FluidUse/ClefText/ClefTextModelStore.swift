import CryptoKit
import Foundation

/// Downloads the pinned clef-text-0.6b Core ML bundle (FluidInference/clef-text-0.6b-coreml): decoder (a function per
/// bucket), joint schema head, embedding table, tokenizer and manifest, as `ClefTextManager.load(from:)` expects.
/// About 1.3 GB; files are checksummed once per pinned revision.
public enum ClefTextModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    public static let repository = "FluidInference/clef-text-0.6b-coreml"
    public static let revision = "3c61a60954478ef385aa42d67d5764642dcc9135"
    static let assets: [Asset] = [
        Asset(path: "config.json", sha256: "af356b87b840722b77068df83612d6ad6efdd19804c6c955e4689a0e10c6bf58"),
        Asset(
            path: "Decoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "3e665021d57e3e7ff05396574a592ab95aea98a72a7744401336b641e796cee7"),
        Asset(
            path: "Decoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "cbe7a6f22befa4edee09e7c806b2b0b9db8533d1ede737540fa82a0261435742"),
        Asset(
            path: "Decoder.mlpackage/Manifest.json",
            sha256: "3f54235a774d5891b2199e6d55e6a1de4a2f7aef3ce8b48a22d1904c66a3dffd"),
        Asset(path: "embeddings.f16", sha256: "616f9374ddbaae3a033a0174557d502719154882e21e5d975ab5b58e0244459d"),
        Asset(
            path: "Head.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "02e92c4f2cd13b5217e6402934c1fbe20a7c8afac19a79d3d02ed0c6279cd7c3"),
        Asset(
            path: "Head.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "d028526027d6ce12313fbc80ad35f148ecb815f09f83695c5bb2b5872dbd4657"),
        Asset(
            path: "Head.mlpackage/Manifest.json",
            sha256: "9eac1a73bb7772ea2ff3cb5167339bd0090ad6a819e3a3671fa8a80ef90f2ad2"),
        Asset(path: "tokenizer.json", sha256: "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
    ]

    /// Ensure the bundle exists in the FluidUse cache and return its directory.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("clef-text-0.6b-coreml", isDirectory: true)
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
            if manager.fileExists(atPath: destination.path),
                try ClefVisionModelStore.checksum(of: destination) == asset.sha256
            {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw ClefVisionError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw ClefVisionError.download(
                    "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) for \(asset.path)")
            }
            guard try ClefVisionModelStore.checksum(of: temporary) == asset.sha256 else {
                throw ClefVisionError.checksumMismatch(asset.path)
            }
            if manager.fileExists(atPath: destination.path) { try manager.removeItem(at: destination) }
            try manager.moveItem(at: temporary, to: destination)
            progress?(
                asset.path, Int64((try? manager.attributesOfItem(atPath: destination.path)[.size] as? Int64) ?? 0))
        }
        manager.createFile(atPath: verified.path, contents: nil)
        return directory
    }
}
