import CryptoKit
import Foundation

/// Downloads the pinned SigLIP 2 Core ML packages from Hugging Face into the FluidUse cache.
public enum SigLIP2ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    public static let repository = "FluidInference/siglip2-base-patch16-256-coreml"
    static let revision = "524a5a7d666f23002853831915cf1cc13734f73f"

    private struct Asset {
        let path: String
        let sha256: String
    }

    private static let assets = [
        Asset(path: "config.json", sha256: "b5d7aaa84399aaa277f9cc00c6c401edc05d535f47f4feb1880e0e4ed2a1a8d4"),
        Asset(
            path: "siglip2-base-patch16-256-image-fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "702fdc0b557984e16e4bbcce4eec3568f08a7459bd3c27c770a487823bda808b"),
        Asset(
            path: "siglip2-base-patch16-256-image-fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "da086438b60ada3566f8c91bd8a632462f28d344186e7f60cb5abd223d2e99a8"),
        Asset(
            path: "siglip2-base-patch16-256-image-fp16.mlpackage/Manifest.json",
            sha256: "964a40aa63c2d201a30f0aeab68aef8abc95c854a34f4cae533051cd99996e72"),
        Asset(
            path: "siglip2-base-patch16-256-text-fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "69fce0f538fbe78c45b953e1cc396fb912d701c6f985ac825bdec92833aca452"),
        Asset(
            path: "siglip2-base-patch16-256-text-fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "9a52dd8222973b6b4cdf55983689bb2fa18a27fcd20b83f8d8fb89de544530b5"),
        Asset(
            path: "siglip2-base-patch16-256-text-fp16.mlpackage/Manifest.json",
            sha256: "a5dfacf23259261d32c5af26c5cfe4c41a44f0584739b2ef10e70c7f11c2a620"),
        Asset(
            path: "tokenizer_config.json", sha256: "9c8a03337138d3b5509e4c032f6863e769b7448750718372c907407d67f6a91b"),
        Asset(path: "tokenizer.json", sha256: "caefd63119539a63be2d55ef3e05023fbb793948c4bda5bc0c366b42a382f903"),
    ]

    /// Ensures the packages, tokenizer, and config exist and match their checksums; returns their directory.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("siglip2-base-patch16-256-coreml")
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for asset in assets {
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw SigLIP2Error.invalidAsset("Invalid Hugging Face asset URL")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw SigLIP2Error.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw SigLIP2Error.invalidAsset(
                    "Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actual)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(asset.path, size)
        }
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
