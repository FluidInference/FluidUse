import CryptoKit
import Foundation

/// Downloads the pinned Qwen2.5-Coder-0.5B Core ML snapshot (FluidInference/qwen2.5-coder-0.5b-coreml): the
/// multifunction package, the tokenizer and the host config, laid out as `CodeWriterManager.load(from:)` expects.
public enum CodeWriterModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    public static let repository = "FluidInference/qwen2.5-coder-0.5b-coreml"
    static let revision = "48abc2c5b3ae57f39b0c7b9d884345389b298097"
    static let assets: [Asset] = [
        Asset(path: "config.json", sha256: "36b8b8eba9f2285e72c6560ce1b3f660534d2e3d780c910cd04aadf84871b553"),
        Asset(path: "tokenizer.json", sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
        Asset(
            path: "qwen2_5_coder_0_5b.mlpackage/Manifest.json",
            sha256: "b50d536c73db6ed30cfcaacd3a92bfda794e6b7d501f0e9c119c840b3adcee0e"),
        Asset(
            path: "qwen2_5_coder_0_5b.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "e70fe1ecdc5fb13aa62d31ffd9c9d4f52c9aafa0846d34efdb25758e0be833b8"),
        Asset(
            path: "qwen2_5_coder_0_5b.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "3cca098e86bc97aeca18a7593f32869a937e6d09bcae7ec15ac83e5333d0e11b"),
    ]

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("qwen2.5-coder-0.5b-coreml", isDirectory: true)
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
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw CodeWriterError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw CodeWriterError.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw CodeWriterError.invalidAsset(
                    "Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actual)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(asset.path, size)
        }
        try Data().write(to: verified)
        return directory
    }

    private static func checksum(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 4_194_304), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
