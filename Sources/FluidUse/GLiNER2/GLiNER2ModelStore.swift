import CryptoKit
import Foundation

/// Downloads the pinned, eight-bit GLiNER 2.5 classification packages from Hugging Face.
public enum GLiNER2ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    private struct Asset {
        let path: String
        let sha256: String
    }

    private static func revision(for variant: GLiNER2Variant) -> String {
        switch variant {
        case .base: "c1843f2c193b11b05f09ac7f258cb9202d8f5e71"
        case .multilingual: "5dab512eb89b88a3680bd6c86841877c3ea49893"
        }
    }

    private static func assets(for variant: GLiNER2Variant) -> [Asset] {
        let package = variant.packageName
        switch variant {
        case .base:
            return [
                Asset(
                    path: "config.json",
                    sha256: "0eb92d00584d613aab32b2178f84a85176b62c87ae3689ce9084e83f6eba64d1"),
                Asset(
                    path: "encoder_config/config.json",
                    sha256: "d36a845b9f25dcaf1ec45a1c4bdf65ea4ac20596537e14530ec9f660a63aeca4"),
                Asset(
                    path: "tokenizer/tokenizer.json",
                    sha256: "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3"),
                Asset(
                    path: "tokenizer/tokenizer_config.json",
                    sha256: "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1"),
                Asset(
                    path: "\(package)/Manifest.json",
                    sha256: "b0e9c35bf2fe7a2d87f62cc6244cf8f39dd7968d9a3f6b6d289b803eaeaddede"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/model.mlmodel",
                    sha256: "11c56d5d686afb3812181024125c8102756d616230606e849a0c97ebd9b34dd9"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/weights/weight.bin",
                    sha256: "037f91cb176d31a38bc1795272d99c47531f3ac235b6f4533bab0df9ae4de327"),
            ]
        case .multilingual:
            return [
                Asset(
                    path: "config.json",
                    sha256: "8b59a0f426a65859c89cd1ea850c3529c09aa3be3a6fafd8eddfdd17b1bf0146"),
                Asset(
                    path: "encoder_config/config.json",
                    sha256: "fa4f9ef2903b5369ab172333aae4574e6a476511d7465845cf59f8360ee18716"),
                Asset(
                    path: "tokenizer/tokenizer.json",
                    sha256: "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a"),
                Asset(
                    path: "tokenizer/tokenizer_config.json",
                    sha256: "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1"),
                Asset(
                    path: "\(package)/Manifest.json",
                    sha256: "51c7a741b26b804175983bde9bc85b5572864ef40ad5986a75fafba45342c195"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/model.mlmodel",
                    sha256: "ca9a69ef5e8e39a4a6f5f82033f4a1266a5be6c0254f57283caef76a250c9d82"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/weights/weight.bin",
                    sha256: "655460ce8e9420b55130f3b6b976d4797aa197cd14c909aee96967cea24d766e"),
            ]
        }
    }

    /// Ensure one variant's tokenizer and W8 Core ML package exist in the FluidUse cache.
    public static func ensure(
        variant: GLiNER2Variant, cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent(
            variant.repository.split(separator: "/").last.map(String.init) ?? variant.rawValue)
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for asset in assets(for: variant) {
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let url = try downloadURL(variant: variant, path: asset.path)
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw GLiNER2Error.invalidAsset("Download failed for \(asset.path)")
            }
            let actualChecksum = try checksum(of: temporary)
            guard actualChecksum == asset.sha256 else {
                throw GLiNER2Error.invalidAsset(
                    "Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actualChecksum)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(asset.path, size)
        }
        return directory
    }

    private static func downloadURL(variant: GLiNER2Variant, path: String) throws -> URL {
        let escaped = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
        guard
            let url = URL(
                string: "https://huggingface.co/\(variant.repository)/resolve/\(revision(for: variant))/\(escaped)")
        else {
            throw GLiNER2Error.invalidAsset("Invalid Hugging Face asset URL")
        }
        return url
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
