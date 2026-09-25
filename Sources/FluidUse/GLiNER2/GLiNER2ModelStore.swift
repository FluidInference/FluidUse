import CryptoKit
import Foundation

/// Downloads the pinned GLiNER 2.5 classification packages from Hugging Face.
public enum GLiNER2ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    private struct Asset {
        let path: String
        let sha256: String
    }

    private static func revision(for variant: GLiNER2Variant) -> String {
        switch variant {
        case .small: "9dcac8a315ca49e71412cf5dec2ee7b7609b614e"
        case .base: "c1843f2c193b11b05f09ac7f258cb9202d8f5e71"
        case .multilingual: "5dab512eb89b88a3680bd6c86841877c3ea49893"
        case .decide, .decideLong: "cd0d7b1ef32b10e1e3a5a73c9d9ac8411d819c5a"
        }
    }

    private static func assets(for variant: GLiNER2Variant) -> [Asset] {
        let package = variant.packageName
        switch variant {
        case .small:
            return [
                Asset(path: "config.json", sha256: "0b7d9e1401ceeb83e992ec66d2f93bff7e5646428f1b4706ec527cf88f53578a"),
                Asset(
                    path: "tokenizer/tokenizer.json",
                    sha256: "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3"),
                Asset(
                    path: "tokenizer/tokenizer_config.json",
                    sha256: "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1"),
                Asset(
                    path: "\(package)/Manifest.json",
                    sha256: "c3036b354640b9ec5cb8e899706a5e1238ab55916ee6733ca8cfcb09aae754d7"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/model.mlmodel",
                    sha256: "b56af2cdc4863696b205caba6b1aea4efcee70e5040c31daf09ec8f8eaadc6ab"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/weights/weight.bin",
                    sha256: "0bb3f77dff1b85fac4ad8d1ea309eac23d30a626ad08a5f20fbe064ec12d9189"),
            ]
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
        case .decide:
            return [
                Asset(path: "config.json", sha256: "e748e5b80575471c91b3f0dd00f513ba58242544fcb7e1236e0021e61abd7673"),
                Asset(
                    path: "encoder_config/config.json",
                    sha256: "bd32f1484ba5a199f7a63df44df3814b839fffcf6e64478323c4689868ef6015"),
                Asset(
                    path: "tokenizer.json", sha256: "3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8"),
                Asset(
                    path: "tokenizer_config.json",
                    sha256: "323199a4e946039410899f3779f2aa3eaef1500213c512727ad0f623d4f21309"),
                Asset(
                    path: "\(package)/Manifest.json",
                    sha256: "6d62c3f4c7331d836cebf29c541815e0c6d7da9e4612ee45ca726847acbf6ed8"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/model.mlmodel",
                    sha256: "b9d4d8a497ea986b5d3163259694e8fc2571c9e7e2cdca2bf1b568bab8111c2a"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/weights/weight.bin",
                    sha256: "54501158f56baf0ebe295d99ac251eb0204cea2594881062b731bef9154032e2"),
            ]
        case .decideLong:
            return [
                Asset(path: "config.json", sha256: "e748e5b80575471c91b3f0dd00f513ba58242544fcb7e1236e0021e61abd7673"),
                Asset(
                    path: "encoder_config/config.json",
                    sha256: "bd32f1484ba5a199f7a63df44df3814b839fffcf6e64478323c4689868ef6015"),
                Asset(
                    path: "tokenizer.json", sha256: "3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8"),
                Asset(
                    path: "tokenizer_config.json",
                    sha256: "323199a4e946039410899f3779f2aa3eaef1500213c512727ad0f623d4f21309"),
                Asset(
                    path: "\(package)/Manifest.json",
                    sha256: "542249fd40bfd27f7da11b303dda14932e8027be4279aab5e65d1c9abcc730c2"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/model.mlmodel",
                    sha256: "626d74aac1448e0666e5a80d79302139dfd84d5ca80c44ec213c1c6a5497d823"),
                Asset(
                    path: "\(package)/Data/com.apple.CoreML/weights/weight.bin",
                    sha256: "54501158f56baf0ebe295d99ac251eb0204cea2594881062b731bef9154032e2"),
            ]
        }
    }

    /// Ensure one variant's tokenizer and Core ML package exist in the FluidUse cache.
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
