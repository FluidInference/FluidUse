import CryptoKit
import Foundation

/// Downloads the pinned Granite-Embedding-30M-Sparse Core ML packages from Hugging Face into the FluidUse cache.
public enum EvokeModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    public static let repository = "FluidInference/granite-embedding-30m-sparse-coreml"
    static let revision = "4ef29e090fd5a8480cc0830132d8011ac68dd71b"

    private struct Asset {
        let path: String
        let sha256: String
    }

    private static let shared = [
        Asset(path: "config.json", sha256: "c82d13490aa7919a83915f810cb16f614cb2737121ff98cee7f428ee8c851ea9"),
        Asset(path: "tokenizer.json", sha256: "33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd"),
    ]

    /// Per-length package files. Only the lengths a caller asks for are downloaded.
    private static let packages: [Int: [Asset]] = [
        64: [
            Asset(
                path: "granite-embedding-30m-sparse-L64-fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                sha256: "1910f036d332ab96ad0152fb89634a02de44a956192d8775389c893cb8224ff9"),
            Asset(
                path: "granite-embedding-30m-sparse-L64-fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                sha256: "6dea4b191d4ddeb2804e4474048b36de09ce3ff8a999164ca470796dc27ebe78"),
            Asset(
                path: "granite-embedding-30m-sparse-L64-fp16.mlpackage/Manifest.json",
                sha256: "01754dbd180d03a7c6917991996fc2fb43d730a56c87eb0f3dd3cc1852ec958f"),
        ],
        128: [
            Asset(
                path: "granite-embedding-30m-sparse-L128-fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                sha256: "a251809739863713350bcb4e3756f2a92255f4dc8339c5a8c83a93fe5a5db729"),
            Asset(
                path: "granite-embedding-30m-sparse-L128-fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                sha256: "6aabd73f00e4b11454db6325394802ad0142f487e915fafcebe8cab7b155c213"),
            Asset(
                path: "granite-embedding-30m-sparse-L128-fp16.mlpackage/Manifest.json",
                sha256: "66ea678362041f64ae12660ff875c71da915a65015665896a56fd1db8cd1f2ce"),
        ],
        256: [
            Asset(
                path: "granite-embedding-30m-sparse-L256-fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                sha256: "361b551045501600a849122e38eddfb19d8112a4a8c7973a062a2a27275e105c"),
            Asset(
                path: "granite-embedding-30m-sparse-L256-fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                sha256: "11eaab3a66ac137b8afacb650ce4a59135e82bb8dfaa28c28fdb4bf0a73dbe17"),
            Asset(
                path: "granite-embedding-30m-sparse-L256-fp16.mlpackage/Manifest.json",
                sha256: "9e1a19d9a72ba39b05087e6a6433897da8fbc3d80611cd97ada2771ae86f14ab"),
        ],
        512: [
            Asset(
                path: "granite-embedding-30m-sparse-L512-fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                sha256: "e4f38f1f54c387c803afbbf1313cc71aafdb1744575a8b3eaecaab43061a24cf"),
            Asset(
                path: "granite-embedding-30m-sparse-L512-fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                sha256: "5e34eda532f69c44fa9ff0e2f028fa93640899f6577e8ce740be95cba59325a0"),
            Asset(
                path: "granite-embedding-30m-sparse-L512-fp16.mlpackage/Manifest.json",
                sha256: "7e5868a6895c4b6d3340f40e25576a79c27194fc7f6eb41042e25f158115ecdf"),
        ],
    ]

    public static func packageName(length: Int) -> String { "granite-embedding-30m-sparse-L\(length)-fp16" }

    /// Ensures `config.json`, `tokenizer.json` and the packages for `lengths` exist and match their checksums;
    /// returns their directory.
    public static func ensure(
        lengths: [Int] = [64], cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("granite-embedding-30m-sparse-coreml")
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        var assets = shared
        for length in lengths {
            guard let files = packages[length] else {
                throw EvokeError.invalidAsset("No published package for length \(length)")
            }
            assets += files
        }
        for asset in assets {
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw EvokeError.invalidAsset("Invalid Hugging Face asset URL")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw EvokeError.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw EvokeError.invalidAsset(
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
