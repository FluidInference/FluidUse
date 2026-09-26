import CryptoKit
import Foundation

/// Downloads the pinned Kev-0.8B Core ML snapshot (FluidInference/kev-0.8b-coreml): the fused multifunction package,
/// the row buckets, the embedding table and the tokenizer, laid out as `KevFastManager.load(from:)` expects.
public enum KevModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    static let repository = "FluidInference/kev-0.8b-coreml"
    static let revision = "8fa7089169d82a1534a723500ffd7106735b6bc1"
    static let assets: [Asset] = [
        Asset(path: "config.json", sha256: "abe42540d7ebef1f3fd1891f37fb2bc79a3c8fb43b76a1a64af2fd274e9cf11c"),
        Asset(path: "fused/config.json", sha256: "fdea999e887837f62d4fa4b07d1a941e0474ef89b9c3442fac8f3441bc6bb96e"),
        Asset(
            path: "fused/KevFused.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "b628b2c64e3c8052c6b1d78b61a56c219dd033751791c9dc65b68dd58331e5b8"),
        Asset(
            path: "fused/KevFused.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "7e39fc13d99f1ff0d79a5b1965978e6640873893c79dafb874609aa9808ee49b"),
        Asset(
            path: "fused/KevFused.mlpackage/Manifest.json",
            sha256: "94ba26f822aa37427a348dfb74bcdfd56ae086906c23104bdd71aeb51c5eb558"),
        Asset(
            path: "L1024_K80/config.json", sha256: "83ca9a1e92c1f5d43cb7731ef2ebcc95520755d2b5a7eb41692eddd3da88e084"),
        Asset(
            path: "L1024_K80/KevRow_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "f72f7558cbf0e59d61621dd73bea866d721bc6a97991ef41b64e71071773e1ea"),
        Asset(
            path: "L1024_K80/KevRow_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "47b57aee6be1bde59e3ad4779897888710e63c425b6aabb8b09c0e1f149273c9"),
        Asset(
            path: "L1024_K80/KevRow_fp16.mlpackage/Manifest.json",
            sha256: "e5d266706c569f7ad1201c06334f17a31158cf83c400d89b1ba2ea661b348b51"),
        Asset(path: "L512_K16/config.json", sha256: "c734c4c44adfb669d226b57d75ec5532f6778c58cce0cacd51ebffdf0569c2b7"),
        Asset(
            path: "L512_K16/embeddings.f16", sha256: "2146dc176cff21240562283cf0b709a91499917d5b5f2c4fb0b7b41b21d407b0"),
        Asset(
            path: "L512_K16/KevRow_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "6a7f904d6fbecc84210ac1399f7aada1573af744c795f6dbc9c5206cb3895757"),
        Asset(
            path: "L512_K16/KevRow_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "d4607666040c82d5b1b3cd2409e1a7aa279acfe56a28ba6090dc8a0270e3e2a2"),
        Asset(
            path: "L512_K16/KevRow_fp16.mlpackage/Manifest.json",
            sha256: "f772d07c1da6795d99333c982e7697815c9420e108a289b11fed06ba2498b1cd"),
        Asset(path: "tokenizer.json", sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    ]

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("kev-0.8b-coreml", isDirectory: true)
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
                throw KevError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw KevError.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw KevError.invalidAsset(
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
