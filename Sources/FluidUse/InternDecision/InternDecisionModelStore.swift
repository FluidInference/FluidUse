import CryptoKit
import Foundation

/// Downloads the pinned Intern-Decision-0.8B Core ML snapshot (FluidInference/intern-decision-0.8b-coreml): the
/// fp16 buckets, the embedding table and the tokenizer, laid out as `InternDecisionManager.load(from:)` expects.
public enum InternDecisionModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    static let repository = "FluidInference/intern-decision-0.8b-coreml"
    static let revision = "08239aad89500d02d91fdf34e38bf50777b4866b"
    static let assets: [Asset] = [
        Asset(path: "config.json", sha256: "78c857bc95d240e5972cf2a1483bad65fecc2aece0de40935b56564e6232c964"),
        Asset(path: "embeddings.f16", sha256: "703d76a6923d2b1fad57d11e9e11dc60b2c2db6b448ab4a1a98aa7e1c6327877"),
        Asset(path: "tokenizer.json", sha256: "94a639c4b33b192cc5a22cd3d7f0aaf6d97efa9577957a0382b55292ca4f0f00"),
        Asset(path: "L320_F8/config.json", sha256: "6677edb11461dd4a2016a5233e010c64f62209be0e4195f01e5f89cc90c235b2"),
        Asset(
            path: "L320_F8/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "db276e362f6ae1ae2981ece330d41212c4d61fa002b9ba0252ecfb5edf12ce1e"),
        Asset(
            path: "L320_F8/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "833d24e2eae57ca5bb2b1ce69c462856a17113b59a2afe914b179d876b1abf5b"),
        Asset(
            path: "L320_F8/DecisionRow_fp16.mlpackage/Manifest.json",
            sha256: "bcb003a7871567aeb57d9d7434bed2f0db4496c4de85ab6541c67e9c28f79a0d"),
        Asset(path: "L512_F8/config.json", sha256: "8485208f223b85fe93f8a8a208241b3e64a4b529511cf2e98c9faf1c01377c57"),
        Asset(
            path: "L512_F8/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "9901131f16db1af933e57d6709dbae92b33a19b4c8ac45495861e39d46bf92e2"),
        Asset(
            path: "L512_F8/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "59bae3abb5ae32978590e2af65dfae91d02f05e4485222090afe98ffa37aeaf1"),
        Asset(
            path: "L512_F8/DecisionRow_fp16.mlpackage/Manifest.json",
            sha256: "57c1c7ddc79b126c4833d80b098f161854d9b28eb53dd61d2de7b186c7f77849"),
        Asset(
            path: "L1024_F16/config.json", sha256: "2fa46a5ee9844b89eee0f43c04206d544540a4fe7113cd74229266b546030b69"),
        Asset(
            path: "L1024_F16/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "6efb3c4dcf5fe3de41e68a5b4e95318128a8b8bfc34f3454a893c03899c6f6ee"),
        Asset(
            path: "L1024_F16/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "8b966e5fd62b0dd64ed897dc5dcc0ca997ef2d07744aba145fbebc0f0c6ce7cf"),
        Asset(
            path: "L1024_F16/DecisionRow_fp16.mlpackage/Manifest.json",
            sha256: "363c3adc8aea3a5a70679e78f062ffa5b8651e14c9ec759d556fc7976f3bfd35"),
    ]

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("intern-decision-0.8b-coreml", isDirectory: true)
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
                throw InternDecisionError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw InternDecisionError.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw InternDecisionError.invalidAsset(
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
