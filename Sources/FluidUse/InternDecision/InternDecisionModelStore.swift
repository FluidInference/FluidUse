import CryptoKit
import Foundation

/// A published Intern-Decision-0.8B Core ML snapshot: the fp16 buckets, the embedding table and the tokenizer, laid
/// out as `InternDecisionManager.load(from:)` expects.
public enum InternDecisionModel: String, Sendable, CaseIterable {
    /// The stock checkpoint (FluidInference/intern-decision-0.8b-coreml): general typed decisions.
    case stock = "intern-decision-0.8b-coreml"
    /// Fine-tuned for Pokémon Showdown battle actions, distilled from Intern-Decision-4B
    /// (FluidInference/intern-decision-0.8b-showdown-coreml). Still answers any typed question.
    case showdown = "intern-decision-0.8b-showdown-coreml"

    var repository: String { "FluidInference/\(rawValue)" }

    var revision: String {
        switch self {
        case .stock: "08239aad89500d02d91fdf34e38bf50777b4866b"
        case .showdown: InternDecisionModelStore.showdownRevision
        }
    }

    var assets: [InternDecisionModelStore.Asset] {
        switch self {
        case .stock: InternDecisionModelStore.stockAssets
        case .showdown: InternDecisionModelStore.showdownAssets
        }
    }
}

/// Downloads a pinned, checksummed Intern-Decision snapshot into the FluidUse cache.
public enum InternDecisionModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    static let showdownRevision = "636dee2eacf95a18077e5c798245cc658d0f8747"

    static func package(_ bucket: String, model: String, weights: String, manifest: String, config: String) -> [Asset] {
        [
            Asset(path: "\(bucket)/config.json", sha256: config),
            Asset(path: "\(bucket)/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: model),
            Asset(
                path: "\(bucket)/DecisionRow_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: weights),
            Asset(path: "\(bucket)/DecisionRow_fp16.mlpackage/Manifest.json", sha256: manifest),
        ]
    }

    static let stockAssets: [Asset] =
        [
            Asset(path: "config.json", sha256: "78c857bc95d240e5972cf2a1483bad65fecc2aece0de40935b56564e6232c964"),
            Asset(path: "embeddings.f16", sha256: "703d76a6923d2b1fad57d11e9e11dc60b2c2db6b448ab4a1a98aa7e1c6327877"),
            Asset(path: "tokenizer.json", sha256: "94a639c4b33b192cc5a22cd3d7f0aaf6d97efa9577957a0382b55292ca4f0f00"),
        ]
        + package(
            "L320_F8", model: "db276e362f6ae1ae2981ece330d41212c4d61fa002b9ba0252ecfb5edf12ce1e",
            weights: "833d24e2eae57ca5bb2b1ce69c462856a17113b59a2afe914b179d876b1abf5b",
            manifest: "bcb003a7871567aeb57d9d7434bed2f0db4496c4de85ab6541c67e9c28f79a0d",
            config: "6677edb11461dd4a2016a5233e010c64f62209be0e4195f01e5f89cc90c235b2")
        + package(
            "L512_F8", model: "9901131f16db1af933e57d6709dbae92b33a19b4c8ac45495861e39d46bf92e2",
            weights: "59bae3abb5ae32978590e2af65dfae91d02f05e4485222090afe98ffa37aeaf1",
            manifest: "57c1c7ddc79b126c4833d80b098f161854d9b28eb53dd61d2de7b186c7f77849",
            config: "8485208f223b85fe93f8a8a208241b3e64a4b529511cf2e98c9faf1c01377c57")
        + package(
            "L1024_F16", model: "6efb3c4dcf5fe3de41e68a5b4e95318128a8b8bfc34f3454a893c03899c6f6ee",
            weights: "8b966e5fd62b0dd64ed897dc5dcc0ca997ef2d07744aba145fbebc0f0c6ce7cf",
            manifest: "363c3adc8aea3a5a70679e78f062ffa5b8651e14c9ec759d556fc7976f3bfd35",
            config: "2fa46a5ee9844b89eee0f43c04206d544540a4fe7113cd74229266b546030b69")

    static let showdownAssets: [Asset] =
        [
            Asset(path: "config.json", sha256: "9f41af5eb81d91e736045849e0b5cc65948edd53a7893043ad391fefeddc89f7"),
            Asset(path: "embeddings.f16", sha256: "703d76a6923d2b1fad57d11e9e11dc60b2c2db6b448ab4a1a98aa7e1c6327877"),
            Asset(path: "tokenizer.json", sha256: "94a639c4b33b192cc5a22cd3d7f0aaf6d97efa9577957a0382b55292ca4f0f00"),
        ]
        + package(
            "L512_F8", model: "e090f2632d7c6c944c1f102591a81a4836c7bf3a55a3b925c652cbac180e8968",
            weights: "c0dff14d537bff0193e81f61da39e6edcb8a6ded4d6c7cf6f48c9a5128860875",
            manifest: "8eaf456ff4f5ff1916f93354ac64521fd7e2fb0a0aed1a320cec799afdde1de2",
            config: "544bc212531abb4ebd47fbbcb5b6cba3084186115971d84eaf874e736a4490bc")
        + package(
            "L640_F8", model: "29f4535b956e1999aef623a02f73ba42f7a320f1ca33ae30446895b160f0afb9",
            weights: "3aebedd7762bd1230f491dfe8d1fbd8d2c77785cdac7836f907f36e721bce65b",
            manifest: "ea6d9f587d2d1d81b28e20cdf3ad1fa0b47ae1f14bc0053dac0dabeea3f6e8d3",
            config: "b6301674536dc7ead157ad70b5d7b5990aad7c7f5073379638ae71dc775c6ef9")
        + package(
            "L1024_F16", model: "1129b6bd92a4d4e368968c81b26d699ce51fb9f39e0844bb3790a239fab7f133",
            weights: "2fbffc2bc4be9b6d965e2a1ca55bf64e49e603f74470b29f12c3b1589dedad80",
            manifest: "3e730b08ddaffa138a77647724cecc6895f444f76c4ae36328c719f28d52aad4",
            config: "b0b3a036af825b3457e4a0f5225648021def8513d44d7a77b983ab55add6bc2f")

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(
        _ model: InternDecisionModel = .stock, cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent(model.rawValue, isDirectory: true)
        let manager = FileManager.default
        let verified = directory.appendingPathComponent(".verified-\(model.revision)")
        if manager.fileExists(atPath: verified.path),
            model.assets.allSatisfy({ manager.fileExists(atPath: directory.appendingPathComponent($0.path).path) })
        {
            return directory
        }
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for asset in model.assets {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard
                let url = URL(string: "https://huggingface.co/\(model.repository)/resolve/\(model.revision)/\(escaped)")
            else {
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
            // A package file changed: the .mlmodelc compiled from the old package beside it must not outlive it.
            if let range = asset.path.range(of: ".mlpackage/") {
                let compiled = directory.appendingPathComponent(String(asset.path[..<range.lowerBound]) + ".mlmodelc")
                try? manager.removeItem(at: compiled)
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
        while let chunk = try handle.read(upToCount: 4_194_304), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
