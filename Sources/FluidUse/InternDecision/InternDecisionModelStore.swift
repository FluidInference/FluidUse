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

    static let showdownRevision = "878e1404df1f257e02feef3a91e874fdccdd3780"

    static func package(
        _ bucket: String, name: String = "DecisionRow_fp16", model: String, weights: String, manifest: String,
        config: String
    ) -> [Asset] {
        [
            Asset(path: "\(bucket)/config.json", sha256: config),
            Asset(path: "\(bucket)/\(name).mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: model),
            Asset(path: "\(bucket)/\(name).mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: weights),
            Asset(path: "\(bucket)/\(name).mlpackage/Manifest.json", sha256: manifest),
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

    /// The int8 (weight-only, per-channel) buckets: same speed as fp16, about 0.85 GB in memory with one bucket in
    /// use instead of about 1.5 GB, and no measurable change in play.
    static let showdownAssets: [Asset] =
        [
            Asset(path: "config.json", sha256: "9f41af5eb81d91e736045849e0b5cc65948edd53a7893043ad391fefeddc89f7"),
            Asset(path: "embeddings.f16", sha256: "703d76a6923d2b1fad57d11e9e11dc60b2c2db6b448ab4a1a98aa7e1c6327877"),
            Asset(path: "tokenizer.json", sha256: "94a639c4b33b192cc5a22cd3d7f0aaf6d97efa9577957a0382b55292ca4f0f00"),
        ]
        + package(
            "L512_F8", name: "DecisionRow_w8",
            model: "1c53fecc9fc67d03884a593448074113e1db040b9908afe59e6482d67381b290",
            weights: "4433b38de2fda23446174fb0d011736c195c32dd9a924c2b8c53ad028b50c21a",
            manifest: "171b2dca62328c5358becfa36d43c186bd64678ba4fed2b1d14ff8f21af02da5",
            config: "544bc212531abb4ebd47fbbcb5b6cba3084186115971d84eaf874e736a4490bc")
        + package(
            "L640_F8", name: "DecisionRow_w8",
            model: "2ed49dcf8978cbed9dabc19365cba6507eca638eb03baab0f977a61e39d4b91f",
            weights: "3711e818b4612c7024a543fbc6bf0b22acf43b647e2930b2ce624f70dcb8cfcc",
            manifest: "759db0f58151fbcd5c27fbb80f918bb1ae419e925a04de42fd03e6c82419685f",
            config: "b6301674536dc7ead157ad70b5d7b5990aad7c7f5073379638ae71dc775c6ef9")
        + package(
            "L1024_F16", name: "DecisionRow_w8",
            model: "4ce919d28e3ce2661b3e04164e44cfa039badfdc34c22eba1384fd2bb6df1fce",
            weights: "db8c66f760c48627e7fe4957c82c9b5ff0ea4ee23a6da9f570b138f27dc8e006",
            manifest: "add62157ac7623bc8cde8820c55eafd23322a4fb815f69a451685bb9afee9461",
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
        try Self.removeStalePackages(in: directory, keeping: model.assets)
        try Data().write(to: verified)
        return directory
    }

    /// After a revision change, packages (and their compiled forms) that the pinned snapshot no longer lists must
    /// go, or `InternDecisionManager.load` could pick an older precision left in the same folder.
    static func removeStalePackages(in directory: URL, keeping assets: [Asset]) throws {
        let manager = FileManager.default
        let kept = Set(
            assets.compactMap { asset -> String? in
                guard let range = asset.path.range(of: ".mlpackage/") else { return nil }
                return String(asset.path[..<range.lowerBound])
            })
        guard let files = manager.enumerator(at: directory, includingPropertiesForKeys: nil) else { return }
        var stale: [URL] = []
        for case let url as URL in files where ["mlpackage", "mlmodelc"].contains(url.pathExtension) {
            let relative = url.path.replacingOccurrences(of: directory.path + "/", with: "")
            let stem = String(relative.dropLast(url.pathExtension.count + 1))
            if !kept.contains(stem) { stale.append(url) }
            files.skipDescendants()
        }
        for url in stale { try manager.removeItem(at: url) }
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
