import CryptoKit
import Foundation

/// A published Decision 2.0 Core ML snapshot, laid out as `Decision2Manager.load(from:)` expects.
public enum Decision2Model: String, Sendable, CaseIterable {
    /// Decision-2.0-Kai-0.6B (Qwen3): fastest, 1.1 GB fp16 (FluidInference/decision-2.0-kai-coreml).
    case kai = "decision-2.0-kai-coreml"
    /// Decision-2.0-Eos-0.8B (Qwen3.5 hybrid): 1.4 GB fp16 (FluidInference/decision-2.0-eos-coreml).
    case eos = "decision-2.0-eos-coreml"

    var repository: String { "FluidInference/\(rawValue)" }

    var revision: String {
        switch self {
        case .kai: "73344136ce1d0b68be69ebc5ad2c014ff868ea66"
        case .eos: "7bf832369ecdca988a2376e0212b65b0680163f0"
        }
    }

    var assets: [Decision2ModelStore.Asset] {
        switch self {
        case .kai: Decision2ModelStore.kaiAssets
        case .eos: Decision2ModelStore.eosAssets
        }
    }
}

/// Downloads a pinned, checksummed Decision 2.0 snapshot into the FluidUse cache.
public enum Decision2ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    static func package(_ name: String, model: String, weights: String, manifest: String) -> [Asset] {
        [
            Asset(path: "\(name).mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: model),
            Asset(path: "\(name).mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: weights),
            Asset(path: "\(name).mlpackage/Manifest.json", sha256: manifest),
        ]
    }

    static let kaiAssets: [Asset] =
        [
            Asset(path: "coreml_config.json", sha256: "d24a234e8e471225c5ae7191527d1a2a675132e48ad1c14ca230c823ec90bfcb"),
            Asset(path: "score_bias.json", sha256: "7d3a060fe1823aa7fe759776df3db872408de580f227120fbe08b9eef53dcf6e"),
            Asset(path: "tokenizer.json", sha256: "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
        ]
        + package(
            "Decision2KaiPacked", model: "7c4ee2802024c09d66b66b3d7a50f57e3195fd3c175facfd2902df54825abc77",
            weights: "153390b2b6722768b8150b148f2aa6f5fcc960ede365754903a22ed60f853994",
            manifest: "419d3a2521d3886ce608df0204761de533059261ede0276473804daf46216624")

    static let eosAssets: [Asset] =
        [
            Asset(path: "coreml_config.json", sha256: "99be4d683b0c4163c00e103c0ff024e353ba247e61499ec2743126a181a75acf"),
            Asset(path: "tokenizer.json", sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        ]
        + package(
            "Decision2EosPacked", model: "e8a434580763e03408199536441dcfabbb3e58f479e2ef9e823e69d8b497e443",
            weights: "680030e081b927dc5902d522e0df4f24bdf3a13910c228625d7fd98ac324e939",
            manifest: "8933a45677029671d29f614359c9f9d8f2a4394ac30cfd38ee1b765d880c071b")

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(
        _ model: Decision2Model = .kai, cacheDirectory: URL? = nil, progress: Progress? = nil
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
                throw Decision2Error.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw Decision2Error.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw Decision2Error.invalidAsset(
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
