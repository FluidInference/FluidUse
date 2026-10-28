import CryptoKit
import Foundation

/// Downloads the pinned, checksummed Vela 3.0 Core ML snapshot
/// ([FluidInference/vela-3.0-coreml](https://huggingface.co/FluidInference/vela-3.0-coreml)) into the FluidUse cache,
/// laid out as `Vela3Manager.load(from:)` expects (~1.48 GB: the decision and span-sets encoders, heads, tokenizer).
public enum Vela3ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    static let repository = "FluidInference/vela-3.0-coreml"
    static let revision = "deaf348c702054ad27189b4d5eebfb66314f3b5b"
    static let folder = "vela-3.0-coreml"

    static let assets: [(path: String, sha256: String)] = [
        ("config.json", "a85b358ac11b71d780deb09e8a082efddc67938a2eb4fc16d68d8ae888c69d14"),
        ("vela3_config.json", "ee5088605aa5151210fbc12e3f6ee0c9a382dddf90d3a43ac606c0df305d9ff2"),
        ("tokenizer.json", "5c7d325b4759ac77f8b95a70d8fb885bb39b66749e728bd745ec67984754b244"),
        ("heads_decision.safetensors", "402dffef5218b64d55e11372db414046219e0d6dccae93193e92cd6e8a0635d5"),
        ("heads_span_sets.safetensors", "71bac50e691185155ba9d30a08c4f70bb07bbce37a6caee0f7bb922859a83e92"),
        ("Vela3Decision.mlpackage/Manifest.json", "71d8d30fd441df570149bca488634b5a1d22caf314643e4533000b4f3c6c8d30"),
        ("Vela3Decision.mlpackage/Data/com.apple.CoreML/model.mlmodel", "3983ec2a96ba8403cf48feaefc9f16cf73645d80e25ceb054718c4ee7f44fdb4"),
        ("Vela3Decision.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "af7431750cca9421b75d9ef1420a29b7ce4e6cdbe61e9029c81f8129ca344eff"),
        ("Vela3SpanSets.mlpackage/Manifest.json", "5b5298314d69ced1e8e832eed230675c101a2f37bdd8649a015765ef03985431"),
        ("Vela3SpanSets.mlpackage/Data/com.apple.CoreML/model.mlmodel", "3e8d0a64e7c85378c23e1e12e3eb2db5f9e6dad50a4172cb57966f6a6657be05"),
        ("Vela3SpanSets.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "2c3a719b4df3d682f399222484a7daa756954cdbca0c4a74c9943dcfb14c544a"),
    ]

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent(folder, isDirectory: true)
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
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 { continue }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw Vela3Error.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw Vela3Error.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw Vela3Error.invalidAsset("Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actual)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            for package in ["Vela3Decision", "Vela3SpanSets"] where asset.path.hasPrefix("\(package).mlpackage/") {
                try? manager.removeItem(at: directory.appendingPathComponent("\(package).mlmodelc"))  // no stale compiled model
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
        while let chunk = try handle.read(upToCount: 4_194_304), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
