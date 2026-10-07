import CryptoKit
import Foundation

/// Downloads the pinned, checksummed Vela 2.0 0.3B Core ML snapshot
/// ([FluidInference/vela-2.0-0.3b-coreml](https://huggingface.co/FluidInference/vela-2.0-0.3b-coreml)) into the FluidUse
/// cache, laid out as `Vela2Manager.load(from:)` expects (~610 MB).
public enum Vela2ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    static let repository = "FluidInference/vela-2.0-0.3b-coreml"
    static let revision = "fb68b856fb921fc91e74bda9ff83960b0c1d70af"
    static let folder = "vela-2.0-0.3b-coreml"

    static let assets: [(path: String, sha256: String)] = [
        ("coreml_config.json", "7e27fbb8d7f7ff1cd26281ced657756d8c57ff2587816b708aa85f0828f812af"),
        ("calibration.json", "f8557b3922e9bef33e688ece12da20436b5ae868ed7ff1375b346bd26403506a"),
        ("tokenizer.json", "e977adcc6faffc532cb774decca6a9e4f44ad6075930d625dee7cb1ac3aa6121"),
        ("heads.bin", "633f1be5f42471529525538fdefbd7ac6adde0039eadc6745018f9b272acccc5"),
        ("heads.json", "88f7aa50500a6c1e334848909f2d17dcc24835220cadaa97d5fd31e11a33eb85"),
        ("Vela2Encoder.mlpackage/Manifest.json", "2f5ccd0863731722bf402b1f7b8d811877de0a03929112f733457a004afbed26"),
        ("Vela2Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", "e2c94576294101637cfa7d32f45a68575dea08c486c0ec62a0fe7139d5eaeabe"),
        ("Vela2Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "04825523ebbf8cc0f372198f9468c6315df3f9b656ea681cd37f2da1e9e1ca9d"),
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
                throw Vela2Error.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw Vela2Error.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw Vela2Error.invalidAsset("Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actual)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            if asset.path.hasPrefix("Vela2Encoder.mlpackage/") {  // a stale compiled model must not outlive its package
                try? manager.removeItem(at: directory.appendingPathComponent("Vela2Encoder.mlmodelc"))
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
