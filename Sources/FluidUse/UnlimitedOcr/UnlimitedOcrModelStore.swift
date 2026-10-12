import Foundation

/// Downloads the pinned Unlimited-OCR Core ML bundle (FluidInference/unlimited-ocr-coreml): page encoder (fp16),
/// 8-bit multi-page decoder, embedding table, tokenizer and config, as `UnlimitedOcrManager.load(from:)` expects.
/// About 3.8 GB; files are checksummed once per pinned revision.
public enum UnlimitedOcrModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    public static let repository = "FluidInference/unlimited-ocr-coreml"
    public static let revision = "9921f1f46054870446009ddc3ebd01f23bec0027"
    static let assets: [Asset] = [
        Asset(path: "config.json", sha256: "210d30d4698482e1672cbbb83846d88b5ca2b67333a3fbbbefa056663080945b"),
        Asset(path: "embeddings.f16", sha256: "5d1df8b73bdff79c672f53ddc000451158cd16cd04ed143a1c0054b30ca03455"),
        Asset(path: "LICENSE", sha256: "d985048c6d69429d685fdbe7557340aa0897c0fd8dc038299b148b9c75dc3383"),
        Asset(path: "tokenizer.json", sha256: "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4"),
        Asset(
            path: "UnlimitedOcrDecoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "bbd94029b3343cf3e5fa79d73feffd49f299fa6f445d30c51ef65df4905d37e0"),
        Asset(
            path: "UnlimitedOcrDecoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "9dca216cc86a99767f5837c2b17e132c96001ec21006c12eae45830fbebc468e"),
        Asset(
            path: "UnlimitedOcrDecoder.mlpackage/Manifest.json",
            sha256: "d972e5852f362d203d94eb3df0d2e124a7803c39045c162f94183e76d6cfeefe"),
        Asset(
            path: "UnlimitedOcrPageEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "7dcce3bed4dcf35e3dde7bd2510aa7cbc7b1bd3ca8eff93d859d11456005fc9d"),
        Asset(
            path: "UnlimitedOcrPageEncoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "767a77ddc86f6721dfe1e46fbb44c1e624429dfc3920682e49f68734189e40f9"),
        Asset(
            path: "UnlimitedOcrPageEncoder.mlpackage/Manifest.json",
            sha256: "614ed2ad82f254936583de24a8001d440c39b00828f81343791f9eaa42913d1a"),
    ]

    /// Ensure the bundle exists in the FluidUse cache and return its directory.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("unlimited-ocr-coreml", isDirectory: true)
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
            if manager.fileExists(atPath: destination.path),
                try ClefVisionModelStore.checksum(of: destination) == asset.sha256
            {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw UnlimitedOcrError.download("bad URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw UnlimitedOcrError.download(
                    "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) for \(asset.path)")
            }
            guard try ClefVisionModelStore.checksum(of: temporary) == asset.sha256 else {
                throw UnlimitedOcrError.download("checksum mismatch for \(asset.path)")
            }
            if manager.fileExists(atPath: destination.path) { try manager.removeItem(at: destination) }
            try manager.moveItem(at: temporary, to: destination)
            progress?(
                asset.path, Int64((try? manager.attributesOfItem(atPath: destination.path)[.size] as? Int64) ?? 0))
        }
        manager.createFile(atPath: verified.path, contents: nil)
        return directory
    }
}
