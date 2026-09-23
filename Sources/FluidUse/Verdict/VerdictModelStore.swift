import CryptoKit
import Foundation

/// Downloads and verifies the published Verdict tokenizer, calibrator, and FP16 Core ML buckets.
public enum VerdictModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    private static let repository = "FluidInference/verdict-coreml"
    private static let revision = "835208c443699f1f95c4bed1b177dc916300cffe"

    private struct Asset {
        let path: String
        let sha256: String
    }

    private static let shared: [Asset] = [
        Asset(path: "config.json", sha256: "303f8eef1009cfdcb0cfba3e653247e625f16a4501e2351bad1a633f1f644695"),
        Asset(path: "tokenizer.json", sha256: "8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723"),
        Asset(
            path: "tokenizer_config.json", sha256: "fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016"),
        Asset(path: "calibrator.json", sha256: "af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de"),
    ]

    private static let buckets: [Int: [Asset]] = [
        128: [
            Asset(
                path: "verdict_fp16_L128_candidates25.mlpackage/Manifest.json",
                sha256: "fc5d31df174a9f450bfb9ba406307c379c30fabba5cad38fcaf8717b806c69a8"),
            Asset(
                path: "verdict_fp16_L128_candidates25.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                sha256: "6cc079930a63c1069a7c259f4c086879981021047767f2bb4d68b6d1f02b00"),
            Asset(
                path: "verdict_fp16_L128_candidates25.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                sha256: "a982e14147982df9abb99a3637075a0193835c6b99edc396a52c1a43c2a4854a"),
        ],
        512: [
            Asset(
                path: "verdict_fp16_L512_candidates25.mlpackage/Manifest.json",
                sha256: "047cc80f7d66d822ca0d9c0100a6a6dc96ae2d70eac42e292a2b246e01e64ddd"),
            Asset(
                path: "verdict_fp16_L512_candidates25.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                sha256: "7ad21a9ee93d1c573e6d89e6c35511f08c984a5c47108bb2b908c96344c03903"),
            Asset(
                path: "verdict_fp16_L512_candidates25.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                sha256: "fb438de5823dd91c8448b373d026b7e6dcb7149bf871fde54057fd6d82586886"),
        ],
    ]

    public static func packageName(length: Int) throws -> String {
        guard buckets[length] != nil else { throw VerdictError.invalidAsset("Unsupported length \(length)") }
        return "verdict_fp16_L\(length)_candidates25.mlpackage"
    }

    public static func ensure(
        lengths: [Int] = [128, 512], cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        let directory = (cacheDirectory ?? LayaModelStore.defaultCacheDirectory())
            .appendingPathComponent("verdict-coreml", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var assets = shared
        for length in lengths {
            guard let entries = buckets[length] else { throw VerdictError.invalidAsset("Unsupported length \(length)") }
            assets.append(contentsOf: entries)
        }
        for asset in assets {
            let destination = directory.appendingPathComponent(asset.path)
            if FileManager.default.fileExists(atPath: destination.path), try digest(destination) == asset.sha256 {
                continue
            }
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(asset.path)") else {
                throw VerdictError.invalidAsset("Invalid download URL for \(asset.path)")
            }
            progress?(asset.path, 0)
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? FileManager.default.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw VerdictError.invalidAsset("Download failed for \(asset.path)")
            }
            guard try digest(temporary) == asset.sha256 else {
                throw VerdictError.invalidAsset("Checksum mismatch for \(asset.path)")
            }
            let size =
                (try FileManager.default.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(asset.path, size)
        }
        return directory
    }

    private static func digest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var sha = SHA256()
        while let chunk = try file.read(upToCount: 1_048_576), !chunk.isEmpty { sha.update(data: chunk) }
        return sha.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
