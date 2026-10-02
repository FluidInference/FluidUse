import CryptoKit
import Foundation

/// Downloads the pinned clef-vision-0.8b Core ML bundle (FluidInference/clef-vision-0.8b-coreml): vision tower, LM row
/// buckets, head buckets, embedding table, vision position table and tokenizer, as `ClefVisionManager.load(from:)` expects.
public enum ClefVisionModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    public static let repository = "FluidInference/clef-vision-0.8b-coreml"
    public static let revision = "9ccaa421be71ab5abb28421e25bba3d962780113"
    static let assets: [Asset] = [
        Asset(path: "Head_L1024_Q16_O64/Head_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: "c239faf7a20a49bf69464ab360217b9fc3ad9c2348acb480368ab79b0dfeeaaa"),
        Asset(path: "Head_L1024_Q16_O64/Head_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: "a09ada271a4b9d1ecebfddaa4ad431b17a411d3d6c6abdd8b896c20f1a02a433"),
        Asset(path: "Head_L1024_Q16_O64/Head_fp32.mlpackage/Manifest.json", sha256: "e6d7995a29ba210ba4d8cbcfa80f8373632f75458e1d5e080e908b08887b6274"),
        Asset(path: "Head_L1024_Q16_O64/config.json", sha256: "18b6c83243bf48b57883ff86ad8e8a6b4ce88ddd997983193e89d6dcb1596c3a"),
        Asset(path: "Head_L2048_Q16_O64/Head_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: "eafd4d6348a68d5d771fffa81a8d43efba430cf9f718caa2025f4b46f4cc31c5"),
        Asset(path: "Head_L2048_Q16_O64/Head_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: "a09ada271a4b9d1ecebfddaa4ad431b17a411d3d6c6abdd8b896c20f1a02a433"),
        Asset(path: "Head_L2048_Q16_O64/Head_fp32.mlpackage/Manifest.json", sha256: "959417d3f2ea27428060a29c6f6293a8c48c6e5af61b46f6d04402bdae82c464"),
        Asset(path: "Head_L2048_Q16_O64/config.json", sha256: "f84cef8d439b2f764e7a2029e9b5fbd4038a1c73778bfec05850403140e1755b"),
        Asset(path: "Head_L512_Q16_O64/Head_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: "28e57752ad58b2e7cdd20e6e54a710b2b9052b76ce46fbd0046229be6ba1550b"),
        Asset(path: "Head_L512_Q16_O64/Head_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: "a09ada271a4b9d1ecebfddaa4ad431b17a411d3d6c6abdd8b896c20f1a02a433"),
        Asset(path: "Head_L512_Q16_O64/Head_fp32.mlpackage/Manifest.json", sha256: "9dd365d72acf921bdc0e9865c888ef596a97e5df4aee5a145363a67e43ca9aa4"),
        Asset(path: "Head_L512_Q16_O64/config.json", sha256: "e60c695ea045e1e452ebf5e8030f8a9e6e513a1401ac9d2d88a60d695d542169"),
        Asset(path: "LICENSE", sha256: "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        Asset(path: "LM_L1024/LMRows_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: "337fa858a413f93a51670c6d19d463632e628cbb9f38a29c51953e49d76665d5"),
        Asset(path: "LM_L1024/LMRows_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: "d35107d8c86bb6501ece13ecbba125e8b7fe40778e8c7e208b281cf6ec203335"),
        Asset(path: "LM_L1024/LMRows_fp16.mlpackage/Manifest.json", sha256: "5c4319a3db03c2a1ee2b4ae9bce5736738a39f3a563beed946d1ee44009c7316"),
        Asset(path: "LM_L1024/config.json", sha256: "8f3bfa6e070cfdfb28d63d4709d43464b1659d004e8bea980be0160d41097684"),
        Asset(path: "LM_L2048/LMRows_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: "e23acb43d255dc4e9659c3440a61f676c5745d8a7c48cef4299c804bcd045779"),
        Asset(path: "LM_L2048/LMRows_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: "5c076838927c030b5f59adb0576243c872ef773218dcd3889ff4c3ff1b5fa3ac"),
        Asset(path: "LM_L2048/LMRows_fp16.mlpackage/Manifest.json", sha256: "3a08a19950c6f9b8fed2dbb0bedba1632c49768ec01bbb06cdedf73c5c8e7b64"),
        Asset(path: "LM_L2048/config.json", sha256: "9b1729d71002fd7f37bc58107db7cc74e2fd4ba1780e41813b39e87ad5407416"),
        Asset(path: "LM_L512/LMRows_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: "d297c41c8df9731954d96899ccb0251cc34b508abb26dfcca4367c4f1f34091d"),
        Asset(path: "LM_L512/LMRows_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: "e96a5acb02537b48dd491f4c83be9d01f20671a4ee1798ebcf71a85915b7c7a1"),
        Asset(path: "LM_L512/LMRows_fp16.mlpackage/Manifest.json", sha256: "c7bd2d73383d4823277661b926929f9b4c56ea8bcfadb066a713817ae093bd0d"),
        Asset(path: "LM_L512/config.json", sha256: "95537637d70a4e635b575379397ca43b87abc0007d3945004fe3723f2032e8a3"),
        Asset(path: "Vision_P784/VisionTower_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel", sha256: "6eea281b0ee69209e0a37215c16b08726137188e9f1d90b5d9e33380730fcde6"),
        Asset(path: "Vision_P784/VisionTower_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin", sha256: "3d571710dd88a5f2e34d0b51b815a839bc0b53425eca17a45f965d474218ec6e"),
        Asset(path: "Vision_P784/VisionTower_fp32.mlpackage/Manifest.json", sha256: "dc03fc76cb06fa85ddfff734f9f6bc33d6eb4b37b3de946666d5444ed0dd1250"),
        Asset(path: "Vision_P784/config.json", sha256: "b11ffdc05e8ee10f64644fe35eb61603a9627c7908e887722d3c5e59eb4d2f6a"),
        Asset(path: "Vision_P784/pos_embed.f32", sha256: "9a883241c3d1f55bbb10b1aca1202f463dc0ebe0efba8d11c4d00aaf1e95a4db"),
        Asset(path: "config.json", sha256: "c5eea6b91054bdbbd04488f1a896f7c32018c6398dbea4aa7cbb4974c5bf53c4"),
        Asset(path: "embeddings.f16", sha256: "24375b3c4ceac14cb5a66efa63545eae9ca0f37a6b57ea7a713797784249771b"),
        Asset(path: "tokenizer.json", sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        Asset(path: "tokenizer_config.json", sha256: "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"),
    ]

    /// Ensure the bundle exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present. `buckets` limits the LM/head buckets fetched.
    public static func ensure(
        cacheDirectory: URL? = nil, buckets: [Int]? = nil, progress: Progress? = nil, only: Set<String>? = nil
    ) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("clef-vision-0.8b-coreml", isDirectory: true)
        let manager = FileManager.default
        let wanted = assets.filter { asset in
            if let only { return only.contains(asset.path) }
            guard let buckets else { return true }
            guard let bucket = bucketOf(asset.path) else { return true }
            return buckets.contains(bucket)
        }
        let selection = only.map { $0.sorted().joined(separator: "+") } ?? (buckets ?? []).map(String.init).joined(separator: "-")
        let verified = directory.appendingPathComponent(".verified-\(revision)-\(selection)")
        if manager.fileExists(atPath: verified.path),
            wanted.allSatisfy({ manager.fileExists(atPath: directory.appendingPathComponent($0.path).path) })
        {
            return directory
        }
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for asset in wanted {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 { continue }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw ClefVisionError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw ClefVisionError.download("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) for \(asset.path)")
            }
            guard try checksum(of: temporary) == asset.sha256 else { throw ClefVisionError.checksumMismatch(asset.path) }
            if manager.fileExists(atPath: destination.path) { try manager.removeItem(at: destination) }
            try manager.moveItem(at: temporary, to: destination)
            progress?(asset.path, Int64((try? manager.attributesOfItem(atPath: destination.path)[.size] as? Int64) ?? 0))
        }
        manager.createFile(atPath: verified.path, contents: nil)
        return directory
    }

    /// Only `config.json` and `tokenizer.json` (a few MB): enough for `ClefVisionManager.encoder(from:)`.
    public static func ensureEncoderAssets(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        try await ensure(cacheDirectory: cacheDirectory, buckets: nil, progress: progress,
                         only: ["config.json", "tokenizer.json"])
    }

    /// Bucket length of an `LM_L*` / `Head_L*` asset path, nil for shared files.
    static func bucketOf(_ path: String) -> Int? {
        guard let component = path.split(separator: "/").first else { return nil }
        for prefix in ["LM_L", "Head_L"] where component.hasPrefix(prefix) {
            let digits = component.dropFirst(prefix.count).prefix { $0.isNumber }
            return Int(digits)
        }
        return nil
    }

    static func checksum(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
