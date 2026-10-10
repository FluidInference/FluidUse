import CryptoKit
import Foundation

/// Downloads the pinned clef-flash Core ML bundle (FluidInference/clef-flash-coreml): 8 decoder parts (8-bit), the
/// joint schema head, input / output embedding tables, tokenizer and manifest, as `ClefFlashManager.load(from:)` expects.
/// About 11 GB; files are checksummed once per pinned revision.
public enum ClefFlashModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    public static let repository = "FluidInference/clef-flash-coreml"
    public static let revision = "c1b8fdae2ec1bcc643af30aa801d42ad4df216e1"
    static let assets: [Asset] = [
        Asset(path: "config.json", sha256: "7b1a40bc187d7147cc8cd2598a2e76c8ea7a73eaa1ab0498925ee0b873763a8c"),
        Asset(path: "embeddings.f16", sha256: "7d87bcd3dbacdc2ab33fff49adaa4c4f35691d829c8aae75670bcf8792a10242"),
        Asset(
            path: "Head.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "4948692cc38639f0fbba22c4b01c46279bfa792aef4a45c67081d270abce0cc5"),
        Asset(
            path: "Head.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "728644e4e8d20925b35a8473a67696db4e6d1d1098b5ac31edb95fe7f98deeca"),
        Asset(
            path: "Head.mlpackage/Manifest.json",
            sha256: "bcea53e44b3ada9db0e46b93d22e814a54d8511df95642f04b0e2ad409b87aef"),
        Asset(path: "LICENSE", sha256: "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        Asset(
            path: "output_embeddings.f16", sha256: "bfccf00b8d5d3f6d810839a52c2ec627700a9aaebaa26aedcb96931bd21ba36d"),
        Asset(
            path: "part00.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0dc30324169ab34c6d5bbdc294bae25a2294a8b17ffeaaf3a6f066654c681b15"),
        Asset(
            path: "part00.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "12d820ca17fa2fe995115022acb9ee0fc9bdeb1fbf30f516431dfa7c07546daa"),
        Asset(
            path: "part00.mlpackage/Manifest.json",
            sha256: "b3435c31ee258ec40122f5804a590361d214f6ac87521d6b0117fa08b5146817"),
        Asset(
            path: "part01.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0dc30324169ab34c6d5bbdc294bae25a2294a8b17ffeaaf3a6f066654c681b15"),
        Asset(
            path: "part01.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "e890c6f147687144bca6c075d24b3bffd7f4552dcfe1b7e0c3ae6ea90d8771fd"),
        Asset(
            path: "part01.mlpackage/Manifest.json",
            sha256: "6bd6afb214e91dff8ca2e34445492ca19f61f88b589b386f4f74f01042971c2a"),
        Asset(
            path: "part02.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0dc30324169ab34c6d5bbdc294bae25a2294a8b17ffeaaf3a6f066654c681b15"),
        Asset(
            path: "part02.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "024bd67205cfd750b352744d8193d34fe36338f3ba8ca43bde937be4a4f91286"),
        Asset(
            path: "part02.mlpackage/Manifest.json",
            sha256: "5efd6b116698c67b9d5135d120fcc3c3aadab6f07c1c15552c25ece41023e191"),
        Asset(
            path: "part03.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0dc30324169ab34c6d5bbdc294bae25a2294a8b17ffeaaf3a6f066654c681b15"),
        Asset(
            path: "part03.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "987723220468840b1068117cf389a676809d632c1cff3493882498bbcd4f533a"),
        Asset(
            path: "part03.mlpackage/Manifest.json",
            sha256: "748d86dc2e3c62a2796a488d63f04413aa7ea0703c97fc260749031d8e794541"),
        Asset(
            path: "part04.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0dc30324169ab34c6d5bbdc294bae25a2294a8b17ffeaaf3a6f066654c681b15"),
        Asset(
            path: "part04.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "ad33fc8b33c26b20830079883e13e48b375f86849b1f75a8120f69d0a3ed16f9"),
        Asset(
            path: "part04.mlpackage/Manifest.json",
            sha256: "5c93ccf6bf2506fc833e5f4b06dc9282e0fe0220bdb69937e6d8dbc6311f188e"),
        Asset(
            path: "part05.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0dc30324169ab34c6d5bbdc294bae25a2294a8b17ffeaaf3a6f066654c681b15"),
        Asset(
            path: "part05.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "144daf8ac9d192e5dbd1ca0f1b1c1324f02497f51d6d461b2f8d7c678fc72263"),
        Asset(
            path: "part05.mlpackage/Manifest.json",
            sha256: "ea9459ddf3ea325d2552861e7132f7f76bcd21883ff1a97d316ac5091c94de5c"),
        Asset(
            path: "part06.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0dc30324169ab34c6d5bbdc294bae25a2294a8b17ffeaaf3a6f066654c681b15"),
        Asset(
            path: "part06.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "b4c14e83eecc79791379d23dd21f2585ed15bf5a3e6420bd06c674a94dec4a40"),
        Asset(
            path: "part06.mlpackage/Manifest.json",
            sha256: "42c3546baca964b35bae0c9bf300bcd08f69f12f26ef6cef84badb9dd88f9661"),
        Asset(
            path: "part07.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "c1c2077fc17ac52bfdc7663680ac9d36b4a94b5845bb6c4b0c68933eadb85b34"),
        Asset(
            path: "part07.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "1c8387edfb85910631c044dd9c96ea410004aea3c8cd15c80c6c4d68eb3b3e6d"),
        Asset(
            path: "part07.mlpackage/Manifest.json",
            sha256: "1a5b1ce459ba1ff551c64da117ddc186bb4deafe503963e8be5f14868036cdf0"),
        Asset(path: "tokenizer.json", sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    ]

    /// Ensure the bundle exists in the FluidUse cache and return its directory.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("clef-flash-coreml", isDirectory: true)
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
                throw ClefVisionError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw ClefVisionError.download(
                    "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) for \(asset.path)")
            }
            guard try ClefVisionModelStore.checksum(of: temporary) == asset.sha256 else {
                throw ClefVisionError.checksumMismatch(asset.path)
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
