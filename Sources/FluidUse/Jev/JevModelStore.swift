import Foundation

/// Downloads the pinned JEV-9B Core ML bundle (FluidInference/jev-9b-coreml): 8 decoder parts (8-bit, System 1 LoRA
/// merged), the fp32 vision towers for 256- and 576-patch images, embeddings, head rows, tokenizer and manifest, as
/// `JevManager.load(from:)` expects. About 13 GB; files are checksummed once per pinned revision.
public enum JevModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    public static let repository = "FluidInference/jev-9b-coreml"
    public static let revision = "a718625da54446f09cd63d8c3962bfe53d81c4a9"
    static let assets: [Asset] = [
        Asset(path: "config.json", sha256: "bccfe3e196fde9da7357fc3ef8f6714999ea9e60b2b7e94248603e123e201a69"),
        Asset(path: "embeddings.f16", sha256: "7d87bcd3dbacdc2ab33fff49adaa4c4f35691d829c8aae75670bcf8792a10242"),
        Asset(path: "head_rows.f32", sha256: "214244c905063010fb2473bca2ecb44645119e056cdf10a284e0f70ad3d9b21f"),
        Asset(path: "head.json", sha256: "a06764e3018281e46ad48f107772f40880dee9e5fbd363582e2df17290a24835"),
        Asset(path: "LICENSE", sha256: "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
        Asset(
            path: "part00.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0bcab81eeb64a8cb38f30ad5999f91da64291a7973431fabeb595978ddd30634"),
        Asset(
            path: "part00.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "d6afcc3b6c7a9747247851778792a5c13f6b244d53eb1cff717a3b7881b271f2"),
        Asset(
            path: "part00.mlpackage/Manifest.json",
            sha256: "c27ee94412bcd5514f120c9848f0eed75c51322714cc4912e22bf23a8ada513a"),
        Asset(
            path: "part01.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0bcab81eeb64a8cb38f30ad5999f91da64291a7973431fabeb595978ddd30634"),
        Asset(
            path: "part01.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "02c09420f6dc3eb1ac93c2adb3c39adede616beabfad0a19e6ef27d3c1117202"),
        Asset(
            path: "part01.mlpackage/Manifest.json",
            sha256: "3513ae77df0cf9d4ccf658519ec7943c04b35edbd6500a996fea197666deadc7"),
        Asset(
            path: "part02.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0bcab81eeb64a8cb38f30ad5999f91da64291a7973431fabeb595978ddd30634"),
        Asset(
            path: "part02.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "0f1c235137345d7a8dd0bec7022bd906a90354d77ac80cd34ac2ea8308ec4994"),
        Asset(
            path: "part02.mlpackage/Manifest.json",
            sha256: "10d60e47dd518d0c001e692330cbe4a7121f282e17dd870d3b3c82d035655d60"),
        Asset(
            path: "part03.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0bcab81eeb64a8cb38f30ad5999f91da64291a7973431fabeb595978ddd30634"),
        Asset(
            path: "part03.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "30c42923a00bba96303617bb3958a1cc3846cc46961eaba8153f3134bd9ea217"),
        Asset(
            path: "part03.mlpackage/Manifest.json",
            sha256: "f6ac5ed4c39505a824c6e446177163f1484541516163b1b5000d611ae1d3b2f0"),
        Asset(
            path: "part04.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0bcab81eeb64a8cb38f30ad5999f91da64291a7973431fabeb595978ddd30634"),
        Asset(
            path: "part04.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "18588999cf26636cea57fbe27407dd686fa61eb132059d59a6aefbf0bbb668e3"),
        Asset(
            path: "part04.mlpackage/Manifest.json",
            sha256: "2d0ac2e3d8906a2a865bada9a93debeb93791c6dbdd239494ab5b30c91866e61"),
        Asset(
            path: "part05.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0bcab81eeb64a8cb38f30ad5999f91da64291a7973431fabeb595978ddd30634"),
        Asset(
            path: "part05.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "d6bb092984a185b8c5a8034476199c7940b8fa0ef7ddd12407e94faa19340c31"),
        Asset(
            path: "part05.mlpackage/Manifest.json",
            sha256: "6ebd70522296d5d8f5c486a10dfb598ba2e42ebb8e34da3ee3af8f298d738c71"),
        Asset(
            path: "part06.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0bcab81eeb64a8cb38f30ad5999f91da64291a7973431fabeb595978ddd30634"),
        Asset(
            path: "part06.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "12bdf4e71405cfbe7d4e536c3e254b69e9f0f72e92696266a39e21c13abbce73"),
        Asset(
            path: "part06.mlpackage/Manifest.json",
            sha256: "acb70f8f49a6e777996a810ecf0fc6021389d973c91ec660f53b0f1d888f872a"),
        Asset(
            path: "part07.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "f902c868ce2577fdbba3f23df4f1094de1a19afcea33c10ec54ed7c36af5853e"),
        Asset(
            path: "part07.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "2d24b9211e27e40d9ca4cf2feee4c76df38f4cd346cee99ef2f481068bc7d270"),
        Asset(
            path: "part07.mlpackage/Manifest.json",
            sha256: "ccb89391958ae5b9f0969f60ee0bda953aacb41dd5ed0383d1a2b5cdc9d536ed"),
        Asset(path: "pos_embed.f32", sha256: "8d123c332694df52f46467865e58923aa1fe47ee353780eb7a39467eea3490aa"),
        Asset(path: "tokenizer.json", sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        Asset(
            path: "VisionTower_P256.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "4f5671928d0fac2ee2331b6b901ff519771097e392d235a122d92324c53226a6"),
        Asset(
            path: "VisionTower_P256.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "eaceac75cd1c45401c0f910577adb17a3a48a38b9557c335a0e51b7b9b5d0a9b"),
        Asset(
            path: "VisionTower_P256.mlpackage/Manifest.json",
            sha256: "b0545cdc4546413a184efaff4723a21e4b23c8a973acaa9730109a403578b61d"),
        Asset(
            path: "VisionTower_P576.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "50a8894c45ea7447a18d418245205657fcb83c65a9aae0658373393c7b8a94bf"),
        Asset(
            path: "VisionTower_P576.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "eaceac75cd1c45401c0f910577adb17a3a48a38b9557c335a0e51b7b9b5d0a9b"),
        Asset(
            path: "VisionTower_P576.mlpackage/Manifest.json",
            sha256: "3c3053d6eef3e97ac79168f0316f1b97892c048bbe92ab47827031b30aa1b246"),
    ]

    /// Ensure the bundle exists in the FluidUse cache and return its directory.
    public static func ensure(cacheDirectory: URL? = nil, progress: Progress? = nil) async throws -> URL {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("jev-9b-coreml", isDirectory: true)
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
                throw JevError.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw JevError.invalidAsset(
                    "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) for \(asset.path)")
            }
            guard try ClefVisionModelStore.checksum(of: temporary) == asset.sha256 else {
                throw JevError.invalidAsset("checksum mismatch: \(asset.path)")
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
