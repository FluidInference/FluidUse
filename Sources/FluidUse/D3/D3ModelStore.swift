import CryptoKit
import Foundation

/// A published d3 Core ML snapshot (vLLM Semantic Router Decision 3.0 v3.1.0), laid out as `D3Manager.load(from:)`
/// expects: text buckets E256/E640/E1536 (int8 weights), vision buckets P320/P1536/P3072 (fp32), embeddings,
/// readout, vision position table and tokenizer.
public enum D3Model: String, Sendable, CaseIterable {
    /// d3-lite (0.85B): fastest, ~3 GB (`lite/` in FluidInference/d3-coreml).
    case lite = "d3-lite-coreml"
    /// d3-nano (2.2B): ~9 GB (`nano/`).
    case nano = "d3-nano-coreml"
    /// d3-mini (4.5B): strongest, ~16 GB (`mini/`).
    case mini = "d3-mini-coreml"

    static let repository = "FluidInference/d3-coreml"
    static let revision = "31d0a53b53fcabba866acd261bb9209280098509"

    /// The model's folder in the repository.
    var folder: String {
        switch self {
        case .lite: "lite"
        case .nano: "nano"
        case .mini: "mini"
        }
    }

    var assets: [D3ModelStore.Asset] {
        switch self {
        case .lite: D3ModelStore.liteAssets
        case .nano: D3ModelStore.nanoAssets
        case .mini: D3ModelStore.miniAssets
        }
    }
}

/// Downloads a pinned, checksummed d3 Core ML snapshot into the FluidUse cache.
public enum D3ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    struct Asset {
        let path: String
        let sha256: String
    }

    /// Ensure the snapshot exists in the FluidUse cache and return its directory. Files are checksummed once per
    /// pinned revision; later launches only check that they are present.
    public static func ensure(
        _ model: D3Model, cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws
        -> URL
    {
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent(model.rawValue, isDirectory: true)
        let manager = FileManager.default
        let verified = directory.appendingPathComponent(".verified-\(D3Model.revision)")
        if manager.fileExists(atPath: verified.path),
            model.assets.allSatisfy({ manager.fileExists(atPath: directory.appendingPathComponent($0.path).path) })
        {
            return directory
        }
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for asset in model.assets {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(asset.path)
            if manager.fileExists(atPath: destination.path), try checksum(of: destination) == asset.sha256 { continue }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard
                let url = URL(
                    string:
                        "https://huggingface.co/\(D3Model.repository)/resolve/\(D3Model.revision)/\(model.folder)/\(escaped)"
                )
            else { throw D3Error.invalidAsset("Invalid Hugging Face asset URL for \(asset.path)") }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw D3Error.invalidAsset(
                    "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) downloading \(asset.path)")
            }
            guard try checksum(of: temporary) == asset.sha256 else {
                throw D3Error.invalidAsset("Checksum mismatch for \(asset.path)")
            }
            if manager.fileExists(atPath: destination.path) { try manager.removeItem(at: destination) }
            try manager.moveItem(at: temporary, to: destination)
            progress?(
                asset.path, Int64((try? manager.attributesOfItem(atPath: destination.path)[.size] as? Int64) ?? 0))
        }
        manager.createFile(atPath: verified.path, contents: nil)
        return directory
    }

    static func checksum(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static let liteAssets: [Asset] = [
        Asset(path: "coreml_config.json", sha256: "59769cae37d510990513d36634e6857e1cfc1b30c5466a31d3404bff82df3d21"),
        Asset(
            path: "d3_E1536.mlpackage/Manifest.json",
            sha256: "89d0ab3fdeab33bcbbe03619739ccf47e3471771812e0523ad33a6078d9d1337"),
        Asset(
            path: "d3_E1536.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "06625e4407aba711b4d13c0caaa85ea5d11d573d6f7e4245b62b52c49a960534"),
        Asset(
            path: "d3_E1536.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "1570efcc188e884763f440b4f4780383b2b27b7286ff113a0c7b2f8dcdb83ddf"),
        Asset(
            path: "d3_E256.mlpackage/Manifest.json",
            sha256: "712bf90ed6542f82a32e713ee4a76f61bb0965faa803ee6743731721adf3aef1"),
        Asset(
            path: "d3_E256.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "9886980c329516c7732b7c07f21cd480bc6b953a2091876743b5b39799848489"),
        Asset(
            path: "d3_E256.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "a5565253ea689f841b849230ed43061d12f993c0d6c293627cef497cf7d64787"),
        Asset(
            path: "d3_E640.mlpackage/Manifest.json",
            sha256: "11dc54a60726c8740d59efdbee386c737dd8195f890733ed5622492f3fcf08af"),
        Asset(
            path: "d3_E640.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "4c1ef6632dc8144243c270a18f1a4a27c519dd21a98d0f25f443300a325484c8"),
        Asset(
            path: "d3_E640.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "7c18d2ca33683162515a27f22c2ce7cec5832eef740a7eb165a6ffcdb020a7e7"),
        Asset(path: "embeddings.f16", sha256: "dd7ed88d1e676ff4a3dc46d25d4751b1e465540a31522aba620a4948878e14e9"),
        Asset(path: "pos_embed.f32", sha256: "9a883241c3d1f55bbb10b1aca1202f463dc0ebe0efba8d11c4d00aaf1e95a4db"),
        Asset(path: "readout.f32", sha256: "7772300f2e211f3c2028a16b9f493e941c2343bd14a9131145eb34da39f36448"),
        Asset(path: "tokenizer.json", sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        Asset(
            path: "vision_P1536.mlpackage/Manifest.json",
            sha256: "6aefe6df060f5b80bac0429811425245528dffb7e435ec210d6da206d63d8eb1"),
        Asset(
            path: "vision_P1536.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "eec6afaa226859a3d717d2ced945947b3663d1b641076d0abc5e96af19b61d57"),
        Asset(
            path: "vision_P1536.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "3d571710dd88a5f2e34d0b51b815a839bc0b53425eca17a45f965d474218ec6e"),
        Asset(
            path: "vision_P3072.mlpackage/Manifest.json",
            sha256: "d73486fa2517d8957b35a9b40f0faa997c4349d2bbce406460d93eb1f45a4bdb"),
        Asset(
            path: "vision_P3072.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "4d6a1ec684714a1aeb552cb699b9e5303a342db6c0bec6ab481abe1581df1810"),
        Asset(
            path: "vision_P3072.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "3d571710dd88a5f2e34d0b51b815a839bc0b53425eca17a45f965d474218ec6e"),
        Asset(
            path: "vision_P320.mlpackage/Manifest.json",
            sha256: "4015cd57dc6d47170aee6e58c36d8d8ed7a63cd2900379d09c1d4fd407b3f47b"),
        Asset(
            path: "vision_P320.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "9c6e604d1d2e1c2b63d5cd01a0263e0245c10b7a41c16bb177953c9208166b8b"),
        Asset(
            path: "vision_P320.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "3d571710dd88a5f2e34d0b51b815a839bc0b53425eca17a45f965d474218ec6e"),
    ]
    static let nanoAssets: [Asset] = [
        Asset(path: "coreml_config.json", sha256: "14c45cd115eaeda620de6589a0a71c1939c1231993cc74120f1cf71ea279a98a"),
        Asset(
            path: "d3_E1536.mlpackage/Manifest.json",
            sha256: "a24f1306af2275395254130699bf04a69ecab9a3e288e2ea0555ae75c3c81efb"),
        Asset(
            path: "d3_E1536.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "06f7d1405c584d485f5015cc6ebe30d1770ec7025dc6d90bb21ff3cb48a61e03"),
        Asset(
            path: "d3_E1536.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "93465b1962a071fa2e3b806ee68087714707598717b2d68fff1a403420c7ec9c"),
        Asset(
            path: "d3_E256.mlpackage/Manifest.json",
            sha256: "c24c283b2a80db66051dc5843c024bedd7afb854919c9a8898a3b16e2475cc6c"),
        Asset(
            path: "d3_E256.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "e939825745fab97d887a96738e7bc42b8054f1f2155ef288ea49884f0831fd7b"),
        Asset(
            path: "d3_E256.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "e0f8006ceff9c4c7bdf876a50c2a66b11529389b5ee4ed9ab4c89180a91dedf5"),
        Asset(
            path: "d3_E640.mlpackage/Manifest.json",
            sha256: "b0ba3ad1e6843ece7b84df52278d813cb6c43b9e6ed6cdb977cf6adac3fdd279"),
        Asset(
            path: "d3_E640.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "cf938a6d97278f4c818887bad323422df209350a89930f6ca1e9c41b2e52b78d"),
        Asset(
            path: "d3_E640.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "333cfe8ed9a9f5926630fca98be2af1c5eb8431752543acb71c813b4268ecd37"),
        Asset(path: "embeddings.f16", sha256: "8782b61738abba82d7a1c11acdad6614b975a4ed64f508edbb0d76cb5568b8b3"),
        Asset(path: "pos_embed.f32", sha256: "7dfb2506f3eb27d3c899d8c087bab761aaac73b74a2859d3a9cbf74e37d1953c"),
        Asset(path: "readout.f32", sha256: "918534a9659d94b770e12f5f654fe59f274938231e8d260905566b7c5120ceaf"),
        Asset(path: "tokenizer.json", sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        Asset(
            path: "vision_P1536.mlpackage/Manifest.json",
            sha256: "7222a2056e8564096e0d35faa4ff85f1439fbdd81e3182be58c97218abd5b9b3"),
        Asset(
            path: "vision_P1536.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "74e0123c717f387db9787231175e66efded052ff8b2ef43507fba1d490ddfca1"),
        Asset(
            path: "vision_P1536.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "03411782624cea6bf34076d497a5cae62bd27ba6841a4bc260870558498bf77e"),
        Asset(
            path: "vision_P3072.mlpackage/Manifest.json",
            sha256: "28c32eb8e9ad74c0c9f7c87aa76efed9f1f267a5d6f41486aa9ec3208fff9bc5"),
        Asset(
            path: "vision_P3072.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "b049450c0b60b023d388ea2405fd9fd6646d2b6aa448846a42353b65c3e357be"),
        Asset(
            path: "vision_P3072.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "03411782624cea6bf34076d497a5cae62bd27ba6841a4bc260870558498bf77e"),
        Asset(
            path: "vision_P320.mlpackage/Manifest.json",
            sha256: "56af12ad22e330e3d052ef8507a1284c85fdb1bfdc8b67995b77558586a30e1f"),
        Asset(
            path: "vision_P320.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "44d10a0efb0db856213de15ee6e5efd9c41ab4c386d4489efa536d6f6f38407d"),
        Asset(
            path: "vision_P320.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "03411782624cea6bf34076d497a5cae62bd27ba6841a4bc260870558498bf77e"),
    ]
    static let miniAssets: [Asset] = [
        Asset(path: "coreml_config.json", sha256: "f86fd46a2e019a1bb07bd8eb8885f1be7efadae994343f0dda2e652329ee4dc0"),
        Asset(
            path: "d3_E1536.mlpackage/Manifest.json",
            sha256: "a666dc006448d84d3a47040fba65d8ed7d5b677f36e70bfc8ad84812dfee1088"),
        Asset(
            path: "d3_E1536.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "7ef11f9d9059ae672cfdcb86445ffc28bb0a40433d74d2ab59f04c14f35e43e1"),
        Asset(
            path: "d3_E1536.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "6258df13e630f9311833fa21ff921e973f1c8c0225bef9878a4b344ceb3fbae1"),
        Asset(
            path: "d3_E256.mlpackage/Manifest.json",
            sha256: "f56d1126f9803f21e15ee189a45baa4926601024fe982277ad1c8fd3ed2e89c9"),
        Asset(
            path: "d3_E256.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "087c2d2717a302240bf5c6d8f2dc4710d360598e6f84e5602160a7a6d24ad141"),
        Asset(
            path: "d3_E256.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "0c3dcffd1486a75836a6830d3537a3215beae02f20902c956a20e130f25861c0"),
        Asset(
            path: "d3_E640.mlpackage/Manifest.json",
            sha256: "9bc936251cf9698699f1fed01c8bb2d9e9adc950ea63e45866ad2d39bba85ccb"),
        Asset(
            path: "d3_E640.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "61c8e64e6d12fe92848e58bf1656fe2596fb4f08b60efa38c8402eb7ba0fc2c0"),
        Asset(
            path: "d3_E640.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "786564af8f1c8763bf1dfa24f5bbb5e77952fc3197794405a3f59edd4e9d4f13"),
        Asset(path: "embeddings.f16", sha256: "e41907b18d12ac35458d4df8606a51d59a58c0de0220383a6c3e5703ab4e0b08"),
        Asset(path: "pos_embed.f32", sha256: "c860a47dbf5b7bf54636c5620d1de7103a5c1187b2e36e0e5cb2267bc7b9da11"),
        Asset(path: "readout.f32", sha256: "7b4e0c71e8cb3a7087861491588272cbf5a7d398cdb5fea410d4e2d5ae0cd943"),
        Asset(path: "tokenizer.json", sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        Asset(
            path: "vision_P1536.mlpackage/Manifest.json",
            sha256: "e212c960a16a3d7feea557eff2d3cb67b5f08d6d5edc312240c227cce5bcece6"),
        Asset(
            path: "vision_P1536.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "5f7fd6703f695e82a00c00fffd0ab5c1bc7105692ed91add863051cabdb0ba94"),
        Asset(
            path: "vision_P1536.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "b022b2e91b22c8b37a51128ca498d136f577f971bcbf4a91ce5694b1e407e0a8"),
        Asset(
            path: "vision_P3072.mlpackage/Manifest.json",
            sha256: "26415cebcde61612770372bb76ab4c56641fb7e97e3e37e651910cb669fac249"),
        Asset(
            path: "vision_P3072.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "cb25d5c5728301e8e91bab38bf482defc02d0cc4cf14b58e191ffd33fc80212b"),
        Asset(
            path: "vision_P3072.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "b022b2e91b22c8b37a51128ca498d136f577f971bcbf4a91ce5694b1e407e0a8"),
        Asset(
            path: "vision_P320.mlpackage/Manifest.json",
            sha256: "3f27fb96eb7e8632182820c5ee45523b7b6ac0abd46aad1464e1b15121dc5566"),
        Asset(
            path: "vision_P320.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "35592ddff585aa1dad7aeb2c3ddd8a87018189a6599e550ae3ad0458e6b2f14c"),
        Asset(
            path: "vision_P320.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "b022b2e91b22c8b37a51128ca498d136f577f971bcbf4a91ce5694b1e407e0a8"),
    ]
}
