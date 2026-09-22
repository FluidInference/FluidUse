import CryptoKit
import Foundation

/// Downloads the GLiClass Core ML packages described by the Hub repository's `config.json`.
public enum GLiClassModelStore {
    public static let repository = "FluidInference/gliclass-edge-apps-coreml"
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    static let packageMembers = [
        "Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin",
    ]

    struct RepositoryConfig: Decodable {
        struct Bucket: Decodable {
            let length: Int
            let fp16: String?
            let lut8: String?
        }

        let format: String
        let maxOptions: Int
        let tokenizer: String
        let buckets: [Bucket]

        enum CodingKeys: String, CodingKey {
            case format, tokenizer, buckets
            case maxOptions = "max_options"
        }

        func package(length: Int, precision: String) throws -> String {
            guard format == "coreml", maxOptions == GLiClassManager.maximumOptions,
                tokenizer == "tokenizer.json"
            else { throw GLiClassError.invalidAsset("Unexpected GLiClass repository config") }
            guard let bucket = buckets.first(where: { $0.length == length }) else {
                throw GLiClassError.invalidAsset(
                    "GLiClass L\(length) is not published in \(GLiClassModelStore.repository)")
            }
            let package: String?
            switch precision {
            case "fp16": package = bucket.fp16
            case "lut8": package = bucket.lut8
            default: package = nil
            }
            guard let package,
                package == (try GLiClassManager.modelName(length: length, precision: precision)) + ".mlpackage"
            else {
                throw GLiClassError.invalidAsset(
                    "GLiClass \(precision) L\(length) is not published in \(GLiClassModelStore.repository)")
            }
            return package
        }
    }

    /// Ensure selected packages and tokenizer exist under `cacheDirectory/gliclass-edge-apps-coreml`.
    /// The Hub config selects the artifacts; published SHA-256 hashes validate cached and downloaded files.
    public static func ensure(
        lengths: [Int], precision: String = "fp16", cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        guard !lengths.isEmpty else { throw GLiClassError.invalidAsset("At least one GLiClass bucket is required") }
        let root = cacheDirectory ?? LayaModelStore.defaultCacheDirectory()
        let directory = root.appendingPathComponent("gliclass-edge-apps-coreml", isDirectory: true)
        let configData = try await downloadData("config.json")
        let checksumsData = try await downloadData("checksums.json")
        let config = try JSONDecoder().decode(RepositoryConfig.self, from: configData)
        let checksums = try JSONDecoder().decode([String: String].self, from: checksumsData)
        guard let configHash = checksums["config.json"], sha256(configData) == configHash else {
            throw GLiClassError.invalidAsset("GLiClass config.json checksum mismatch")
        }
        var paths = [config.tokenizer]
        for length in lengths {
            let package = try config.package(length: length, precision: precision)
            paths += packageMembers.map { "\(package)/\($0)" }
        }
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try configData.write(to: directory.appendingPathComponent("config.json"), options: .atomic)
        try checksumsData.write(to: directory.appendingPathComponent("checksums.json"), options: .atomic)
        for relative in paths {
            guard let expectedHash = checksums[relative], expectedHash.count == 64 else {
                throw GLiClassError.invalidAsset("No checksum for \(relative)")
            }
            let destination = directory.appendingPathComponent(relative)
            if manager.fileExists(atPath: destination.path), try sha256(file: destination) == expectedHash {
                continue
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(relative, 0)
            let temporary = try await downloadFile(relative)
            defer { try? manager.removeItem(at: temporary) }
            guard try sha256(file: temporary) == expectedHash else {
                throw GLiClassError.invalidAsset("GLiClass checksum mismatch for \(relative)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try LayaModelStore.installDownloadedFile(temporary, at: destination)
            progress?(relative, size)
        }
        return directory
    }

    private static func url(for relative: String) throws -> URL {
        let encoded = relative.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? relative
        guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/main/\(encoded)") else {
            throw GLiClassError.invalidAsset("Bad GLiClass download URL for \(relative)")
        }
        return url
    }

    private static func downloadData(_ relative: String) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: url(for: relative))
        try validate(response, file: relative, size: Int64(data.count))
        return data
    }

    private static func downloadFile(_ relative: String) async throws -> URL {
        let (temporary, response) = try await URLSession.shared.download(from: url(for: relative))
        do {
            let size =
                (try FileManager.default.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?
                .int64Value ?? 0
            try validate(response, file: relative, size: size)
            return temporary
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    private static func validate(_ response: URLResponse, file: String, size: Int64) throws {
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw GLiClassError.invalidAsset(
                "Download of \(file) failed (\((response as? HTTPURLResponse)?.statusCode ?? -1))")
        }
        guard !((http.value(forHTTPHeaderField: "Content-Type") ?? "").contains("text/html")) else {
            throw GLiClassError.invalidAsset("Download of \(file) returned HTML")
        }
        guard size > 0, http.expectedContentLength <= 0 || size == http.expectedContentLength else {
            throw GLiClassError.invalidAsset("Download of \(file) has an unexpected size")
        }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256(file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
