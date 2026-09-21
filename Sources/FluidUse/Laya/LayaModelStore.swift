import Foundation

/// File names and download of the `FluidInference/laya-coreml` artifacts.
///
/// Each bucket is a compiled `.mlmodelc` directory of four files plus one shared `tokenizer.json`.
/// Files are fetched straight from the Hub into `~/Library/Application Support/FluidUse/Models/laya-coreml`
/// (or a caller-supplied cache root); existing files are kept, partial downloads are discarded.
public enum LayaModelStore {
    public static let repository = "FluidInference/laya-coreml"
    /// Fixed sequence lengths exported by the Mobius conversion.
    public static let lengths = [128, 256, 512, 1024]
    /// HuggingFace `tokenizer.json` of the mmBERT/Gemma vocabulary.
    public static let tokenizerFile = "tokenizer.json"
    /// Members of a compiled Core ML bundle as published.
    static let bundleMembers = ["analytics/coremldata.bin", "coremldata.bin", "model.mil", "weights/weight.bin"]

    /// Compiled bucket bundle for one sequence length.
    public static func modelFile(length: Int) throws -> String {
        guard lengths.contains(length) else {
            throw LayaError.invalidAsset("No laya bucket for length \(length); available: \(lengths)")
        }
        return "laya_multilingual_fp16_L\(length)_options\(LayaManager.maximumOptions).mlmodelc"
    }

    /// Default cache root; the repository directory lives underneath it.
    public static func defaultCacheDirectory() -> URL {
        let manager = FileManager.default
        let base =
            manager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? manager.temporaryDirectory
        return base.appendingPathComponent("FluidUse/Models", isDirectory: true)
    }

    /// Progress callback: bytes so far and the file being fetched.
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    /// Ensure the buckets and tokenizer exist under `cacheDirectory/laya-coreml`, downloading what is missing.
    /// Returns the repository directory.
    public static func ensure(
        lengths: [Int], cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        let root = cacheDirectory ?? defaultCacheDirectory()
        let repoDirectory = root.appendingPathComponent("laya-coreml", isDirectory: true)
        var relativePaths = [tokenizerFile]
        for length in lengths {
            let bundle = try modelFile(length: length)
            relativePaths += bundleMembers.map { "\(bundle)/\($0)" }
        }
        let manager = FileManager.default
        for relative in relativePaths {
            let destination = repoDirectory.appendingPathComponent(relative)
            if manager.fileExists(atPath: destination.path) { continue }
            try manager.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoded = relative.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? relative
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/main/\(encoded)") else {
                throw LayaError.invalidAsset("Bad download URL for \(relative)")
            }
            progress?(relative, 0)
            let (temporary, response) = try await URLSession.shared.download(from: url)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                try? manager.removeItem(at: temporary)
                throw LayaError.invalidAsset(
                    "Download of \(relative) failed (\((response as? HTTPURLResponse)?.statusCode ?? -1))")
            }
            try? manager.removeItem(at: destination)
            try manager.moveItem(at: temporary, to: destination)
            let size = (try? manager.attributesOfItem(atPath: destination.path)[.size] as? Int64) ?? 0
            progress?(relative, size)
        }
        return repoDirectory
    }
}
