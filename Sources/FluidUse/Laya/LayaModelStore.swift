import Foundation

/// File names and download of the `FluidInference/laya-coreml` artifacts.
///
/// Each bucket is a compiled `.mlmodelc` directory of four files plus one shared `tokenizer.json`.
/// Files are fetched straight from the Hub into `~/Library/Application Support/FluidUse/Models/laya-coreml`
/// (or a caller-supplied cache root). Existing nonempty files are kept; a download is accepted only with
/// a 2xx status, a non-HTML body, and the advertised Content-Length, and is moved into place atomically.
public enum LayaModelStore {
    public static let repository = "FluidInference/laya-coreml"
    /// Fixed sequence lengths exported by the Mobius conversion.
    public static let lengths = [128, 256, 512, 1024]
    /// HuggingFace `tokenizer.json` of the mmBERT/Gemma vocabulary.
    public static let tokenizerFile = "tokenizer.json"
    /// Members of a compiled Core ML bundle as published.
    static let bundleMembers = ["analytics/coremldata.bin", "coremldata.bin", "model.mil", "weights/weight.bin"]

    /// Weight precisions published for every bucket. `fp16` is the reference; `e8` keeps the encoder in
    /// fp16 and stores the 256k-row embedding table as int8 (30% smaller, same parity gates).
    public static let precisions = ["fp16", "e8"]

    /// Compiled bucket bundle for one sequence length and weight precision.
    public static func modelFile(length: Int, precision: String = "fp16") throws -> String {
        guard lengths.contains(length) else {
            throw LayaError.invalidAsset("No laya bucket for length \(length); available: \(lengths)")
        }
        guard precisions.contains(precision) else {
            throw LayaError.invalidAsset("No laya precision \(precision); available: \(precisions)")
        }
        return "laya_multilingual_\(precision)_L\(length)_options\(LayaManager.maximumOptions).mlmodelc"
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
        lengths: [Int], precision: String = "fp16", cacheDirectory: URL? = nil, progress: Progress? = nil
    ) async throws -> URL {
        let root = cacheDirectory ?? defaultCacheDirectory()
        let repoDirectory = root.appendingPathComponent("laya-coreml", isDirectory: true)
        var relativePaths = [tokenizerFile]
        for length in lengths {
            let bundle = try modelFile(length: length, precision: precision)
            relativePaths += bundleMembers.map { "\(bundle)/\($0)" }
        }
        let manager = FileManager.default
        for relative in relativePaths {
            let destination = repoDirectory.appendingPathComponent(relative)
            if let size = try? manager.attributesOfItem(atPath: destination.path)[.size] as? Int64, size > 0 {
                continue
            }
            try manager.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoded = relative.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? relative
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/main/\(encoded)") else {
                throw LayaError.invalidAsset("Bad download URL for \(relative)")
            }
            progress?(relative, 0)
            // Download next to the destination so the final move is a rename, never a cross-volume copy
            // that could leave a truncated member behind if interrupted.
            let partial = destination.appendingPathExtension("partial")
            try? manager.removeItem(at: partial)
            let (temporary, response) = try await URLSession.shared.download(from: url)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                try? manager.removeItem(at: temporary)
                throw LayaError.invalidAsset(
                    "Download of \(relative) failed (\((response as? HTTPURLResponse)?.statusCode ?? -1))")
            }
            if let type = http.value(forHTTPHeaderField: "Content-Type"), type.contains("text/html") {
                try? manager.removeItem(at: temporary)
                throw LayaError.invalidAsset("Download of \(relative) returned an HTML page instead of the file")
            }
            let size = (try? manager.attributesOfItem(atPath: temporary.path)[.size] as? Int64) ?? 0
            if http.expectedContentLength > 0, size != http.expectedContentLength {
                try? manager.removeItem(at: temporary)
                throw LayaError.invalidAsset(
                    "Download of \(relative) is \(size) bytes, expected \(http.expectedContentLength)")
            }
            guard size > 0 else {
                try? manager.removeItem(at: temporary)
                throw LayaError.invalidAsset("Download of \(relative) is empty")
            }
            try manager.moveItem(at: temporary, to: partial)
            try? manager.removeItem(at: destination)
            try manager.moveItem(at: partial, to: destination)
            progress?(relative, size)
        }
        return repoDirectory
    }
}
