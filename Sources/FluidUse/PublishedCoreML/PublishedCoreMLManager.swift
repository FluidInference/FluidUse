import Foundation

/// Published sub-1B Core ML runtimes whose preprocessing is provided by their conversion toolkits.
public enum PublishedCoreMLModel: String, CaseIterable, Sendable {
    case kev05 = "kev-0-5b"
    case kev06 = "kev-0.6b"
    case kai = "decision-1.0-kai"
    case lex = "decision-1.0-lex"
    case lfm350 = "lfm2-5-350m-rlcd"
    case jeff
    case nanojev

    public var repository: String { "FluidInference/\(rawValue)-coreml" }

    fileprivate func requiredPackages(precision: String) throws -> [String] {
        switch self {
        case .kev05:
            guard ["fp16", "e8"].contains(precision) else { throw PublishedCoreMLError.invalidPrecision(precision) }
            return ["kev_0_5b_\(precision)_L128_options32.mlpackage"]
        case .kev06:
            guard ["fp16", "w8"].contains(precision) else { throw PublishedCoreMLError.invalidPrecision(precision) }
            return ["kev_0_6b_\(precision)_L128_options32.mlpackage"]
        case .kai, .lex:
            guard ["fp16", "w8"].contains(precision) else { throw PublishedCoreMLError.invalidPrecision(precision) }
            let compressed = precision == "w8"
            return ["choice", "noul", "score"].map { kind in
                let useW8 = compressed && (self == .kai || kind != "choice")
                return "coreml/\(kind)\(useW8 ? "-embedding-w8" : "").mlpackage"
            }
        case .lfm350:
            guard precision == "fp16" else { throw PublishedCoreMLError.invalidPrecision(precision) }
            return ["lfm350_rlcd_fp16_L256_B8_V16.mlpackage"]
        case .jeff:
            guard ["fp16", "w8"].contains(precision) else { throw PublishedCoreMLError.invalidPrecision(precision) }
            return [precision == "w8" ? "JeffDecision-L128-W8.mlpackage" : "JeffDecision-L128-FP16.mlpackage"]
        case .nanojev:
            guard precision == "fp16" else { throw PublishedCoreMLError.invalidPrecision(precision) }
            return ["build/nanojev_encoder_fp16_L128_K4.mlpackage", "build/nanojev_heads_fp16_K4.mlpackage"]
        }
    }
}

public enum PublishedCoreMLError: Error, LocalizedError, Sendable {
    case invalidPrecision(String)
    case missingAsset(String)
    case invalidRequest(String)
    case runtime(String)

    public var errorDescription: String? {
        switch self {
        case .invalidPrecision(let value): "Unsupported published Core ML precision: \(value)"
        case .missingAsset(let value): "Missing published Core ML asset: \(value)"
        case .invalidRequest(let value): "Invalid published Core ML request: \(value)"
        case .runtime(let value): "Published Core ML runtime failed: \(value)"
        }
    }
}

/// Mac-only bridge to the published model-specific Core ML serving code.
///
/// The model packages remain local and run through Core ML. Python performs the released tokenizer,
/// prompt rendering, and decoding contract. Start this once and reuse it; actor isolation serializes
/// requests to the process. Install the selected Hub repository's Python dependencies in the supplied
/// interpreter environment before loading it. The API accepts and returns JSON because Kev, Kai/Lex,
/// LFM, and Jeff expose different native decision schemas.
public actor PublishedCoreMLManager {
    public nonisolated let model: PublishedCoreMLModel
    private let process: Process
    private let input: FileHandle
    private let output: FileHandle
    private var buffer = Data()

    /// Start a serving session from a materialized Hub snapshot. NanoJev may fetch its pinned
    /// upstream tokenizer/source files; it never fetches or redistributes trained weights.
    /// - Parameters:
    ///   - directory: Local repository root containing the selected `.mlpackage` and tokenizer files.
    ///   - python: Python 3.12 executable with the published toolkit's dependencies installed.
    public init(
        model: PublishedCoreMLModel, from directory: URL, python: URL,
        precision: String = "fp16"
    ) throws {
        guard directory.isFileURL, python.isFileURL else {
            throw PublishedCoreMLError.invalidRequest("Local file URLs are required")
        }
        let metadata = model == .nanojev ? "assets.lock.json" : "config.json"
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(metadata).path) else {
            throw PublishedCoreMLError.missingAsset(metadata)
        }
        for path in try model.requiredPackages(precision: precision) {
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(path).path) else {
                throw PublishedCoreMLError.missingAsset(path)
            }
        }
        guard
            let worker = Bundle.module.url(
                forResource: "published-coreml-worker", withExtension: "py", subdirectory: "Resources")
        else { throw PublishedCoreMLError.missingAsset("published-coreml-worker.py") }
        self.model = model
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let task = Process()
        task.executableURL = python
        task.arguments = [worker.path, "--model", model.rawValue, "--root", directory.path, "--precision", precision]
        task.standardInput = inputPipe
        task.standardOutput = outputPipe
        task.standardError = FileHandle.standardError
        self.process = task
        self.input = inputPipe.fileHandleForWriting
        self.output = outputPipe.fileHandleForReading
        do { try task.run() } catch {
            throw PublishedCoreMLError.runtime("Could not start Python: \(error.localizedDescription)")
        }
        var startupBuffer = Data()
        guard let ready = try Self.readMessage(from: outputPipe.fileHandleForReading, buffer: &startupBuffer),
            ready["ready"] as? Bool == true
        else {
            task.terminate()
            throw PublishedCoreMLError.runtime("Worker did not become ready; see standard error")
        }
        self.buffer = startupBuffer
    }

    deinit {
        if process.isRunning { process.terminate() }
    }

    /// Evaluate one request in the model's published JSON schema and return its JSON answer.
    public func evaluate(_ request: Data) throws -> Data {
        try Task.checkCancellation()
        guard process.isRunning else { throw PublishedCoreMLError.runtime("Worker exited") }
        let object = try JSONSerialization.jsonObject(with: request)
        guard object is [String: Any] else {
            throw PublishedCoreMLError.invalidRequest("Top-level JSON must be an object")
        }
        var line = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        line.append(0x0A)
        try input.write(contentsOf: line)
        guard let response = try Self.readMessage(from: output, buffer: &buffer) else {
            throw PublishedCoreMLError.runtime("Worker closed its output")
        }
        if let error = response["error"] as? String { throw PublishedCoreMLError.runtime(error) }
        guard let answer = response["ok"] else { throw PublishedCoreMLError.runtime("Worker returned no answer") }
        return try JSONSerialization.data(withJSONObject: answer, options: [.sortedKeys, .fragmentsAllowed])
    }

    private static func readMessage(from output: FileHandle, buffer: inout Data) throws -> [String: Any]? {
        while true {
            if let index = buffer.firstIndex(of: 0x0A) {
                let line = Data(buffer[..<index])
                buffer.removeSubrange(...index)
                guard let object = try JSONSerialization.jsonObject(with: line) as? [String: Any] else {
                    throw PublishedCoreMLError.runtime("Worker returned non-object JSON")
                }
                return object
            }
            guard buffer.count < 4_194_304 else { throw PublishedCoreMLError.runtime("Worker response exceeded 4 MiB") }
            let chunk = output.availableData
            guard !chunk.isEmpty else { return nil }
            buffer.append(chunk)
        }
    }
}
