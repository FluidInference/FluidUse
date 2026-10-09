@preconcurrency import CoreML
import Foundation

/// Learned sparse retrieval with Granite-Embedding-30M-Sparse (Evoke P2.2) on Core ML: text -> weighted vocabulary
/// terms, including related terms the text never uses. Each loaded sequence length is its own fixed-shape package;
/// `terms(for:kind:)` picks the smallest that fits and truncates beyond the largest. Prediction uses Core ML's async
/// API, so callers may keep several texts in flight.
public final class EvokeManager: Sendable {
    public let config: EvokeConfig
    public let tokenizer: QwenBPETokenizer

    /// Ascending by length.
    private let models: [(length: Int, model: MLModel)]
    private let bosId = 0
    private let eosId = 2
    private let padId: Int32 = 1

    public init(config: EvokeConfig, tokenizer: QwenBPETokenizer, models: [Int: MLModel]) {
        self.config = config
        self.tokenizer = tokenizer
        self.models = models.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    /// Downloads (once, checksum-verified) and loads the published packages for `lengths`.
    /// The Neural Engine is fastest up to 128 tokens; the GPU wins at 256 and 512.
    public static func load(
        lengths: [Int] = [64], cacheDirectory: URL? = nil, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progress: EvokeModelStore.Progress? = nil
    ) async throws -> EvokeManager {
        let directory = try await EvokeModelStore.ensure(
            lengths: lengths, cacheDirectory: cacheDirectory, progress: progress)
        return try await load(from: directory, lengths: lengths, computeUnits: computeUnits)
    }

    /// Loads from `EVOKE_MODEL_DIR` when set, otherwise from the published packages.
    public static func loadDefault(
        lengths: [Int] = [64], computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progress: EvokeModelStore.Progress? = nil
    ) async throws -> EvokeManager {
        if let path = ProcessInfo.processInfo.environment["EVOKE_MODEL_DIR"], !path.isEmpty {
            return try await load(from: URL(fileURLWithPath: path), lengths: lengths, computeUnits: computeUnits)
        }
        return try await load(lengths: lengths, computeUnits: computeUnits, progress: progress)
    }

    /// Loads `config.json`, `tokenizer.json`, and the packages for `lengths` (`.mlmodelc` preferred) from `directory`.
    public static func load(
        from directory: URL, lengths: [Int] = [64], computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> EvokeManager {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")) else {
            throw EvokeError.invalidAsset("Missing config.json in \(directory.path)")
        }
        let config = try JSONDecoder().decode(EvokeConfig.self, from: data)
        let tokenizer = try QwenBPETokenizer(
            tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"), preTokenizer: .gpt2)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        var models: [Int: MLModel] = [:]
        for length in lengths {
            guard config.sequenceLengths.contains(length) else {
                throw EvokeError.invalidAsset("config.json has no length \(length)")
            }
            models[length] = try await loadModel(
                named: EvokeModelStore.packageName(length: length), in: directory, configuration)
        }
        guard !models.isEmpty else { throw EvokeError.invalidAsset("No sequence lengths requested") }
        return EvokeManager(config: config, tokenizer: tokenizer, models: models)
    }

    private static func loadModel(
        named name: String, in directory: URL, _ configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiled = directory.appendingPathComponent("\(name).mlmodelc")
        let package = directory.appendingPathComponent("\(name).mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            url = try await MLModel.compileModel(at: package)
        } else {
            throw EvokeError.invalidAsset("Missing \(name).mlmodelc or .mlpackage in \(directory.path)")
        }
        return try await MLModel.load(contentsOf: url, configuration: configuration)
    }

    /// `<s>` + BPE ids + `</s>`, truncated to the longest loaded package.
    public func tokenize(_ text: String) throws -> [Int] {
        let maxLength = models.last?.length ?? 0
        var ids = [bosId] + (try tokenizer.encode(text)) + [eosId]
        if ids.count > maxLength { ids = Array(ids.prefix(maxLength - 1)) + [eosId] }
        return ids
    }

    /// Weighted terms for `text`. Score a query against documents with `EvokeTerms.score`.
    public func terms(for text: String, kind: EvokeTextKind) async throws -> EvokeTerms {
        try await terms(tokenIds: tokenize(text), kind: kind).terms
    }

    /// Terms plus the wall time of the Core ML call alone, in milliseconds.
    public func terms(
        tokenIds ids: [Int], kind: EvokeTextKind
    ) async throws -> (terms: EvokeTerms, predictionMs: Double) {
        guard let (length, model) = models.first(where: { $0.length >= ids.count }) ?? models.last else {
            throw EvokeError.predictionFailed("No model loaded")
        }
        let input = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let pointer = input.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<length { pointer[index] = index < ids.count ? Int32(ids[index]) : padId }
        let features = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: input)])
        let start = DispatchTime.now().uptimeNanoseconds
        let output = try await model.prediction(from: features)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        guard let values = output.featureValue(for: "max_logits")?.multiArrayValue,
            let vocab = output.featureValue(for: "vocab_ids")?.multiArrayValue
        else { throw EvokeError.predictionFailed("Missing max_logits / vocab_ids") }
        let logits = (0..<values.count).map { values[$0].floatValue }
        let vocabIds = (0..<vocab.count).map { vocab[$0].intValue }
        return (config.transform(for: kind).terms(maxLogits: logits, vocabIds: vocabIds), elapsed)
    }

    /// Readable form of a vocabulary id ("Ġelderly" -> "elderly"); nil for special tokens and sub-word fragments
    /// that are not letters or digits.
    public func word(for id: Int) -> String? {
        let word = tokenizer.decode([id]).trimmingCharacters(in: .whitespaces).lowercased()
        guard word.count > 1, word.allSatisfy({ $0.isLetter || $0.isNumber }) else { return nil }
        return word
    }
}
