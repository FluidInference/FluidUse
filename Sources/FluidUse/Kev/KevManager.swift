@preconcurrency import CoreML
import Foundation

/// One typed question, as in Kev's System One API.
public enum KevQuestion: Sendable {
    /// Pick one of `options` (key, optional description).
    case choice(String, options: [(key: String, description: String?)])
    /// Yes or no, with optional descriptions of each side.
    case noul(String, no: String? = nil, yes: String? = nil)
    /// One of an ordered list of levels.
    case score(String, levels: [String])

    var instruction: String {
        switch self {
        case .choice(let text, _), .noul(let text, _, _), .score(let text, _): text
        }
    }

    /// Option texts exactly as Kev's `api.to_record` renders them.
    public var optionTexts: [String] {
        switch self {
        case .choice(_, let options): options.map { Self.optionText($0.key, $0.description) }
        case .noul(_, let no, let yes): [Self.optionText("no", no), Self.optionText("yes", yes)]
        case .score(_, let levels): levels
        }
    }

    /// Keys the probabilities are reported under: choice keys, ["false", "true"], or level indices.
    public var keys: [String] {
        switch self {
        case .choice(_, let options): options.map(\.key)
        case .noul: ["false", "true"]
        case .score(_, let levels): levels.indices.map(String.init)
        }
    }

    private static func optionText(_ name: String, _ description: String?) -> String {
        guard let description, !description.isEmpty else { return name }
        return "\(name): \(description)"
    }
}

public struct KevAnswer: Sendable {
    public let keys: [String]
    public let probabilities: [Float]
    public let tokens: Int

    public var bestIndex: Int { probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0 }
    public var best: String { keys[bestIndex] }
    public var confidence: Float { probabilities[bestIndex] }
}

public enum KevError: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case tooLong(String)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid Kev asset: \(reason)"
        case .tooLong(let reason): "Kev input too long: \(reason)"
        case .invalidOutput(let reason): "Invalid Kev output: \(reason)"
        }
    }
}

/// Kev (Qwen3.5 backbone + pointer head) on Core ML. Each question runs as its own causal row, state first, exactly
/// as Kev scores its Qwen3.5 checkpoints; the smallest bucket that fits the row is used. Calls are not serialized:
/// Core ML's async prediction is thread-safe, so several questions can be in flight at once.
public final class KevManager: Sendable {
    struct Bucket: Sendable {
        let length: Int
        let maxOptions: Int
        let model: MLModel
    }

    struct Special: Sendable {
        let state: Int
        let question: Int
        let option: Int
        let closeOption: Int
        let decide: Int
    }

    let tokenizer: QwenBPETokenizer
    private let buckets: [Bucket]
    private let embeddings: Data
    private let hiddenSize: Int
    private let rotaryDim: Int
    private let ropeTheta: Double
    private let padID: Int
    let special: Special

    /// `directory` holds `tokenizer.json` and one `L<length>_K<options>/` folder per bucket with `config.json`,
    /// `embeddings.f16` and a `KevRow_*.mlpackage` (or compiled `.mlmodelc`).
    public static func load(from directory: URL, computeUnits: MLComputeUnits = .all) async throws -> KevManager {
        let tokenizer = try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let folders = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .map { $0.resolvingSymlinksInPath() }
            .filter { $0.lastPathComponent.range(of: #"^L\d+_K\d+$"#, options: .regularExpression) != nil }
        guard !folders.isEmpty else { throw KevError.invalidAsset("No L<length>_K<options> bucket folders") }
        var buckets: [Bucket] = []
        var config: [String: Any] = [:]
        for folder in folders {
            guard
                let parsed = try JSONSerialization.jsonObject(
                    with: Data(contentsOf: folder.appendingPathComponent("config.json"))) as? [String: Any]
            else { throw KevError.invalidAsset("Unreadable \(folder.lastPathComponent)/config.json") }
            config = parsed
            let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            let modelURL: URL
            if let compiled = files.first(where: { $0.pathExtension == "mlmodelc" }) {
                modelURL = compiled
            } else if let package = files.first(where: { $0.pathExtension == "mlpackage" }) {
                modelURL = try await MLModel.compileModel(at: package)
            } else {
                throw KevError.invalidAsset("No Core ML model in \(folder.lastPathComponent)")
            }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            buckets.append(
                Bucket(
                    length: parsed["length"] as? Int ?? 0, maxOptions: parsed["max_options"] as? Int ?? 0,
                    model: try await MLModel.load(contentsOf: modelURL, configuration: configuration)))
        }
        guard let hidden = config["hidden_size"] as? Int, let rotary = config["rotary_dim"] as? Int,
            let theta = (config["rope_theta"] as? NSNumber)?.doubleValue, let pad = config["pad_id"] as? Int,
            let vocab = config["vocab_size"] as? Int, let tokens = config["special_tokens"] as? [String: Int],
            let state = tokens["state"], let question = tokens["q"], let option = tokens["opt"],
            let close = tokens["close_opt"], let decide = tokens["decide"]
        else { throw KevError.invalidAsset("config.json is missing Kev fields") }
        guard
            let table = folders.map({ $0.appendingPathComponent("embeddings.f16") })
                .first(where: { FileManager.default.fileExists(atPath: $0.path) })
        else { throw KevError.invalidAsset("no bucket folder has embeddings.f16") }
        let embeddings = try Data(contentsOf: table, options: .alwaysMapped)
        guard embeddings.count == vocab * hidden * 2 else {
            throw KevError.invalidAsset("embeddings.f16 has \(embeddings.count) bytes, expected \(vocab * hidden * 2)")
        }
        return KevManager(
            tokenizer: tokenizer, buckets: buckets.sorted { $0.length < $1.length }, embeddings: embeddings,
            hiddenSize: hidden, rotaryDim: rotary, ropeTheta: theta, padID: pad,
            special: Special(state: state, question: question, option: option, closeOption: close, decide: decide))
    }

    init(
        tokenizer: QwenBPETokenizer, buckets: [Bucket], embeddings: Data, hiddenSize: Int, rotaryDim: Int,
        ropeTheta: Double, padID: Int, special: Special
    ) {
        self.tokenizer = tokenizer
        self.buckets = buckets
        self.embeddings = embeddings
        self.hiddenSize = hiddenSize
        self.rotaryDim = rotaryDim
        self.ropeTheta = ropeTheta
        self.padID = padID
        self.special = special
    }

    /// Kev's training and evaluation state limit (`kev.model.MAX_STATE`): the state marker plus 383 text tokens.
    public static let evaluationMaxStateTokens = 384

    /// Answers every question about `state`; each question is one Core ML call. The state is cut to
    /// `maxStateTokens` (marker included) as Kev's `encode` does; the default matches its published evaluations.
    public func answer(
        state: String, questions: [KevQuestion], maxStateTokens: Int = KevManager.evaluationMaxStateTokens
    ) async throws -> [KevAnswer] {
        let stateIDs = [special.state] + Array(try userTokens(state).prefix(maxStateTokens - 1))
        var answers: [KevAnswer] = []
        for question in questions {
            answers.append(try await answer(stateIDs: stateIDs, question: question))
        }
        return answers
    }

    /// Like `answer(state:questions:)`, but every question is in flight at once (one Core ML call each).
    public func answerConcurrently(
        state: String, questions: [KevQuestion], maxStateTokens: Int = KevManager.evaluationMaxStateTokens
    ) async throws -> [KevAnswer] {
        let stateIDs = [special.state] + Array(try userTokens(state).prefix(maxStateTokens - 1))
        return try await withThrowingTaskGroup(of: (Int, KevAnswer).self) { group in
            for (index, question) in questions.enumerated() {
                group.addTask { (index, try await self.answer(stateIDs: stateIDs, question: question)) }
            }
            var answers = [KevAnswer?](repeating: nil, count: questions.count)
            for try await (index, answer) in group { answers[index] = answer }
            return answers.compactMap { $0 }
        }
    }

    /// Kev's `user_tokens`: caller text can never produce `<|name|>` control tokens.
    func userTokens(_ text: String) throws -> [Int] {
        let escaped = text.replacingOccurrences(
            of: #"<\|([A-Za-z0-9_]+)\|>"#, with: "<¦$1¦>", options: .regularExpression)
        return try tokenizer.encode(escaped)
    }

    func answer(stateIDs: [Int], question: KevQuestion) async throws -> KevAnswer {
        var ids = stateIDs + [special.question] + (try userTokens(question.instruction))
        var optionEnds: [Int] = []
        for text in question.optionTexts {
            ids += [special.option] + (try userTokens(text)) + [special.closeOption]
            optionEnds.append(ids.count - 1)
        }
        ids.append(special.decide)
        let decideIndex = ids.count - 1
        guard let bucket = buckets.first(where: { ids.count <= $0.length && optionEnds.count <= $0.maxOptions }) else {
            throw KevError.tooLong("\(ids.count) tokens / \(optionEnds.count) options exceed every bucket")
        }
        let features = try inputs(ids: ids, decide: decideIndex, options: optionEnds, bucket: bucket)
        let output = try await bucket.model.prediction(from: features)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw KevError.invalidOutput("missing logits")
        }
        let count = optionEnds.count
        var values = (0..<count).map { Float(truncating: logits[$0]) }
        let peak = values.max() ?? 0
        values = values.map { exp($0 - peak) }
        let total = values.reduce(0, +)
        return KevAnswer(keys: question.keys, probabilities: values.map { $0 / total }, tokens: ids.count)
    }

    private func inputs(ids: [Int], decide: Int, options: [Int], bucket: Bucket) throws -> MLDictionaryFeatureProvider {
        let length = bucket.length
        let hidden = try MLMultiArray(
            shape: [1, NSNumber(value: length), NSNumber(value: hiddenSize)], dataType: .float32)
        let hiddenPointer = hidden.dataPointer.assumingMemoryBound(to: Float.self)
        embeddings.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self)
            for position in 0..<length {
                let token = position < ids.count ? ids[position] : padID
                let row = token * hiddenSize
                let destination = hiddenPointer + position * hiddenSize
                for column in 0..<hiddenSize {
                    destination[column] = Float(table[row + column])
                }
            }
        }
        let half = rotaryDim / 2
        let cos = try MLMultiArray(shape: [NSNumber(value: length), NSNumber(value: rotaryDim)], dataType: .float32)
        let sin = try MLMultiArray(shape: [NSNumber(value: length), NSNumber(value: rotaryDim)], dataType: .float32)
        let cosPointer = cos.dataPointer.assumingMemoryBound(to: Float.self)
        let sinPointer = sin.dataPointer.assumingMemoryBound(to: Float.self)
        for position in 0..<length {
            for i in 0..<half {
                let frequency = Double(position) / pow(ropeTheta, Double(2 * i) / Double(rotaryDim))
                let c = Float(Foundation.cos(frequency))
                let s = Float(Foundation.sin(frequency))
                cosPointer[position * rotaryDim + i] = c
                cosPointer[position * rotaryDim + i + half] = c
                sinPointer[position * rotaryDim + i] = s
                sinPointer[position * rotaryDim + i + half] = s
            }
        }
        let decideOneHot = try zeros([1, length])
        decideOneHot.dataPointer.assumingMemoryBound(to: Float.self)[decide] = 1
        let optionOneHot = try zeros([bucket.maxOptions, length])
        let optionMask = try zeros([bucket.maxOptions])
        let onePointer = optionOneHot.dataPointer.assumingMemoryBound(to: Float.self)
        let maskPointer = optionMask.dataPointer.assumingMemoryBound(to: Float.self)
        for (slot, index) in options.enumerated() {
            onePointer[slot * length + index] = 1
            maskPointer[slot] = 1
        }
        return try MLDictionaryFeatureProvider(dictionary: [
            "hidden": MLFeatureValue(multiArray: hidden), "cos": MLFeatureValue(multiArray: cos),
            "sin": MLFeatureValue(multiArray: sin), "decide_onehot": MLFeatureValue(multiArray: decideOneHot),
            "option_onehot": MLFeatureValue(multiArray: optionOneHot),
            "option_mask": MLFeatureValue(multiArray: optionMask),
        ])
    }

    private func zeros(_ shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        array.dataPointer.initializeMemory(as: Float.self, repeating: 0, count: array.count)
        return array
    }
}
