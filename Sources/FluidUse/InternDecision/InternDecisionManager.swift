@preconcurrency import CoreML
import Foundation

/// One typed question for Intern-Decision, as in its `questions` object.
public enum InternDecisionQuestion: Sendable {
    /// Pick one of `options` (label, description); labels are reported in this order.
    case choice(String, options: [(label: String, description: String)])
    /// Yes or no, with optional descriptions of each side. Reported under `no`, `yes`, in that order.
    case noul(String, no: String? = nil, yes: String? = nil)
    /// One of an ordered list of levels; labels are `"0"`, `"1"`, … (a `criteria` list).
    case score(String, levels: [String])
    /// Ordered numeric labels with descriptions (a `criteria` object such as `{"0": "none", "0.5": "some"}`).
    case scoreKeyed(String, levels: [(label: String, description: String)])

    public var instructions: String {
        switch self {
        case .choice(let text, _), .noul(let text, _, _), .score(let text, _), .scoreKeyed(let text, _): text
        }
    }

    /// `(label, description)` pairs exactly as the checkpoint's `inference._options` orders them.
    public var options: [(label: String, description: String)] {
        switch self {
        case .choice(_, let options): options
        case .noul(_, let no, let yes):
            [
                (label: "no", description: no ?? "The answer is no (negative, or disagree with the claim)."),
                (label: "yes", description: yes ?? "The answer is yes (affirmative, or align with the claim)."),
            ]
        case .score(_, let levels): levels.enumerated().map { (label: String($0.offset), description: $0.element) }
        case .scoreKeyed(_, let levels): levels
        }
    }

    var isScore: Bool {
        switch self {
        case .score, .scoreKeyed: true
        default: false
        }
    }
}

public struct InternDecisionAnswer: Sendable {
    public let field: String
    public let labels: [String]
    /// Restricted softmax over the field's answer symbols, after the checkpoint's temperature.
    public let probabilities: [Float]
    /// The same before temperature scaling.
    public let rawProbabilities: [Float]

    /// Highest probability; ties go to the smaller label, as the reference `argmax` does.
    public var bestIndex: Int {
        var best = 0
        for index in labels.indices.dropFirst() {
            if probabilities[index] > probabilities[best]
                || (probabilities[index] == probabilities[best]
                    && labels[index].unicodeScalars.lexicographicallyPrecedes(labels[best].unicodeScalars))
            {
                best = index
            }
        }
        return best
    }
    public var decision: String { labels[bestIndex] }
    public var confidence: Float { probabilities[bestIndex] }
    /// Probability of `yes` for a yes/no question.
    public var yes: Float? { labels == ["no", "yes"] ? probabilities[1] : nil }
    /// Expected numeric value for a score question (labels are numbers).
    public var expectedScore: Double? {
        var total = 0.0
        for (label, p) in zip(labels, probabilities) {
            guard let value = Double(label) else { return nil }
            total += value * Double(p)
        }
        return total
    }
}

public struct InternDecisionResult: Sendable {
    public let answers: [InternDecisionAnswer]
    public let inputTokens: Int
    public let bucketLength: Int
}

public enum InternDecisionError: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case invalidRequest(String)
    case tooLong(String)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid Intern-Decision asset: \(reason)"
        case .invalidRequest(let reason): "Invalid Intern-Decision request: \(reason)"
        case .tooLong(let reason): "Intern-Decision input too long: \(reason)"
        case .invalidOutput(let reason): "Invalid Intern-Decision output: \(reason)"
        }
    }
}

/// Intern-Decision-0.8B (Qwen3.5 backbone, `<decision>` marker readout) on Core ML. A request is one prompt with
/// every question; the logits before each marker, restricted to that field's answer symbols, are its answer. One
/// Core ML call per request, in the smallest bucket that fits.
public final class InternDecisionManager: Sendable {
    public static let decisionToken = "<decision>"
    public static let maxQuestions = 16
    static let userPreamble = "Return one answer for every field using the supplied answer symbols.\n\n## State\n"
    static let controlTokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]

    struct Bucket: Sendable {
        let length: Int
        let maxFields: Int
        let url: URL
    }

    actor Models {
        private let computeUnits: MLComputeUnits
        /// One load per bucket even when several callers miss the cache at once.
        private var loads: [Int: Task<MLModel, Error>] = [:]

        init(computeUnits: MLComputeUnits) { self.computeUnits = computeUnits }

        func model(for bucket: Bucket) async throws -> MLModel {
            if let load = loads[bucket.length] { return try await load.value }
            let units = computeUnits
            let load = Task<MLModel, Error> {
                let url =
                    bucket.url.pathExtension == "mlpackage" ? try await KevManager.compiled(bucket.url) : bucket.url
                let configuration = MLModelConfiguration()
                configuration.computeUnits = units
                return try await MLModel.load(contentsOf: url, configuration: configuration)
            }
            loads[bucket.length] = load
            do {
                return try await load.value
            } catch {
                loads[bucket.length] = nil
                throw error
            }
        }
    }

    public let tokenizer: QwenBPETokenizer
    public let systemPrompt: String
    public let temperature: Float
    let symbols: [Character]
    private let buckets: [Bucket]
    private let models: Models
    private let embeddings: Data
    private let hiddenSize: Int
    private let rotaryDim: Int
    private let ropeTheta: Double
    private let padID: Int
    private let markerID: Int
    /// RoPE tables per bucket length; they depend only on the length, theta and rotary dim. Core ML never mutates
    /// its inputs, so one pair is shared by every call.
    private let rope: [Int: (cos: MLMultiArray, sin: MLMultiArray)]

    /// `directory` holds `tokenizer.json`, `embeddings.f16` and one `L<length>_F<fields>/` folder per bucket with
    /// `config.json` and a `DecisionRow_*.mlpackage` (or compiled `.mlmodelc`); the layout of
    /// `FluidInference/intern-decision-0.8b-coreml`.
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU, eager: Bool = true
    ) async throws -> InternDecisionManager {
        let tokenizer = try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let folders = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .map { $0.resolvingSymlinksInPath() }
            .filter { $0.lastPathComponent.range(of: #"^L\d+_F\d+$"#, options: .regularExpression) != nil }
        guard !folders.isEmpty else { throw InternDecisionError.invalidAsset("No L<length>_F<fields> bucket folders") }
        var buckets: [Bucket] = []
        var config: [String: Any] = [:]
        for folder in folders {
            guard
                let parsed = try JSONSerialization.jsonObject(
                    with: Data(contentsOf: folder.appendingPathComponent("config.json"))) as? [String: Any]
            else { throw InternDecisionError.invalidAsset("Unreadable \(folder.lastPathComponent)/config.json") }
            config = parsed
            let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            let preferred = files.filter { $0.lastPathComponent.hasPrefix("DecisionRow_fp16") }
            guard
                let modelURL = preferred.first(where: { $0.pathExtension == "mlmodelc" })
                    ?? preferred.first(where: { $0.pathExtension == "mlpackage" })
                    ?? files.first(where: { $0.pathExtension == "mlmodelc" })
                    ?? files.first(where: { $0.pathExtension == "mlpackage" })
            else { throw InternDecisionError.invalidAsset("No Core ML model in \(folder.lastPathComponent)") }
            buckets.append(
                Bucket(
                    length: parsed["length"] as? Int ?? 0, maxFields: parsed["max_fields"] as? Int ?? 0, url: modelURL))
        }
        guard let hidden = config["hidden_size"] as? Int, let rotary = config["rotary_dim"] as? Int,
            let theta = (config["rope_theta"] as? NSNumber)?.doubleValue, let pad = config["pad_id"] as? Int,
            let vocab = config["vocab_size"] as? Int, let marker = config["marker_id"] as? Int,
            let symbols = config["symbols"] as? String, let symbolIDs = config["symbol_ids"] as? [Int],
            let temperature = (config["temperature"] as? NSNumber)?.floatValue,
            let system = config["system_prompt"] as? String
        else { throw InternDecisionError.invalidAsset("config.json is missing Intern-Decision fields") }
        guard symbols.count == symbolIDs.count, tokenizer.id(for: decisionToken) == marker else {
            throw InternDecisionError.invalidAsset("tokenizer and config disagree on the decision token")
        }
        let table = directory.appendingPathComponent("embeddings.f16")
        let embeddings = try Data(contentsOf: table, options: .alwaysMapped)
        guard embeddings.count == vocab * hidden * 2 else {
            throw InternDecisionError.invalidAsset(
                "embeddings.f16 has \(embeddings.count) bytes, expected \(vocab * hidden * 2)")
        }
        let models = Models(computeUnits: computeUnits)
        if eager {
            for bucket in buckets { _ = try await models.model(for: bucket) }
        }
        return try InternDecisionManager(
            tokenizer: tokenizer, systemPrompt: system, temperature: temperature, symbols: Array(symbols),
            buckets: buckets.sorted { $0.length < $1.length }, models: models, embeddings: embeddings,
            hiddenSize: hidden, rotaryDim: rotary, ropeTheta: theta, padID: pad, markerID: marker)
    }

    init(
        tokenizer: QwenBPETokenizer, systemPrompt: String, temperature: Float, symbols: [Character],
        buckets: [Bucket], models: Models, embeddings: Data, hiddenSize: Int, rotaryDim: Int, ropeTheta: Double,
        padID: Int, markerID: Int
    ) throws {
        self.tokenizer = tokenizer
        self.systemPrompt = systemPrompt
        self.temperature = temperature
        self.symbols = symbols
        self.buckets = buckets
        self.models = models
        self.embeddings = embeddings
        self.hiddenSize = hiddenSize
        self.rotaryDim = rotaryDim
        self.ropeTheta = ropeTheta
        self.padID = padID
        self.markerID = markerID
        var rope: [Int: (cos: MLMultiArray, sin: MLMultiArray)] = [:]
        for bucket in buckets {
            rope[bucket.length] = try Self.ropeTables(length: bucket.length, rotaryDim: rotaryDim, theta: ropeTheta)
        }
        self.rope = rope
    }

    static func ropeTables(length: Int, rotaryDim: Int, theta: Double) throws -> (cos: MLMultiArray, sin: MLMultiArray)
    {
        let half = rotaryDim / 2
        let cos = try MLMultiArray(shape: [NSNumber(value: length), NSNumber(value: rotaryDim)], dataType: .float32)
        let sin = try MLMultiArray(shape: [NSNumber(value: length), NSNumber(value: rotaryDim)], dataType: .float32)
        let cosPointer = cos.dataPointer.assumingMemoryBound(to: Float.self)
        let sinPointer = sin.dataPointer.assumingMemoryBound(to: Float.self)
        for position in 0..<length {
            for i in 0..<half {
                let frequency = Double(position) / pow(theta, Double(2 * i) / Double(rotaryDim))
                let c = Float(Foundation.cos(frequency))
                let s = Float(Foundation.sin(frequency))
                cosPointer[position * rotaryDim + i] = c
                cosPointer[position * rotaryDim + i + half] = c
                sinPointer[position * rotaryDim + i] = s
                sinPointer[position * rotaryDim + i + half] = s
            }
        }
        return (cos, sin)
    }

    /// Largest request any bucket accepts.
    public var maxTokens: Int { buckets.last?.length ?? 0 }

    /// The checkpoint's `compile_row` + chat template (`enable_thinking=False`), as one string.
    public func prompt(
        state: OrderedJSON, questions: [(name: String, question: InternDecisionQuestion)]
    ) throws -> String {
        let fields = try Self.validated(questions, symbolCount: symbols.count)
        var schema: [String] = []
        for (name, question) in fields {
            schema.append("\(name): \(question.instructions)")
            for (symbol, option) in zip(symbols, question.options) {
                schema.append("    \(symbol) = \(option.label): \(option.description)")
            }
        }
        let user =
            Self.userPreamble + state.pythonDump(indent: 2) + "\n## Decision schema\n" + schema.joined(separator: "\n")
        guard !user.contains(Self.decisionToken) else {
            throw InternDecisionError.invalidRequest("Reserved decision marker appears in input evidence")
        }
        // The tokenizer matches added tokens literally, as the reference does; refusing them keeps caller data from
        // closing the user turn (the reference compiler only checks the decision marker).
        if let token = Self.controlTokens.first(where: { user.contains($0) }) {
            throw InternDecisionError.invalidRequest("Control token \(token) appears in input evidence")
        }
        let skeleton = OrderedJSON.object(fields.map { (key: $0.name, value: .string(Self.decisionToken)) })
            .pythonDump(indent: 4)
        return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n\(skeleton)<|im_end|>\n"
    }

    static func validated(
        _ questions: [(name: String, question: InternDecisionQuestion)], symbolCount: Int
    ) throws
        -> [(name: String, question: InternDecisionQuestion)]
    {
        guard (1...maxQuestions).contains(questions.count) else {
            throw InternDecisionError.invalidRequest("Supply 1–\(maxQuestions) questions.")
        }
        var seen = Set<String>()
        for (name, question) in questions {
            guard !name.isEmpty, seen.insert(name).inserted else {
                throw InternDecisionError.invalidRequest("Question names must be nonempty and unique.")
            }
            let count = question.options.count
            guard (1...symbolCount).contains(count) else {
                throw InternDecisionError.invalidRequest("Supply 1–\(symbolCount) options for \(name).")
            }
            if question.isScore {
                guard question.options.allSatisfy({ Double($0.label).map(\.isFinite) ?? false }) else {
                    throw InternDecisionError.invalidRequest("Score option keys must be finite numbers.")
                }
            }
        }
        return questions
    }

    /// Token ids of the rendered prompt and, per field, the index of the token before its `<decision>` marker.
    public func encode(
        state: OrderedJSON, questions: [(name: String, question: InternDecisionQuestion)]
    ) throws
        -> (ids: [Int], positions: [Int])
    {
        let ids = try tokenizer.encode(try prompt(state: state, questions: questions))
        let positions = ids.indices.filter { ids[$0] == markerID }.map { $0 - 1 }
        guard positions.count == questions.count, positions.allSatisfy({ $0 >= 0 }) else {
            throw InternDecisionError.invalidRequest("Decision marker count or position mismatch")
        }
        return (ids, positions)
    }

    /// Answers every question about `state` in one Core ML call.
    public func decide(
        state: OrderedJSON, questions: [(name: String, question: InternDecisionQuestion)]
    ) async throws
        -> InternDecisionResult
    {
        let (ids, positions) = try encode(state: state, questions: questions)
        guard let bucket = buckets.first(where: { ids.count <= $0.length && positions.count <= $0.maxFields }) else {
            throw InternDecisionError.tooLong("\(ids.count) tokens / \(positions.count) fields exceed every bucket")
        }
        let features = try inputs(ids: ids, positions: positions, bucket: bucket)
        let output = try await models.model(for: bucket).prediction(from: features)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue, logits.shape.count == 2,
            logits.shape[1].intValue == symbols.count
        else { throw InternDecisionError.invalidOutput("missing logits [fields, symbols]") }
        let stride = symbols.count
        var answers: [InternDecisionAnswer] = []
        for (index, (name, question)) in questions.enumerated() {
            let count = question.options.count
            let row = (0..<count).map { Float(truncating: logits[index * stride + $0]) }
            answers.append(
                InternDecisionAnswer(
                    field: name, labels: question.options.map(\.label), probabilities: Self.softmax(row, temperature),
                    rawProbabilities: Self.softmax(row, 1)))
        }
        return InternDecisionResult(answers: answers, inputTokens: ids.count, bucketLength: bucket.length)
    }

    /// `softmax(logits / T)`, which equals the reference's `softmax(log softmax(logits) / T)`.
    static func softmax(_ logits: [Float], _ temperature: Float) -> [Float] {
        let scaled = logits.map { $0 / temperature }
        let peak = scaled.max() ?? 0
        let weights = scaled.map { exp($0 - peak) }
        let total = weights.reduce(0, +)
        return weights.map { $0 / total }
    }

    private func inputs(ids: [Int], positions: [Int], bucket: Bucket) throws -> MLDictionaryFeatureProvider {
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
        guard let (cos, sin) = rope[length] else {
            throw InternDecisionError.invalidAsset("no RoPE table for \(length)")
        }
        let fieldOneHot = try MLMultiArray(
            shape: [NSNumber(value: bucket.maxFields), NSNumber(value: length)], dataType: .float32)
        let onePointer = fieldOneHot.dataPointer.assumingMemoryBound(to: Float.self)
        onePointer.initialize(repeating: 0, count: fieldOneHot.count)
        for (slot, index) in positions.enumerated() {
            onePointer[slot * length + index] = 1
        }
        return try MLDictionaryFeatureProvider(dictionary: [
            "hidden": MLFeatureValue(multiArray: hidden), "cos": MLFeatureValue(multiArray: cos),
            "sin": MLFeatureValue(multiArray: sin), "field_onehot": MLFeatureValue(multiArray: fieldOneHot),
        ])
    }
}

extension InternDecisionQuestion {
    /// A question from its JSON form (`type`, `instructions`, `criteria`), as the HTTP interface and the benchmark
    /// records write it.
    public init(json: OrderedJSON) throws {
        guard case .object(let members) = json else {
            throw InternDecisionError.invalidRequest("question must be an object")
        }
        func member(_ key: String) -> OrderedJSON? { members.first { $0.key == key }?.value }
        let instructions: String
        if case .string(let text)? = member("instructions") { instructions = text } else { instructions = "" }
        guard case .string(let type)? = member("type") else {
            throw InternDecisionError.invalidRequest("question type must be choice, score, or noul")
        }
        let criteria = member("criteria")
        switch type {
        case "choice":
            guard case .object(let options)? = criteria else {
                throw InternDecisionError.invalidRequest("choice criteria must be an object")
            }
            self = .choice(instructions, options: options.map { (label: $0.key, description: Self.text($0.value)) })
        case "score":
            if case .array(let levels)? = criteria {
                self = .score(instructions, levels: levels.map(Self.text))
            } else if case .object(let levels)? = criteria {
                self = .scoreKeyed(
                    instructions, levels: levels.map { (label: $0.key, description: Self.text($0.value)) })
            } else {
                throw InternDecisionError.invalidRequest("score criteria must be a list or object")
            }
        case "noul":
            var no: String?
            var yes: String?
            if case .object(let descriptions)? = criteria {
                for (key, value) in descriptions {
                    switch key.lowercased() {
                    case "yes", "true", "1": yes = yes ?? Self.text(value)
                    case "no", "false", "0": no = no ?? Self.text(value)
                    default: break
                    }
                }
            }
            self = .noul(instructions, no: no, yes: yes)
        default:
            throw InternDecisionError.invalidRequest("unsupported question type \(type)")
        }
    }

    /// Python `str(value)` for the JSON values that appear as option descriptions.
    static func text(_ value: OrderedJSON) -> String { value.pythonStr }
}
