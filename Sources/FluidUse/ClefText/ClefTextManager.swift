import CoreML
import Foundation

/// clef-text-0.6b on Core ML: a Qwen3-0.6B text decision model distilled from Cloudflare clef-flash (9B), same
/// Clef / SystemOne contract, running on the Neural Engine. One decoder package (a function per sequence bucket)
/// and the joint schema head, both fp16. An actor: Core ML's synchronous `prediction` is not thread-safe.
@available(macOS 15.0, iOS 18.0, *)
public actor ClefTextManager {
    struct Config: Sendable {
        let hidden: Int
        let headDim: Int
        let ropeTheta: Double
        let padID: Int
        let vocab: Int
        let buckets: [Int]
        let maxQuestions: Int
        let maxOptions: Int
        let decoder: String
        let head: String

        init(json: [String: Any]) throws {
            func int(_ key: String) throws -> Int {
                guard let v = (json[key] as? NSNumber)?.intValue else {
                    throw ClefVisionError.invalidAsset("config.json: \(key)")
                }
                return v
            }
            guard let buckets = json["buckets"] as? [NSNumber], let decoder = json["decoder"] as? String,
                let head = json["head"] as? String, let theta = (json["rope_theta"] as? NSNumber)?.doubleValue
            else { throw ClefVisionError.invalidAsset("config.json is missing sections") }
            hidden = try int("hidden_size")
            headDim = try int("head_dim")
            ropeTheta = theta
            padID = try int("pad_id")
            vocab = try int("vocab_size")
            self.buckets = buckets.map(\.intValue).sorted()
            maxQuestions = try int("max_questions")
            maxOptions = try int("max_options")
            self.decoder = decoder
            self.head = head
        }
    }

    public nonisolated let buckets: [Int]
    let config: Config
    let tokenizer: QwenBPETokenizer
    private let decoders: [Int: MLModel]
    private let heads: [Int: MLModel]
    private let embeddings: Data  // fp16 [vocab, hidden]; tied, so also the head's lexical option rows
    private var ropeCache: [Int: (MLMultiArray, MLMultiArray)] = [:]

    /// Loads the given buckets (records up to the largest one). Packages are compiled once into `compiled/`.
    public static func load(
        from directory: URL, buckets: [Int] = [256, 512], computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> ClefTextManager {
        let json = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let dict = json as? [String: Any] else { throw ClefVisionError.invalidAsset("config.json") }
        let config = try Config(json: dict)
        let tokenizer = try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let embeddings = try Data(
            contentsOf: directory.appendingPathComponent("embeddings.f16"), options: .alwaysMapped)
        guard embeddings.count == config.vocab * config.hidden * 2 else {
            throw ClefVisionError.invalidAsset("embeddings.f16 size")
        }
        let compiledDir = directory.appendingPathComponent("compiled")
        try FileManager.default.createDirectory(at: compiledDir, withIntermediateDirectories: true)
        func compiled(_ path: String) async throws -> URL {
            let source = directory.appendingPathComponent(path)
            let target = compiledDir.appendingPathComponent(
                source.deletingPathExtension().lastPathComponent + ".mlmodelc")
            if !FileManager.default.fileExists(atPath: target.path) {
                let temporary = try await MLModel.compileModel(at: source)
                try FileManager.default.moveItem(at: temporary, to: target)
            }
            return target
        }
        let decoderURL = try await compiled(config.decoder)
        let headURL = try await compiled(config.head)
        var decoders: [Int: MLModel] = [:]
        var heads: [Int: MLModel] = [:]
        for bucket in buckets {
            guard config.buckets.contains(bucket) else {
                throw ClefVisionError.invalidInput("no \(bucket)-token bucket")
            }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            configuration.functionName = "L\(bucket)"
            decoders[bucket] = try await MLModel.load(contentsOf: decoderURL, configuration: configuration)
            heads[bucket] = try await MLModel.load(contentsOf: headURL, configuration: configuration)
        }
        return ClefTextManager(
            config: config, tokenizer: tokenizer, decoders: decoders, heads: heads, embeddings: embeddings)
    }

    init(config: Config, tokenizer: QwenBPETokenizer, decoders: [Int: MLModel], heads: [Int: MLModel], embeddings: Data)
    {
        self.config = config
        self.tokenizer = tokenizer
        self.decoders = decoders
        self.heads = heads
        self.embeddings = embeddings
        self.buckets = decoders.keys.sorted()
    }

    /// Plain RoPE tables [L, head_dim] fp16 (rotate-half layout: the two halves repeat).
    static func ropeTables(length: Int, headDim: Int, theta: Double) throws -> (MLMultiArray, MLMultiArray) {
        let half = headDim / 2
        let cosArray = try MLMultiArray(shape: [length as NSNumber, headDim as NSNumber], dataType: .float16)
        let sinArray = try MLMultiArray(shape: [length as NSNumber, headDim as NSNumber], dataType: .float16)
        let c = cosArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sinArray.dataPointer.assumingMemoryBound(to: Float16.self)
        for position in 0..<length {
            for i in 0..<half {
                let angle = Double(position) / pow(theta, Double(2 * i) / Double(headDim))
                let cv = Float16(Foundation.cos(angle))
                let sv = Float16(Foundation.sin(angle))
                c[position * headDim + i] = cv
                c[position * headDim + half + i] = cv
                s[position * headDim + i] = sv
                s[position * headDim + half + i] = sv
            }
        }
        return (cosArray, sinArray)
    }

    /// Additive causal mask [1, 1, L, L] fp16 for `n` real tokens right-padded to `L` (padded keys masked out).
    static func mask(length: Int, realTokens n: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 1, length as NSNumber, length as NSNumber], dataType: .float16)
        let m = array.dataPointer.assumingMemoryBound(to: Float16.self)
        let blocked = Float16(-1e4)
        for query in 0..<length {
            for key in 0..<length {
                let visible = query < n ? (key <= query) : (key < n || (key >= n && key <= query))
                m[query * length + key] = visible ? 0 : blocked
            }
        }
        return array
    }

    public func warm() throws {
        for bucket in buckets {
            let filler = String(repeating: "warm ", count: max(1, bucket / 4))
            _ = try? answer(state: filler, questions: [("ready", .noul(instructions: "Ready?"))])
        }
    }

    public func encode(state: Any, questions: [(id: String, question: ClefQuestion)]) throws -> ClefEncodedRecord {
        try ClefRecordEncoder.encode(
            state: state, questions: questions, imageTokenCounts: [], tokenizer: tokenizer, imageTokenID: -1,
            visionStartID: -1, visionEndID: -1, maxLength: buckets.max() ?? 0)
    }

    /// Decide `questions` about `state`. Answers follow the question order; probabilities follow Clef's option order.
    public func answer(state: Any, questions: [(id: String, question: ClefQuestion)]) throws -> ClefFlashManager.Result
    {
        let started = Date()
        let record = try encode(state: state, questions: questions)
        let n = record.inputIDs.count
        guard let L = buckets.first(where: { $0 >= n }), let decoder = decoders[L], let head = heads[L] else {
            throw ClefVisionError.tooLong(tokens: n, bucket: buckets.max() ?? 0)
        }
        let D = config.hidden
        if ropeCache[L] == nil {
            ropeCache[L] = try Self.ropeTables(length: L, headDim: config.headDim, theta: config.ropeTheta)
        }
        let (cos, sin) = ropeCache[L]!
        let row = try MLMultiArray(shape: [1, L as NSNumber, D as NSNumber], dataType: .float16)
        let padID = config.padID
        embeddings.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self).baseAddress!
            let out = row.dataPointer.assumingMemoryBound(to: Float16.self)
            for position in 0..<L {
                let token = position < n ? record.inputIDs[position] : padID
                (out + position * D).update(from: table + token * D, count: D)
            }
        }
        let decoderStarted = Date()
        let output = try decoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "hidden": row, "cos": cos, "sin": sin, "mask": try Self.mask(length: L, realTokens: n),
            ]))
        guard let hidden = output.featureValue(for: "states")?.multiArrayValue else {
            throw ClefVisionError.invalidAsset("decoder returned no states")
        }
        let decoderMs = Date().timeIntervalSince(decoderStarted) * 1000
        let headStarted = Date()
        let states = try MLMultiArray(shape: [L as NSNumber, D as NSNumber], dataType: .float32)
        let dst = states.dataPointer.assumingMemoryBound(to: Float.self)
        let rowStride = hidden.strides[1].intValue
        let colStride = hidden.strides[2].intValue
        let src = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        for r in 0..<L {
            for c in 0..<D { dst[r * D + c] = Float(src[r * rowStride + c * colStride]) }
        }
        let (headInputs, optionCount) = try embeddings.withUnsafeBytes { raw in
            try ClefVisionHost.headInputs(
                record: record, states: states, length: L, maxQ: config.maxQuestions, maxO: config.maxOptions,
                embeddings: raw.bindMemory(to: Float16.self), hidden: D)
        }
        let headOutput = try head.prediction(from: headInputs)
        guard let logits = headOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw ClefVisionError.invalidAsset("head returned no logits")
        }
        let headMs = Date().timeIntervalSince(headStarted) * 1000
        var answers: [ClefAnswer] = []
        var offset = 0
        let stride = logits.strides[0].intValue
        let pointer = logits.dataPointer
        for question in record.questions {
            let count = question.optionIDs.count
            let values: [Float] = (0..<count).map { k in
                let index = (offset + k) * stride
                return logits.dataType == .float16
                    ? Float(pointer.assumingMemoryBound(to: Float16.self)[index])
                    : pointer.assumingMemoryBound(to: Float.self)[index]
            }
            let peak = values.max() ?? 0
            let exps = values.map { expf($0 - peak) }
            let sum = exps.reduce(0, +)
            answers.append(
                .init(
                    questionID: question.id, optionIDs: question.optionIDs, logits: values,
                    probabilities: exps.map { $0 / sum }))
            offset += count
        }
        precondition(offset == optionCount)
        return ClefFlashManager.Result(
            answers: answers, inputTokens: n, bucket: L, decoderMilliseconds: decoderMs, headMilliseconds: headMs,
            totalMilliseconds: Date().timeIntervalSince(started) * 1000)
    }
}
