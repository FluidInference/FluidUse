import CoreML
import Foundation

/// Cloudflare clef-flash (Qwen3.5-9B decision model) on Core ML, text only: Clef-style typed decisions in one
/// prefill pass. The decoder ships as layer-group packages (one function per sequence bucket) chained on the GPU,
/// followed by the joint schema head. An actor: Core ML's synchronous `prediction` is not thread-safe.
@available(macOS 15.0, iOS 18.0, *)
public actor ClefFlashManager {
    public struct Result: Sendable {
        public let answers: [ClefAnswer]
        public let inputTokens: Int
        public let bucket: Int
        public let decoderMilliseconds: Double
        public let headMilliseconds: Double
        public let totalMilliseconds: Double
    }

    struct Config: Sendable {
        let hidden: Int
        let rotaryDim: Int
        let ropeTheta: Double
        let padID: Int
        let vocab: Int
        let buckets: [Int]
        let maxQuestions: Int
        let maxOptions: Int
        let imageTokenID: Int
        let visionStartID: Int
        let visionEndID: Int
        let parts: [String]
        let head: String

        init(json: [String: Any]) throws {
            func int(_ key: String) throws -> Int {
                guard let v = (json[key] as? NSNumber)?.intValue else {
                    throw ClefVisionError.invalidAsset("config.json: \(key)")
                }
                return v
            }
            guard let buckets = json["buckets"] as? [NSNumber], let parts = json["parts"] as? [String],
                let head = json["head"] as? String, let theta = (json["rope_theta"] as? NSNumber)?.doubleValue
            else { throw ClefVisionError.invalidAsset("config.json is missing sections") }
            hidden = try int("hidden_size")
            rotaryDim = try int("rotary_dim")
            ropeTheta = theta
            padID = try int("pad_id")
            vocab = try int("vocab_size")
            self.buckets = buckets.map(\.intValue).sorted()
            maxQuestions = try int("max_questions")
            maxOptions = try int("max_options")
            imageTokenID = try int("image_token_id")
            visionStartID = try int("vision_start_token_id")
            visionEndID = try int("vision_end_token_id")
            self.parts = parts
            self.head = head
        }
    }

    public nonisolated let bucket: Int
    let config: Config
    let tokenizer: QwenBPETokenizer
    private let parts: [MLModel]
    private let head: MLModel
    private let embeddings: Data  // fp16 [vocab, hidden], input gather
    private let outputEmbeddings: Data  // fp16 [vocab, hidden], the head's lexical option rows
    private let cos: MLMultiArray
    private let sin: MLMultiArray

    /// Loads one sequence bucket (records up to `bucket` tokens). Packages are compiled once into `compiled/`.
    public static func load(
        from directory: URL, bucket: Int = 512, computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws
        -> ClefFlashManager
    {
        let json = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let dict = json as? [String: Any] else { throw ClefVisionError.invalidAsset("config.json") }
        let config = try Config(json: dict)
        guard config.buckets.contains(bucket) else { throw ClefVisionError.invalidInput("no \(bucket)-token bucket") }
        let tokenizer = try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let tableBytes = config.vocab * config.hidden * 2
        let embeddings = try Data(
            contentsOf: directory.appendingPathComponent("embeddings.f16"), options: .alwaysMapped)
        let outputEmbeddings = try Data(
            contentsOf: directory.appendingPathComponent("output_embeddings.f16"), options: .alwaysMapped)
        guard embeddings.count == tableBytes, outputEmbeddings.count == tableBytes else {
            throw ClefVisionError.invalidAsset("embedding table size")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.functionName = "L\(bucket)"
        let compiledDir = directory.appendingPathComponent("compiled")
        try FileManager.default.createDirectory(at: compiledDir, withIntermediateDirectories: true)
        func load(_ path: String) async throws -> MLModel {
            let source = directory.appendingPathComponent(path)
            let target = compiledDir.appendingPathComponent(
                source.deletingPathExtension().lastPathComponent + ".mlmodelc")
            if !FileManager.default.fileExists(atPath: target.path) {
                let temporary = try await MLModel.compileModel(at: source)
                try FileManager.default.moveItem(at: temporary, to: target)
            }
            return try await MLModel.load(contentsOf: target, configuration: configuration)
        }
        var parts: [MLModel] = []
        for path in config.parts { parts.append(try await load(path)) }
        let head = try await load(config.head)
        let (cos, sin) = try ropeTables(length: bucket, config: config)
        return ClefFlashManager(
            bucket: bucket, config: config, tokenizer: tokenizer, parts: parts, head: head, embeddings: embeddings,
            outputEmbeddings: outputEmbeddings, cos: cos, sin: sin)
    }

    init(
        bucket: Int, config: Config, tokenizer: QwenBPETokenizer, parts: [MLModel], head: MLModel, embeddings: Data,
        outputEmbeddings: Data, cos: MLMultiArray, sin: MLMultiArray
    ) {
        self.bucket = bucket
        self.config = config
        self.tokenizer = tokenizer
        self.parts = parts
        self.head = head
        self.embeddings = embeddings
        self.outputEmbeddings = outputEmbeddings
        self.cos = cos
        self.sin = sin
    }

    /// Text-only rows: position p on every M-RoPE axis, so the interleaved sections collapse to plain RoPE. fp16 [L, R].
    static func ropeTables(length: Int, config: Config) throws -> (MLMultiArray, MLMultiArray) {
        let dim = config.rotaryDim
        let half = dim / 2
        let cosArray = try MLMultiArray(shape: [length as NSNumber, dim as NSNumber], dataType: .float16)
        let sinArray = try MLMultiArray(shape: [length as NSNumber, dim as NSNumber], dataType: .float16)
        let c = cosArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sinArray.dataPointer.assumingMemoryBound(to: Float16.self)
        for position in 0..<length {
            for i in 0..<half {
                let angle = Double(position) / pow(config.ropeTheta, Double(2 * i) / Double(dim))
                let cv = Float16(Foundation.cos(angle))
                let sv = Float16(Foundation.sin(angle))
                c[position * dim + i] = cv
                c[position * dim + half + i] = cv
                s[position * dim + i] = sv
                s[position * dim + half + i] = sv
            }
        }
        return (cosArray, sinArray)
    }

    /// Run once so the first real request does not pay Core ML's lazy GPU setup.
    public func warm() throws {
        _ = try answer(state: "warm-up", questions: [("ready", .noul(instructions: "Ready?"))])
    }

    public func encode(state: Any, questions: [(id: String, question: ClefQuestion)]) throws -> ClefEncodedRecord {
        try ClefRecordEncoder.encode(
            state: state, questions: questions, imageTokenCounts: [], tokenizer: tokenizer,
            imageTokenID: config.imageTokenID, visionStartID: config.visionStartID, visionEndID: config.visionEndID,
            maxLength: bucket)
    }

    /// Decide `questions` about `state` (text or JSON-like value). Answers follow the question order; option
    /// probabilities follow Clef's option order (`ClefAnswer.optionIDs`).
    public func answer(state: Any, questions: [(id: String, question: ClefQuestion)]) throws -> Result {
        let started = Date()
        let record = try encode(state: state, questions: questions)
        let L = bucket
        let D = config.hidden
        // 1. embedding gather (fp16 table -> fp16 row)
        let row = try MLMultiArray(shape: [1, L as NSNumber, D as NSNumber], dataType: .float16)
        let padID = config.padID
        embeddings.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self).baseAddress!
            let out = row.dataPointer.assumingMemoryBound(to: Float16.self)
            for position in 0..<L {
                let token = position < record.inputIDs.count ? record.inputIDs[position] : padID
                (out + position * D).update(from: table + token * D, count: D)
            }
        }
        var hidden = row
        // 2. decoder parts
        let decoderStarted = Date()
        for part in parts {
            let output = try part.prediction(
                from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden, "cos": cos, "sin": sin]))
            guard let next = output.featureValue(for: "out")?.multiArrayValue else {
                throw ClefVisionError.invalidAsset("decoder part returned no output")
            }
            hidden = next
        }
        let decoderMs = Date().timeIntervalSince(decoderStarted) * 1000
        // 3. head: fp32 [L, D] states (stride-aware copy; Core ML outputs may carry padded row strides)
        let headStarted = Date()
        let states = try MLMultiArray(shape: [L as NSNumber, D as NSNumber], dataType: .float32)
        let dst = states.dataPointer.assumingMemoryBound(to: Float.self)
        let rowStride = hidden.strides[1].intValue
        let colStride = hidden.strides[2].intValue
        let src = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        for row in 0..<L {
            for col in 0..<D { dst[row * D + col] = Float(src[row * rowStride + col * colStride]) }
        }
        let (headInputs, optionCount) = try outputEmbeddings.withUnsafeBytes { raw in
            try ClefVisionHost.headInputs(
                record: record, states: states, length: L, maxQ: config.maxQuestions, maxO: config.maxOptions,
                embeddings: raw.bindMemory(to: Float16.self), hidden: D)
        }
        let headOutput = try head.prediction(from: headInputs)
        guard let logits = headOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw ClefVisionError.invalidAsset("head returned no logits")
        }
        let headMs = Date().timeIntervalSince(headStarted) * 1000
        let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
        var answers: [ClefAnswer] = []
        var offset = 0
        for question in record.questions {
            let count = question.optionIDs.count
            let values = (0..<count).map { pointer[offset + $0] }
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
        return Result(
            answers: answers, inputTokens: record.inputIDs.count, bucket: L, decoderMilliseconds: decoderMs,
            headMilliseconds: headMs, totalMilliseconds: Date().timeIntervalSince(started) * 1000)
    }
}
