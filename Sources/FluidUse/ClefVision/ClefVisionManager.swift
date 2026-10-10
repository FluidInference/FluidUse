import CoreGraphics
import CoreML
import Foundation

/// clef-vision-0.8b on Core ML: Clef-style typed decisions over text and images in one prefill pass.
///
/// Three packages (vision tower, language-model rows per sequence bucket, joint schema head) with the gathers done
/// here. Load with `ClefVisionManager.load(from:)` on the directory `ClefVisionModelStore.ensure()` returns. An
/// actor: Core ML's synchronous `prediction` is not thread-safe, so requests are serialised per manager.
@available(macOS 15.0, iOS 18.0, *)
public actor ClefVisionManager {
    /// Constants callers may want.
    public struct Limits: Sendable {
        public let buckets: [Int]
        public let maxQuestions: Int
        public let maxOptions: Int
        public let maxImagePatches: Int
    }

    public nonisolated let limits: Limits
    let config: ClefVisionConfig
    let tokenizer: QwenBPETokenizer
    private let vision: MLModel
    private let languageModels: [Int: MLModel]  // bucket length -> model
    private let heads: [Int: MLModel]  // bucket length -> head
    private let embeddings: Data  // fp16 [vocab, hidden]
    private let posEmbed: Data  // fp32 [side*side, hidden]

    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU, buckets: [Int]? = nil
    ) async throws -> ClefVisionManager {
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let json = manifest as? [String: Any] else { throw ClefVisionError.invalidAsset("config.json") }
        let config = try ClefVisionConfig(json: json)
        let tokenizer = try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let embeddings = try Data(
            contentsOf: directory.appendingPathComponent("embeddings.f16"), options: .alwaysMapped)
        guard embeddings.count == config.vocabSize * config.hiddenSize * 2 else {
            throw ClefVisionError.invalidAsset("embeddings.f16 size")
        }
        let posEmbed = try Data(
            contentsOf: directory.appendingPathComponent("Vision_P784/pos_embed.f32"), options: .alwaysMapped)
        guard posEmbed.count == config.vision.depthPositions * config.vision.hidden * 4 else {
            throw ClefVisionError.invalidAsset("pos_embed.f32 size")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        func load(_ path: String) async throws -> MLModel {
            let compiled = try await MLModel.compileModel(at: directory.appendingPathComponent(path))
            return try await MLModel.load(contentsOf: compiled, configuration: configuration)
        }
        let vision = try await load("Vision_P784/VisionTower_fp32.mlpackage")
        var languageModels: [Int: MLModel] = [:]
        var heads: [Int: MLModel] = [:]
        for length in buckets ?? config.lmBuckets {
            languageModels[length] = try await load("LM_L\(length)/LMRows_fp16.mlpackage")
            heads[length] = try await load(
                "Head_L\(length)_Q\(config.headMaxQuestions)_O\(config.headMaxOptions)/Head_fp32.mlpackage")
        }
        return ClefVisionManager(
            config: config, tokenizer: tokenizer, vision: vision, languageModels: languageModels, heads: heads,
            embeddings: embeddings, posEmbed: posEmbed)
    }

    init(
        config: ClefVisionConfig, tokenizer: QwenBPETokenizer, vision: MLModel, languageModels: [Int: MLModel],
        heads: [Int: MLModel], embeddings: Data, posEmbed: Data
    ) {
        self.config = config
        self.limits = Limits(
            buckets: languageModels.keys.sorted(), maxQuestions: config.headMaxQuestions,
            maxOptions: config.headMaxOptions,
            maxImagePatches: config.vision.patches)
        self.tokenizer = tokenizer
        self.vision = vision
        self.languageModels = languageModels
        self.heads = heads
        self.embeddings = embeddings
        self.posEmbed = posEmbed
    }

    /// Run the smallest bucket once so the first real request does not pay Core ML's lazy setup.
    public func warm() throws {
        _ = try answer(state: "warm-up", images: [], questions: [("ready", .noul(instructions: "Ready?"))])
    }

    /// Token ids and spans for a request (exposed for tests and the parity CLI). `imageTokenCounts` are the merged
    /// vision tokens per image (grid_h · grid_w / 4). The state is trimmed to the largest *loaded* bucket.
    public func encode(
        state: Any, questions: [(id: String, question: ClefQuestion)], imageTokenCounts: [Int]
    ) throws
        -> ClefEncodedRecord
    {
        try ClefRecordEncoder.encode(
            state: state, questions: questions, imageTokenCounts: imageTokenCounts, tokenizer: tokenizer,
            imageTokenID: config.imageTokenID,
            visionStartID: config.visionStartID, visionEndID: config.visionEndID,
            maxLength: limits.buckets.max() ?? config.lmBuckets.max() ?? 0)
    }

    /// Tokenizer + manifest only (no Core ML packages): enough to encode records, for tests and tooling.
    public static func encoder(from directory: URL) throws -> ClefVisionEncoder {
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let json = manifest as? [String: Any] else { throw ClefVisionError.invalidAsset("config.json") }
        return ClefVisionEncoder(
            config: try ClefVisionConfig(json: json),
            tokenizer: try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json")))
    }

    /// Debug: the preprocessed patches, the patch grid and the vision tower's merged tokens for one image.
    public func visionTokens(for image: CGImage) throws -> (patches: [Float], gridH: Int, gridW: Int, tokens: [Float]) {
        let patches = try ClefImagePreprocessor.patches(from: image, config: config)
        let inputs = try posEmbed.withUnsafeBytes { raw in
            try ClefVisionHost.visionInputs(patches, posEmbed: raw.bindMemory(to: Float.self), config: config)
        }
        let output = try vision.prediction(from: inputs.features)
        guard let tokens = output.featureValue(for: "tokens")?.multiArrayValue else {
            throw ClefVisionError.invalidAsset("vision tower returned no tokens")
        }
        let count = inputs.mergedTokens * config.hiddenSize
        var rows = [Float](repeating: 0, count: count)
        let pointer = tokens.dataPointer.assumingMemoryBound(to: Float.self)
        rows.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: count) }
        return (patches.rows, patches.gridH, patches.gridW, rows)
    }

    /// Decide `questions` about `state` and `images`. Answers follow the question order; option probabilities
    /// follow Clef's option order (`ClefAnswer.optionIDs`).
    public func answer(
        state: Any, images: [CGImage], questions: [(id: String, question: ClefQuestion)]
    ) throws
        -> ClefVisionResult
    {
        let hidden = config.hiddenSize
        // 1. images -> vision tokens
        var visionTokens: [[Float]] = []
        var grids: [(h: Int, w: Int)] = []
        var visionMs = 0.0
        for image in images {
            let patches = try ClefImagePreprocessor.patches(from: image, config: config)
            let inputs = try posEmbed.withUnsafeBytes { raw in
                try ClefVisionHost.visionInputs(patches, posEmbed: raw.bindMemory(to: Float.self), config: config)
            }
            let started = Date()
            let output = try vision.prediction(from: inputs.features)
            visionMs += Date().timeIntervalSince(started) * 1000
            guard let tokens = output.featureValue(for: "tokens")?.multiArrayValue else {
                throw ClefVisionError.invalidAsset("vision tower returned no tokens")
            }
            let count = inputs.mergedTokens * hidden
            var rows = [Float](repeating: 0, count: count)
            let pointer = tokens.dataPointer.assumingMemoryBound(to: Float.self)
            rows.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: count) }
            visionTokens.append(rows)
            grids.append((patches.gridH, patches.gridW))
        }
        // 2. record -> tokens, spans; pick the bucket
        let record = try encode(
            state: state, questions: questions, imageTokenCounts: visionTokens.map { $0.count / hidden })
        let largest = config.lmBuckets.max() ?? 0
        guard let bucket = languageModels.keys.sorted().first(where: { $0 >= record.inputIDs.count }),
            let languageModel = languageModels[bucket], let head = heads[bucket]
        else { throw ClefVisionError.tooLong(tokens: record.inputIDs.count, bucket: largest) }
        // 3. language model rows
        let positions = ClefVisionHost.mropePositions(
            inputIDs: record.inputIDs, imageGrids: grids, imageTokenID: config.imageTokenID,
            mergeSize: config.vision.mergeSize)
        let (cos, sin) = try ClefVisionHost.ropeTables(positions: positions, length: bucket, config: config)
        let hiddenRow = try embeddings.withUnsafeBytes { raw in
            try ClefVisionHost.hiddenRow(
                inputIDs: record.inputIDs, length: bucket, embeddings: raw.bindMemory(to: Float16.self), hidden: hidden,
                padID: config.padID, imageRanges: record.imageTokenRanges, visionTokens: visionTokens)
        }
        let lmStarted = Date()
        let lmOutput = try languageModel.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["hidden": hiddenRow, "cos": cos, "sin": sin]))
        let lmMs = Date().timeIntervalSince(lmStarted) * 1000
        guard let states = lmOutput.featureValue(for: "states")?.multiArrayValue else {
            throw ClefVisionError.invalidAsset("language model returned no states")
        }
        // 4. head
        let flatStates = try MLMultiArray(shape: [bucket as NSNumber, hidden as NSNumber], dataType: .float32)
        memcpy(flatStates.dataPointer, states.dataPointer, bucket * hidden * MemoryLayout<Float>.size)
        let headStarted = Date()
        let (headInputs, optionCount) = try embeddings.withUnsafeBytes { raw in
            try ClefVisionHost.headInputs(
                record: record, states: flatStates, length: bucket, maxQ: config.headMaxQuestions,
                maxO: config.headMaxOptions,
                embeddings: raw.bindMemory(to: Float16.self), hidden: hidden)
        }
        let headOutput = try head.prediction(from: headInputs)
        let headMs = Date().timeIntervalSince(headStarted) * 1000
        guard let logits = headOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw ClefVisionError.invalidAsset("head returned no logits")
        }
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
        return ClefVisionResult(
            answers: answers, inputTokens: record.inputIDs.count,
            imageTokens: visionTokens.reduce(0) { $0 + $1.count / hidden },
            visionMilliseconds: visionMs, languageMilliseconds: lmMs, headMilliseconds: headMs)
    }
}

/// Record encoding without any Core ML package loaded (see `ClefVisionManager.encoder(from:)`).
public struct ClefVisionEncoder: Sendable {
    let config: ClefVisionConfig
    let tokenizer: QwenBPETokenizer

    public func encode(
        state: Any, questions: [(id: String, question: ClefQuestion)], imageTokenCounts: [Int],
        maxLength: Int? = nil
    ) throws -> ClefEncodedRecord {
        try ClefRecordEncoder.encode(
            state: state, questions: questions, imageTokenCounts: imageTokenCounts, tokenizer: tokenizer,
            imageTokenID: config.imageTokenID,
            visionStartID: config.visionStartID, visionEndID: config.visionEndID,
            maxLength: maxLength ?? config.lmBuckets.max() ?? 0)
    }

    public var imageTokenID: Int { config.imageTokenID }
    public var mergeSize: Int { config.vision.mergeSize }
}
