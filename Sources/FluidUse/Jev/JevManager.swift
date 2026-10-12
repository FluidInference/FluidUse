import Accelerate
import CoreGraphics
import CoreML
import Foundation

/// autotrust/JEV-9B with vision, System 1, on Core ML: typed decisions (yes/no, 0-5 score, choice) over text and
/// images, one calibrated probability per option from a single prefill.
///
/// Packages: the Qwen3.5-9B vision tower (fp32, one per patch bucket), the decoder with JEV's System 1 LoRA merged in
/// (8 layer-group parts, 8-bit, one function per sequence bucket, chained on the GPU), and on the host the adapted
/// `lm_head` rows at the decision-token ids plus JEV's per-slot bias and per-kind temperature. The prompt is
/// JEV's `serve_decide.py` System 1 template. An actor: Core ML's synchronous `prediction` is not thread-safe.
@available(macOS 15.0, iOS 18.0, *)
public actor JevManager {
    public nonisolated let bucket: Int
    let config: JevConfig
    let tokenizer: QwenBPETokenizer
    private let parts: [MLModel]
    private let embeddings: Data  // fp16 [vocab, hidden]
    private let headRows: [Float]  // fp32 [headCount, hidden]
    private let posEmbed: Data  // fp32 vision position table
    private let vision: [(patches: Int, model: MLModel)]  // ascending patch buckets

    /// Loads one decoder sequence bucket (prompts up to `bucket` tokens) and the vision packages for `visionPatches`
    /// (default: every bucket in the bundle). Packages are compiled once into `compiled/`.
    public static func load(
        from directory: URL, bucket: Int = 256, visionPatches: [Int]? = nil, computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> JevManager {
        func json(_ name: String) throws -> [String: Any] {
            let object = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent(name)))
            guard let dict = object as? [String: Any] else { throw JevError.invalidAsset(name) }
            return dict
        }
        let config = try JevConfig(manifest: try json("config.json"), head: try json("head.json"))
        guard config.buckets.contains(bucket) else { throw JevError.invalidInput("no \(bucket)-token bucket") }
        let tokenizer = try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let embeddings = try Data(
            contentsOf: directory.appendingPathComponent("embeddings.f16"), options: .alwaysMapped)
        guard embeddings.count == config.vocab * config.hidden * 2 else {
            throw JevError.invalidAsset("embeddings.f16 size")
        }
        let rowsData = try Data(contentsOf: directory.appendingPathComponent(config.headRows))
        guard rowsData.count == config.headCount * config.hidden * 4 else {
            throw JevError.invalidAsset("head rows size")
        }
        let headRows = rowsData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let posEmbed = try Data(contentsOf: directory.appendingPathComponent(config.posEmbed))
        let v = config.clef.vision
        guard posEmbed.count == v.depthPositions * v.hidden * 4 else { throw JevError.invalidAsset("pos_embed size") }
        var parts: [MLModel] = []
        for path in config.parts {
            parts.append(try await compiled(path, in: directory, units: computeUnits, function: "L\(bucket)"))
        }
        var vision: [(Int, MLModel)] = []
        for package in config.visionPackages where visionPatches?.contains(package.patches) ?? true {
            vision.append(
                (package.patches, try await compiled(package.file, in: directory, units: computeUnits, function: nil)))
        }
        return JevManager(
            bucket: bucket, config: config, tokenizer: tokenizer, parts: parts, embeddings: embeddings,
            headRows: headRows, posEmbed: posEmbed, vision: vision)
    }

    static func compiled(
        _ path: String, in directory: URL, units: MLComputeUnits, function: String?
    ) async throws
        -> MLModel
    {
        let compiledDir = directory.appendingPathComponent("compiled")
        try FileManager.default.createDirectory(at: compiledDir, withIntermediateDirectories: true)
        let source = directory.appendingPathComponent(path)
        let target = compiledDir.appendingPathComponent(source.deletingPathExtension().lastPathComponent + ".mlmodelc")
        if !FileManager.default.fileExists(atPath: target.path) {
            let temporary = try await MLModel.compileModel(at: source)
            try FileManager.default.moveItem(at: temporary, to: target)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        configuration.functionName = function
        return try await MLModel.load(contentsOf: target, configuration: configuration)
    }

    init(
        bucket: Int, config: JevConfig, tokenizer: QwenBPETokenizer, parts: [MLModel], embeddings: Data,
        headRows: [Float], posEmbed: Data, vision: [(Int, MLModel)]
    ) {
        self.bucket = bucket
        self.config = config
        self.tokenizer = tokenizer
        self.parts = parts
        self.embeddings = embeddings
        self.headRows = headRows
        self.posEmbed = posEmbed
        self.vision = vision.sorted { $0.0 < $1.0 }.map { (patches: $0.0, model: $0.1) }
    }

    /// The smallest loaded vision bucket that holds `patches`.
    private func visionModel(for patches: Int) throws -> (MLModel, Int) {
        guard let package = vision.first(where: { $0.patches >= patches }) else {
            throw JevError.invalidInput(
                "image has \(patches) patches; the largest loaded vision bucket holds \(vision.last?.patches ?? 0)")
        }
        return (package.model, package.patches)
    }

    // MARK: prompt

    struct Encoded {
        let inputIDs: [Int]
        let positions: [[Int]]
        let imageRanges: [Range<Int>]
        let optionCount: Int
    }

    /// `serve_decide.s1_pass` text with each image expanded to its merged-token count. Adjacent text parts are joined
    /// before tokenizing and images are spliced as ids, which matches Hugging Face splitting on added tokens.
    func encode(
        state: [JevStatePart], imageTokens: [Int], grids: [(h: Int, w: Int)], question: JevQuestion
    ) throws
        -> Encoded
    {
        let lines: [String]
        switch question.kind {
        case .noul: lines = ["false", "true"]
        case .score: lines = (0...5).map(String.init)
        case .choice(let options):
            guard (2...config.labels.count).contains(options.count) else {
                throw JevError.invalidInput("choice needs 2-\(config.labels.count) options, got \(options.count)")
            }
            lines = options.enumerated().map { "\(config.labels[$0.offset])) \($0.element)" }
        }
        var ids: [Int] = []
        var ranges: [Range<Int>] = []
        var pending = "[kind] \(question.kind.name)\n[state] "
        var image = 0
        for part in state {
            switch part {
            case .text(let text): pending += text
            case .image:
                ids += try tokenizer.encode(pending)
                pending = ""
                ids.append(config.visionStartID)
                let start = ids.count
                ids += [Int](repeating: config.imageTokenID, count: imageTokens[image])
                ranges.append(start..<ids.count)
                ids.append(config.visionEndID)
                image += 1
            }
        }
        pending += "\n[question] \(question.text)\n[options]\n" + lines.joined(separator: "\n") + "\n[decision]:"
        ids += try tokenizer.encode(pending)
        guard ids.count <= bucket else { throw JevError.tooLong(tokens: ids.count, bucket: bucket) }
        let positions = ClefVisionHost.mropePositions(
            inputIDs: ids, imageGrids: grids, imageTokenID: config.imageTokenID, mergeSize: config.clef.vision.mergeSize
        )
        return Encoded(inputIDs: ids, positions: positions, imageRanges: ranges, optionCount: lines.count)
    }

    /// The prompt's token ids and M-RoPE positions for one question (image preprocessing only, no model calls).
    public func tokens(state: [JevStatePart], question: JevQuestion) throws -> (ids: [Int], positions: [[Int]]) {
        var counts: [Int] = []
        var grids: [(h: Int, w: Int)] = []
        for part in state {
            guard case .image(let image) = part else { continue }
            let patches = try ClefImagePreprocessor.patches(from: image, config: config.clef)
            counts.append(patches.mergedTokens)
            grids.append((patches.gridH, patches.gridW))
        }
        let encoded = try encode(state: state, imageTokens: counts, grids: grids, question: question)
        return (encoded.inputIDs, encoded.positions)
    }

    // MARK: decide

    /// Answer every question about the same state. Images go through the vision tower once; each question is one
    /// decoder pass. Probabilities follow the option order of each question (`["false", "true"]` for noul).
    public func decide(state: [JevStatePart], questions: [JevQuestion]) throws -> JevResult {
        let started = Date()
        // 1. vision tokens per image
        var visionTokens: [[Float]] = []
        var counts: [Int] = []
        var grids: [(h: Int, w: Int)] = []
        let visionStarted = Date()
        for part in state {
            guard case .image(let image) = part else { continue }
            let patches = try ClefImagePreprocessor.patches(from: image, config: config.clef)
            let (model, bucketPatches) = try visionModel(for: patches.count)
            let clef = config.clef(patches: bucketPatches)
            let inputs = try posEmbed.withUnsafeBytes {
                try ClefVisionHost.visionInputs(patches, posEmbed: $0.bindMemory(to: Float.self), config: clef)
            }
            guard let out = try model.prediction(from: inputs.features).featureValue(for: "tokens")?.multiArrayValue
            else { throw JevError.invalidAsset("vision tower returned no tokens") }
            visionTokens.append(Self.rows(out, count: inputs.mergedTokens, width: config.hidden))
            counts.append(inputs.mergedTokens)
            grids.append((patches.gridH, patches.gridW))
        }
        let visionMs = Date().timeIntervalSince(visionStarted) * 1000
        // 2. one decoder pass per question
        let (cos, sin) = try ropeArrays()  // refilled per question
        var decisions: [JevDecision] = []
        var tokens: [Int] = []
        var decoderMs = 0.0
        for question in questions {
            let encoded = try encode(state: state, imageTokens: counts, grids: grids, question: question)
            try fillRope(cos: cos, sin: sin, positions: encoded.positions)
            var hidden = try hiddenRow(encoded, visionTokens: visionTokens)
            let decoderStarted = Date()
            for part in parts {
                let output = try part.prediction(
                    from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden, "cos": cos, "sin": sin]))
                guard let next = output.featureValue(for: "out")?.multiArrayValue else {
                    throw JevError.invalidAsset("decoder part returned no output")
                }
                hidden = next
            }
            decoderMs += Date().timeIntervalSince(decoderStarted) * 1000
            decisions.append(try decision(hidden, row: encoded.inputIDs.count - 1, kind: question.kind))
            tokens.append(encoded.inputIDs.count)
        }
        return JevResult(
            decisions: decisions, inputTokens: tokens, visionMilliseconds: visionMs, decoderMilliseconds: decoderMs,
            totalMilliseconds: Date().timeIntervalSince(started) * 1000)
    }

    /// Head: adapted lm_head rows · last real row, + bias, / temperature, softmax over the question's options.
    func decision(_ hidden: MLMultiArray, row: Int, kind: JevKind) throws -> JevDecision {
        let D = config.hidden
        let rowStride = hidden.strides[1].intValue
        let colStride = hidden.strides[2].intValue
        let src = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        var last = [Float](repeating: 0, count: D)
        for d in 0..<D { last[d] = Float(src[row * rowStride + d * colStride]) }
        let (rows, bias) = try config.decisionRows(kind)
        let temperature = config.temperatures[kind.name] ?? 1
        var logits = [Float](repeating: 0, count: rows.count)
        headRows.withUnsafeBufferPointer { table in
            for (i, r) in rows.enumerated() {
                var dot: Float = 0
                vDSP_dotpr(table.baseAddress! + r * D, 1, last, 1, &dot, vDSP_Length(D))
                logits[i] = (dot + bias[i]) / temperature
            }
        }
        let peak = logits.max() ?? 0
        let exps = logits.map { expf($0 - peak) }
        let sum = exps.reduce(0, +)
        let options: [String]
        switch kind {
        case .noul: options = ["false", "true"]
        case .score: options = (0...5).map(String.init)
        case .choice(let o): options = o
        }
        return JevDecision(options: options, probabilities: exps.map { $0 / sum })
    }

    /// fp16 [1, L, D]: token embeddings, vision tokens spliced at the image-pad positions, padding after the prompt.
    func hiddenRow(_ encoded: Encoded, visionTokens: [[Float]]) throws -> MLMultiArray {
        let L = bucket
        let D = config.hidden
        let padID = config.padID
        let ids = encoded.inputIDs
        let array = try MLMultiArray(shape: [1, L as NSNumber, D as NSNumber], dataType: .float16)
        embeddings.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self).baseAddress!
            let out = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for position in 0..<L {
                let token = position < ids.count ? ids[position] : padID
                (out + position * D).update(from: table + token * D, count: D)
            }
        }
        let out = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for (range, tokens) in zip(encoded.imageRanges, visionTokens) {
            guard tokens.count == range.count * D else { throw JevError.invalidAsset("vision tokens / image pads") }
            for (k, value) in tokens.enumerated() { out[range.lowerBound * D + k] = Float16(value) }
        }
        return array
    }

    func ropeArrays() throws -> (MLMultiArray, MLMultiArray) {
        let shape = [bucket as NSNumber, config.rotaryDim as NSNumber]
        return (
            try MLMultiArray(shape: shape, dataType: .float16), try MLMultiArray(shape: shape, dataType: .float16)
        )
    }

    /// Interleaved M-RoPE tables (fp16 [L, R]); padding continues the position count after the prompt.
    func fillRope(cos: MLMultiArray, sin: MLMultiArray, positions: [[Int]]) throws {
        let (c32, s32) = try ClefVisionHost.ropeTables(positions: positions, length: bucket, config: config.clef)
        let count = bucket * config.rotaryDim
        let c = cos.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sin.dataPointer.assumingMemoryBound(to: Float16.self)
        let cs = c32.dataPointer.assumingMemoryBound(to: Float.self)
        let ss = s32.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            c[i] = Float16(cs[i])
            s[i] = Float16(ss[i])
        }
    }

    /// The first `count` rows of a [N, width] fp32 output, stride-aware.
    static func rows(_ array: MLMultiArray, count: Int, width: Int) -> [Float] {
        let rowStride = array.strides[0].intValue
        let colStride = array.strides[1].intValue
        let src = array.dataPointer.assumingMemoryBound(to: Float.self)
        var out = [Float](repeating: 0, count: count * width)
        for r in 0..<count {
            for d in 0..<width { out[r * width + d] = src[r * rowStride + d * colStride] }
        }
        return out
    }
}
