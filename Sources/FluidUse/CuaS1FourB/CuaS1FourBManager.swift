import Accelerate
@preconcurrency import CoreML
import Foundation

/// On-device Cua-S1-4B-0.2 (Qwen3.5-4B + Cua LoRA) closed-option GUI decisions.
///
/// One prefill pass per decision: the prompt is right-padded to a fixed bucket, run through four
/// Core ML decoder parts (8 Qwen3.5 layers each), and the answer-letter logits A..Z at the last
/// prompt position are read out. Nothing is generated, so there is no KV cache. The embedding
/// gather happens on the host (memory-mapped fp16 table) so screenshot features can be spliced in.
///
/// A manager runs one modality: `text` (accessibility tree) or `multimodal` (screenshot); the two
/// adapters were trained separately and ship as separate decoders. Calls are serialized by the actor.
public actor CuaS1FourBManager {
    public struct Configuration: Sendable {
        public var modality: CuaS1FourBModality
        /// Bucket lengths to load (each loads its own ~6.8 GB fp16 of decoder weights).
        public var lengths: [Int]
        /// Weight variant: `""` fp16 (6.8 GB), `"w8"` int8 linears (3.4 GB), `"gptq"` GPTQ MLP int4 + int8
        /// (2.6 GB, text only; same GUI-360 accuracy as fp16).
        public var variant: String
        /// The 4B prefill runs best on the GPU; the ANE path falls back to CPU for most of the graph.
        public var computeUnits: MLComputeUnits

        public init(
            modality: CuaS1FourBModality = .text, lengths: [Int]? = nil, variant: String = "",
            computeUnits: MLComputeUnits = .cpuAndGPU
        ) {
            self.modality = modality
            self.lengths = lengths ?? (modality == .text ? [1024] : [2048])
            self.variant = variant
            self.computeUnits = computeUnits
        }
    }

    struct Bucket {
        let length: Int
        let parts: [MLModel]
        let letterTokenIds: [Int]
    }

    struct RopeParameters {
        let rotaryDim: Int
        let theta: Double
        let mropeSection: [Int]
    }

    nonisolated public let modality: CuaS1FourBModality
    nonisolated public let tokenizer: QwenTokenizer
    let buckets: [Bucket]
    let embeddings: Data
    let hiddenSize: Int
    let rope: RopeParameters
    let vision: CuaS1FourBVision?

    init(
        modality: CuaS1FourBModality, tokenizer: QwenTokenizer, buckets: [Bucket], embeddings: Data,
        hiddenSize: Int, rope: RopeParameters, vision: CuaS1FourBVision?
    ) {
        self.modality = modality
        self.tokenizer = tokenizer
        self.buckets = buckets.sorted { $0.length < $1.length }
        self.embeddings = embeddings
        self.hiddenSize = hiddenSize
        self.rope = rope
        self.vision = vision
    }

    /// Loaded bucket lengths, ascending.
    public var lengths: [Int] { buckets.map(\.length) }

    /// Load from a local directory laid out like the published repository:
    /// `tokenizer.json`, `embeddings.f16`, `<modality>/L<len>[-variant]/CuaS1Decoder_part{0..3}.mlmodelc|.mlpackage`
    /// (+ `config.json`), and for multimodal `multimodal/vision/`.
    public static func load(
        from directory: URL, configuration: Configuration = Configuration()
    ) async throws
        -> CuaS1FourBManager
    {
        let manager = FileManager.default
        let tokenizer = try QwenTokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let embeddingsURL = try firstExisting(
            [
                directory.appendingPathComponent("embeddings.f16"),
                directory.appendingPathComponent(configuration.modality.rawValue).appendingPathComponent(
                    "embeddings.f16"),
            ], what: "embeddings.f16")
        let embeddings = try Data(contentsOf: embeddingsURL, options: .alwaysMapped)

        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = configuration.computeUnits
        var buckets: [Bucket] = []
        var hiddenSize = 0
        var rope: RopeParameters?
        for length in configuration.lengths {
            let name = "L\(length)" + (configuration.variant.isEmpty ? "" : "-\(configuration.variant)")
            let bucketDir = directory.appendingPathComponent(configuration.modality.rawValue)
                .appendingPathComponent(name)
            let configData = try Data(contentsOf: bucketDir.appendingPathComponent("config.json"))
            guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
                let seqLen = config["seq_len"] as? Int, seqLen == length,
                let hidden = config["hidden_size"] as? Int,
                let partsInfo = config["parts"] as? [Any],
                let letterIds = config["letter_token_ids"] as? [Int],
                let rotaryDim = config["rotary_dim"] as? Int,
                let theta = config["rope_theta"] as? Double,
                let section = config["mrope_section"] as? [Int]
            else {
                throw CuaS1FourBError.invalidAsset("bad config.json in \(bucketDir.path)")
            }
            guard letterIds.count == CuaS1FourBPrompt.letters.count else {
                throw CuaS1FourBError.invalidAsset("config.json letter_token_ids must list A..Z")
            }
            for (letter, id) in zip(CuaS1FourBPrompt.letters, letterIds) where tokenizer.encode(letter) != [id] {
                throw CuaS1FourBError.invalidAsset("letter \(letter) is not the single token \(id) in tokenizer.json")
            }
            hiddenSize = hidden
            rope = RopeParameters(rotaryDim: rotaryDim, theta: theta, mropeSection: section)
            var parts: [MLModel] = []
            for index in 0..<partsInfo.count {
                let url = try await compiledModel(
                    bucketDir.appendingPathComponent("CuaS1Decoder_part\(index)"), manager: manager)
                parts.append(try await MLModel.load(contentsOf: url, configuration: modelConfiguration))
            }
            buckets.append(Bucket(length: length, parts: parts, letterTokenIds: letterIds))
        }
        guard let rope, !buckets.isEmpty else { throw CuaS1FourBError.invalidModel("no bucket loaded") }
        guard embeddings.count % (hiddenSize * 2) == 0 else {
            throw CuaS1FourBError.invalidAsset("embeddings.f16 size is not a multiple of \(hiddenSize) fp16 rows")
        }

        var vision: CuaS1FourBVision?
        if configuration.modality == .multimodal {
            vision = try await CuaS1FourBVision.load(
                from: directory.appendingPathComponent("multimodal/vision"), tokenizer: tokenizer,
                computeUnits: configuration.computeUnits)
        }
        return CuaS1FourBManager(
            modality: configuration.modality, tokenizer: tokenizer, buckets: buckets, embeddings: embeddings,
            hiddenSize: hiddenSize, rope: rope, vision: vision)
    }

    /// Score every option of `state` in one forward pass.
    public func decide(_ state: CuaS1FourBState) throws -> CuaS1FourBDecision {
        try Task.checkCancellation()
        let chat = try CuaS1FourBPrompt.chat(state: state, modality: modality)
        var ids = tokenizer.encode(chat)
        var image: CuaS1FourBVision.ImageFeatures?
        if modality == .multimodal {
            guard let vision, let screenshot = state.screenshot else {
                throw CuaS1FourBError.invalidInput("multimodal modality requires a screenshot")
            }
            let features = try vision.features(for: screenshot)
            ids = try vision.expandImagePads(ids, count: features.tokens)
            image = features
        }
        let logits = try letterLogits(ids: ids, image: image)
        let count = state.options.count
        let used = Array(logits.prefix(count))
        let maxLogit = used.max() ?? 0
        let exps = used.map { expf($0 - maxLogit) }
        let total = exps.reduce(0, +)
        let scored = (0..<count).map { i in
            CuaS1FourBDecision.Scored(
                option: state.options[i], letter: CuaS1FourBPrompt.letters[i], logit: used[i],
                probability: exps[i] / total)
        }
        let bucket = try bucket(for: ids.count)
        return CuaS1FourBDecision(options: scored, tokens: ids.count, bucketLength: bucket.length)
    }

    /// Run every loaded bucket (and the vision tower) once so Core ML specializes its GPU kernels now.
    /// The first prediction of a fresh 4B graph takes on the order of 100 s on an M5 Pro; later calls,
    /// and later launches (the specialization is cached by the OS), take about a second.
    public func prewarm() throws {
        let ids = tokenizer.encode("<|im_start|>assistant\n")
        for bucket in buckets {
            _ = try letterLogits(ids: ids, image: nil, bucket: bucket)
        }
        try vision?.prewarm()
    }

    /// Letter logits A..Z for already tokenized ids (parity checks against the Python reference).
    /// `image` carries spliced screenshot features and their M-RoPE grid for multimodal prompts.
    func letterLogits(ids: [Int], image: CuaS1FourBVision.ImageFeatures?) throws -> [Float] {
        try letterLogits(ids: ids, image: image, bucket: bucket(for: ids.count))
    }

    private func letterLogits(
        ids: [Int], image: CuaS1FourBVision.ImageFeatures?, bucket: Bucket
    ) throws
        -> [Float]
    {
        let length = bucket.length
        let hidden = try MLMultiArray(
            shape: [1, NSNumber(value: length), NSNumber(value: hiddenSize)], dataType: .float16)
        let rowBytes = hiddenSize * 2
        let vocabRows = embeddings.count / rowBytes
        let hiddenPtr = hidden.dataPointer.bindMemory(to: UInt16.self, capacity: length * hiddenSize)
        hiddenPtr.initialize(repeating: 0, count: length * hiddenSize)
        var imageRow = 0
        try embeddings.withUnsafeBytes { raw in
            for (t, id) in ids.enumerated() {
                let dst = UnsafeMutableRawPointer(hiddenPtr + t * hiddenSize)
                if let image, id == image.padTokenId {
                    image.copyRow(imageRow, to: dst)
                    imageRow += 1
                    continue
                }
                guard id >= 0, id < vocabRows else { throw CuaS1FourBError.invalidInput("token id \(id) out of range") }
                guard let base = raw.baseAddress else { throw CuaS1FourBError.invalidAsset("empty embeddings") }
                dst.copyMemory(from: base + id * rowBytes, byteCount: rowBytes)
            }
        }
        if let image, imageRow != image.tokens {
            throw CuaS1FourBError.invalidInput("prompt has \(imageRow) image pads for \(image.tokens) features")
        }

        let positions =
            image.map {
                Self.mropePositions(
                    ids: ids, padTokenId: $0.padTokenId, imageTokens: $0.tokens, gridRows: $0.gridRows,
                    gridCols: $0.gridCols)
            } ?? (0..<ids.count).map { [$0, $0, $0] }
        let (cos, sin) = try ropeTables(positions: positions, length: length)
        let onehot = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float16)
        let onehotPtr = onehot.dataPointer.bindMemory(to: UInt16.self, capacity: length)
        onehotPtr.initialize(repeating: 0, count: length)
        onehotPtr[ids.count - 1] = 0x3C00  // fp16 1.0

        var current = hidden
        for (index, part) in bucket.parts.enumerated() {
            let last = index == bucket.parts.count - 1
            var inputs: [String: Any] = ["hidden": current, "cos": cos, "sin": sin]
            if last { inputs["last_onehot"] = onehot }
            let output = try autoreleasepool {
                try part.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
            }
            let name = last ? "letter_logits" : "hidden_out"
            guard let array = output.featureValue(for: name)?.multiArrayValue else {
                throw CuaS1FourBError.invalidModel("part \(index) returned no \(name)")
            }
            if last { return Self.floats(array) }
            current = try Self.contiguousCopy(array)
        }
        throw CuaS1FourBError.invalidModel("decoder has no parts")
    }

    private func bucket(for tokens: Int) throws -> Bucket {
        guard let bucket = buckets.first(where: { tokens <= $0.length }) else {
            throw CuaS1FourBError.promptTooLong(tokens: tokens, maximum: buckets.last?.length ?? 0)
        }
        return bucket
    }

    /// Interleaved M-RoPE cos/sin [L, rotaryDim] (fp16) for (t, h, w) positions; padding rows use position 0.
    func ropeTables(positions: [[Int]], length: Int) throws -> (MLMultiArray, MLMultiArray) {
        let dim = rope.rotaryDim
        let half = dim / 2
        var cosValues = [Float](repeating: 1, count: length * dim)
        var sinValues = [Float](repeating: 0, count: length * dim)
        let invFreq = (0..<half).map { 1.0 / pow(rope.theta, Double(2 * $0) / Double(dim)) }
        let hEnd = rope.mropeSection[1] * 3
        let wEnd = rope.mropeSection[2] * 3
        for (t, pos) in positions.enumerated() {
            for i in 0..<half {
                // interleaved layout: index 1, 4, 7, ... from h; 2, 5, 8, ... from w; the rest from t
                var axis = 0
                if i % 3 == 1, i < hEnd { axis = 1 } else if i % 3 == 2, i < wEnd { axis = 2 }
                let angle = Double(pos[axis]) * invFreq[i]
                let c = Float(Foundation.cos(angle))
                let s = Float(Foundation.sin(angle))
                cosValues[t * dim + i] = c
                cosValues[t * dim + i + half] = c
                sinValues[t * dim + i] = s
                sinValues[t * dim + i + half] = s
            }
        }
        return (
            try Self.half(cosValues, shape: [length, dim]), try Self.half(sinValues, shape: [length, dim])
        )
    }

    /// `Qwen3_5Model.get_rope_index` for one image: text runs count up on all three axes; the image block
    /// sits at (start, start + row, start + col) and advances the counter by max(rows, cols).
    static func mropePositions(
        ids: [Int], padTokenId: Int, imageTokens: Int, gridRows: Int, gridCols: Int
    )
        -> [[Int]]
    {
        var positions: [[Int]] = []
        positions.reserveCapacity(ids.count)
        var next = 0
        var t = 0
        while t < ids.count {
            if ids[t] == padTokenId {
                let start = next
                for row in 0..<gridRows {
                    for col in 0..<gridCols {
                        positions.append([start, start + row, start + col])
                    }
                }
                t += imageTokens
                next = start + max(gridRows, gridCols)
            } else {
                positions.append([next, next, next])
                next += 1
                t += 1
            }
        }
        return positions
    }

    static func half(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        var source = values
        source.withUnsafeMutableBytes { src in
            var input = vImage_Buffer(
                data: src.baseAddress, height: 1, width: vImagePixelCount(values.count), rowBytes: values.count * 4)
            var output = vImage_Buffer(
                data: array.dataPointer, height: 1, width: vImagePixelCount(values.count), rowBytes: values.count * 2)
            vImageConvert_PlanarFtoPlanar16F(&input, &output, 0)
        }
        return array
    }

    /// Fresh densely packed fp16 copy. GPU outputs can carry padded strides and their backing buffers;
    /// feeding them straight into the next model trips an MPSGraph shape/stride assertion.
    static func contiguousCopy(_ array: MLMultiArray) throws -> MLMultiArray {
        let shape = array.shape.map(\.intValue)
        let copy = try MLMultiArray(shape: array.shape, dataType: .float16)
        let rowLength = shape.last ?? 1
        let rows = array.count / rowLength
        let strides = array.strides.map(\.intValue)
        let dst = copy.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
        let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
        guard array.dataType == .float16, strides.last == 1 else {
            throw CuaS1FourBError.invalidModel("unexpected decoder output layout \(array.dataType.rawValue) \(strides)")
        }
        var outer = [Int](repeating: 0, count: max(shape.count - 1, 0))
        for row in 0..<rows {
            var offset = 0
            var rest = row
            for axis in stride(from: shape.count - 2, through: 0, by: -1) {
                outer[axis] = rest % shape[axis]
                rest /= shape[axis]
                offset += outer[axis] * strides[axis]
            }
            (dst + row * rowLength).update(from: src + offset, count: rowLength)
        }
        return copy
    }

    static func floats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBytes { dst in
            var input = vImage_Buffer(
                data: array.dataPointer, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
            var output = vImage_Buffer(
                data: dst.baseAddress, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
            vImageConvert_Planar16FtoPlanarF(&input, &output, 0)
        }
        return result
    }

    static func compiledModel(_ base: URL, manager: FileManager) async throws -> URL {
        let compiled = base.appendingPathExtension("mlmodelc")
        if manager.fileExists(atPath: compiled.path) { return compiled }
        let package = base.appendingPathExtension("mlpackage")
        guard manager.fileExists(atPath: package.path) else {
            throw CuaS1FourBError.invalidAsset("missing \(base.lastPathComponent).mlmodelc or .mlpackage")
        }
        return try await MLModel.compileModel(at: package)
    }

    static func firstExisting(_ urls: [URL], what: String) throws -> URL {
        guard let url = urls.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw CuaS1FourBError.invalidAsset("missing \(what) (looked in \(urls.map(\.path)))")
        }
        return url
    }
}
