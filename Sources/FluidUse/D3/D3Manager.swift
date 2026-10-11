@preconcurrency import CoreML
import CoreGraphics
import Foundation

/// vLLM Semantic Router Decision 3.0 (d3-lite / d3-nano / d3-mini) on Core ML, text and image requests: the request's images go through the fp32 vision tower once, and
/// every question's row (system prompt, image placeholders, state, question, options) runs the fp16 text graph with
/// the image tokens spliced in at the placeholder positions and 3-D M-RoPE positions. FP32 readout on the host.
///
/// Image preprocessing, the vision tower's inputs and the M-RoPE positions are clef-vision's host code
/// (`ClefImagePreprocessor`, `ClefVisionHost`): same Qwen3.5 vision family.
///
/// The directory holds `coreml_config.json`, `tokenizer.json`, `readout.f32`, `embeddings.f16` ([V, H] fp16),
/// `pos_embed.f32`, `vision_P{patches}.mlpackage` and `d3_E{tokens}.mlpackage` (or their compiled `.mlmodelc`).
@available(macOS 15.0, iOS 18.0, *)
public actor D3Manager {
    private struct TextBucket {
        let length: Int
        let model: MLModel
        let hidden: MLMultiArray
        let cos: MLMultiArray
        let sin: MLMultiArray
        let keyValid: MLMultiArray
        let lastOnehot: MLMultiArray
        let features: MLDictionaryFeatureProvider
    }

    private struct VisionBucket {
        let patches: Int
        let model: MLModel
        let config: ClefVisionConfig
    }

    public nonisolated let modelName: String
    private let tokenizer: QwenBPETokenizer
    private let codes: [String]
    private let readout: [Float]
    private let embeddings: Data
    private let posEmbed: [Float]
    private let hidden: Int
    private let temperature: Double
    private let padID: Int
    private let imageTokenID: Int
    private let mergeSize: Int
    private let rotaryDim: Int
    private let ropeTheta: Double
    private let mropeSection: [Int]
    private let vision: [VisionBucket]
    private let text: [TextBucket]

    private init(
        modelName: String, tokenizer: QwenBPETokenizer, codes: [String], readout: [Float], embeddings: Data,
        posEmbed: [Float], hidden: Int, temperature: Double, padID: Int, imageTokenID: Int, mergeSize: Int,
        rotaryDim: Int, ropeTheta: Double, mropeSection: [Int], vision: [VisionBucket], text: [TextBucket]
    ) {
        self.modelName = modelName
        self.tokenizer = tokenizer
        self.codes = codes
        self.readout = readout
        self.embeddings = embeddings
        self.posEmbed = posEmbed
        self.hidden = hidden
        self.temperature = temperature
        self.padID = padID
        self.imageTokenID = imageTokenID
        self.mergeSize = mergeSize
        self.rotaryDim = rotaryDim
        self.ropeTheta = ropeTheta
        self.mropeSection = mropeSection
        self.vision = vision
        self.text = text
    }

    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws
        -> D3Manager
    {
        func file(_ name: String) -> URL { directory.appendingPathComponent(name) }
        guard
            let config = try JSONSerialization.jsonObject(with: Data(contentsOf: file("coreml_config.json")))
                as? [String: Any],
            let modelName = config["model_name"] as? String, let codes = config["codes"] as? [String],
            codes.count == D3Prompt.maxOptions, let pad = config["pad_token_id"] as? Int,
            let imageToken = config["image_token_id"] as? Int, let start = config["vision_start_token_id"] as? Int,
            let end = config["vision_end_token_id"] as? Int, let hidden = config["hidden_size"] as? Int,
            let vocab = config["vocab_size"] as? Int, let rope = config["rope"] as? [String: Any],
            let rotary = rope["rotary_dim"] as? Int, let theta = (rope["theta"] as? NSNumber)?.doubleValue,
            let section = rope["mrope_section"] as? [Int], let v = config["vision"] as? [String: Any],
            let image = config["image"] as? [String: Any], let visionBuckets = config["vision_buckets"] as? [Int],
            let textBuckets = config["text_buckets"] as? [Int]
        else { throw D3Error.invalidAsset("coreml_config.json is missing fields") }
        func int(_ key: String) throws -> Int {
            guard let value = (v[key] as? NSNumber)?.intValue else { throw D3Error.invalidAsset("vision.\(key)") }
            return value
        }
        let merge = try int("merge_size")
        let processing = ClefVisionConfig.ImageProcessing(
            minPixels: (image["min_pixels"] as? Int) ?? 65_536, maxPixels: (image["max_pixels"] as? Int) ?? 1_638_400,
            mean: ((image["mean"] as? [NSNumber]) ?? [0.5, 0.5, 0.5]).map(\.floatValue),
            std: ((image["std"] as? [NSNumber]) ?? [0.5, 0.5, 0.5]).map(\.floatValue),
            rescale: (image["rescale"] as? NSNumber)?.floatValue ?? 1 / 255)
        let readout = try Data(contentsOf: file("readout.f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let posEmbed = try Data(contentsOf: file("pos_embed.f32")).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let embeddings = try Data(contentsOf: file("embeddings.f16"), options: .alwaysMapped)
        guard embeddings.count == vocab * hidden * 2 else { throw D3Error.invalidAsset("embeddings.f16 size") }

        func compiled(_ name: String) async throws -> URL {
            let c = file(name + ".mlmodelc")
            if FileManager.default.fileExists(atPath: c.path) { return c }
            // The release may symlink packages; compile next to the link, inside the release.
            let temporary = try await MLModel.compileModel(at: file(name + ".mlpackage").resolvingSymlinksInPath())
            try? FileManager.default.moveItem(at: temporary, to: c)
            return FileManager.default.fileExists(atPath: c.path) ? c : temporary
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        var visions: [VisionBucket] = []
        for patches in visionBuckets.sorted() {
            let shape = ClefVisionConfig.VisionShape(
                patches: patches, patchSize: try int("patch_size"), mergeSize: merge,
                temporalPatch: try int("temporal_patch_size"), hidden: try int("hidden_size"),
                headDim: try int("hidden_size") / (try int("num_heads")),
                depthPositions: try int("num_position_embeddings"),
                ropeTheta: (v["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000)
            let clef = ClefVisionConfig(
                imageTokenID: imageToken, visionStartID: start, visionEndID: end, padID: pad, vocabSize: vocab,
                hiddenSize: hidden, rotaryDim: rotary, ropeTheta: theta, mropeSection: section, lmBuckets: textBuckets,
                vision: shape, image: processing)
            let model = try await MLModel.load(
                contentsOf: try await compiled("vision_P\(patches)"), configuration: configuration)
            visions.append(VisionBucket(patches: patches, model: model, config: clef))
        }
        var texts: [TextBucket] = []
        for length in textBuckets.sorted() {
            let model = try await MLModel.load(
                contentsOf: try await compiled("d3_E\(length)"), configuration: configuration)
            let l = NSNumber(value: length)
            let hiddenArray = try MLMultiArray(shape: [1, l, NSNumber(value: hidden)], dataType: .float16)
            let cos = try MLMultiArray(shape: [l, NSNumber(value: rotary)], dataType: .float16)
            let sin = try MLMultiArray(shape: [l, NSNumber(value: rotary)], dataType: .float16)
            let keyValid = try MLMultiArray(shape: [l], dataType: .float16)
            let lastOnehot = try MLMultiArray(shape: [1, l], dataType: .float16)
            let features = try MLDictionaryFeatureProvider(dictionary: [
                "hidden": hiddenArray, "cos": cos, "sin": sin, "key_valid": keyValid, "last_onehot": lastOnehot,
            ])
            texts.append(
                TextBucket(
                    length: length, model: model, hidden: hiddenArray, cos: cos, sin: sin, keyValid: keyValid,
                    lastOnehot: lastOnehot, features: features))
        }
        guard !visions.isEmpty, !texts.isEmpty else { throw D3Error.invalidAsset("no buckets") }
        return D3Manager(
            modelName: modelName,
            tokenizer: try QwenBPETokenizer(tokenizerJsonURL: file("tokenizer.json")), codes: codes, readout: readout,
            embeddings: embeddings, posEmbed: posEmbed, hidden: hidden,
            temperature: (config["temperature"] as? NSNumber)?.doubleValue ?? 1, padID: pad, imageTokenID: imageToken,
            mergeSize: merge, rotaryDim: rotary, ropeTheta: theta, mropeSection: section, vision: visions, text: texts)
    }

    /// One encoded image: its merged tokens [count × hidden] and grid (in patches).
    public struct EncodedImage: Sendable {
        let tokens: [Float]
        let gridH: Int
        let gridW: Int
        public let milliseconds: Double
        public var count: Int { gridH * gridW / 4 }
    }

    /// The vision tower for one image (reuse the result for every question about it).
    public func encode(_ image: CGImage) throws -> EncodedImage {
        let first = vision[0]
        let patches = try ClefImagePreprocessor.patches(from: image, config: first.config)
        guard let bucket = vision.first(where: { $0.patches >= patches.count }) else {
            throw D3Error.tooLong(needed: patches.count, budget: vision.last?.patches ?? 0)
        }
        let inputs = try posEmbed.withUnsafeBufferPointer {
            try ClefVisionHost.visionInputs(patches, posEmbed: $0, config: bucket.config)
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let output = try bucket.model.prediction(from: inputs.features)
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
        guard let array = output.featureValue(for: "tokens")?.multiArrayValue else {
            throw D3Error.invalidOutput("vision tokens")
        }
        let count = inputs.mergedTokens * hidden
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        return EncodedImage(
            tokens: Array(UnsafeBufferPointer(start: pointer, count: count)), gridH: patches.gridH,
            gridW: patches.gridW, milliseconds: milliseconds)
    }

    /// The token ids of one question's row about `images` (for parity checks).
    public func tokens(state: OrderedJSON, question: D3Question, images: [EncodedImage]) throws -> [Int] {
        try tokenizer.encode(
            try D3Prompt.render(state: state, question: question, codes: codes, imageTokens: images.map(\.count)))
    }

    /// Answers each question from its own row, every row seeing all `images` (none for a text request).
    public func answer(
        state: OrderedJSON, images: [EncodedImage], questions: [(id: String, question: D3Question)]
    )
        throws -> [D3Answer]
    {
        try questions.map { item in
            let ids = try tokens(state: state, question: item.question, images: images)
            let started = DispatchTime.now().uptimeNanoseconds
            let final = try run(ids: ids, images: images)
            let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
            let count = item.question.keys.count
            var logits = (0..<count).map { row -> Double in
                var sum = 0.0
                for k in 0..<hidden { sum += Double(readout[row * hidden + k]) * Double(final[k]) }
                return sum / temperature
            }
            let peak = logits.max() ?? 0
            logits = logits.map { exp($0 - peak) }
            let total = logits.reduce(0, +)
            let probabilities = logits.map { $0 / total }
            guard probabilities.allSatisfy(\.isFinite) else { throw D3Error.invalidOutput("non-finite logits") }
            return D3Answer(
                id: item.id, kind: item.question.kind, keys: item.question.keys, probabilities: probabilities,
                tokens: ids.count, milliseconds: milliseconds)
        }
    }

    private func run(ids: [Int], images: [EncodedImage]) throws -> [Float] {
        let n = ids.count
        guard let bucket = text.first(where: { $0.length >= n }) else {
            throw D3Error.tooLong(needed: n, budget: text.last?.length ?? 0)
        }
        let length = bucket.length
        // Embeddings, with each image's tokens over its run of image-pad ids.
        let x = bucket.hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        // The memory-mapped table stays valid as long as `embeddings` does.
        let table = (embeddings as NSData).bytes.assumingMemoryBound(to: Float16.self)
        for position in 0..<length {
            let token = position < n ? ids[position] : padID
            (x + position * hidden).update(from: table + token * hidden, count: hidden)
        }
        var image = 0
        var position = 0
        while position < n {
            guard ids[position] == imageTokenID else {
                position += 1
                continue
            }
            let tokens = images[image].tokens
            let rows = images[image].count
            for r in 0..<rows {
                let dst = x + (position + r) * hidden
                for d in 0..<hidden { dst[d] = Float16(tokens[r * hidden + d]) }
            }
            position += rows
            image += 1
        }
        let valid = bucket.keyValid.dataPointer.assumingMemoryBound(to: Float16.self)
        let last = bucket.lastOnehot.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<length {
            valid[i] = i < n ? 1 : 0
            last[i] = i == n - 1 ? 1 : 0
        }
        // Interleaved M-RoPE: frequency i reads axis t, h (i % 3 == 1) or w (i % 3 == 2) within the sections.
        let positions = ClefVisionHost.mropePositions(
            inputIDs: ids, imageGrids: images.map { ($0.gridH, $0.gridW) }, imageTokenID: imageTokenID,
            mergeSize: mergeSize)
        let half = rotaryDim / 2
        var axisOf = [Int](repeating: 0, count: half)
        for i in 0..<half {
            if i % 3 == 1, i < mropeSection[1] * 3 { axisOf[i] = 1 }
            if i % 3 == 2, i < mropeSection[2] * 3 { axisOf[i] = 2 }
        }
        let cos = bucket.cos.dataPointer.assumingMemoryBound(to: Float16.self)
        let sin = bucket.sin.dataPointer.assumingMemoryBound(to: Float16.self)
        for p in 0..<length {
            for i in 0..<half {
                // Padding repeats the last position: masked keys and causal tails cannot reach real tokens.
                let pos = positions[axisOf[i]][min(p, n - 1)]
                let angle = Double(pos) / pow(ropeTheta, Double(2 * i) / Double(rotaryDim))
                let c = Float16(Foundation.cos(angle))
                let s = Float16(Foundation.sin(angle))
                cos[p * rotaryDim + i] = c
                cos[p * rotaryDim + i + half] = c
                sin[p * rotaryDim + i] = s
                sin[p * rotaryDim + i + half] = s
            }
        }
        let output = try bucket.model.prediction(from: bucket.features)
        guard let array = output.featureValue(for: "final")?.multiArrayValue, array.count == hidden else {
            throw D3Error.invalidOutput("missing final hidden")
        }
        return (0..<hidden).map { array[$0].floatValue }
    }
}
