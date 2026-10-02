import CoreML
import Foundation
import os

/// Drafts one short reply to a social post with the FluidUse short-reply model (Qwen3-0.6B fine-tune) on Core ML.
///
/// The package holds two functions over one set of weights: `prefill` (stateless, fixed prompt length, runs on the
/// Neural Engine) and `decode` (stateful KV cache, runs on the GPU). The host tokenizes the chat prompt, left-pads it to
/// the prefill length, copies the prefill K/V into the decoder's state, then decodes greedily until `<|im_end|>`.
@available(macOS 15.0, iOS 18.0, *)
public actor ShortReplyManager {
    public struct Timing: Sendable {
        public let prefillSeconds: Double
        public let decodeSeconds: Double
        public let generatedTokens: Int
        public var totalSeconds: Double { prefillSeconds + decodeSeconds }
    }

    public struct Draft: Sendable {
        public let reply: String
        public let timing: Timing
        /// The post actually sent to the model (links removed; cut to the prefill length when too long).
        public let post: String
        public let trimmed: Bool
    }

    /// `config.json` next to the package.
    struct Config: Decodable {
        let package: String
        let prefillLength: Int
        let cacheLength: Int
        let decodeLengths: [Int]
        let layers: Int
        let kvHeads: Int
        let headDim: Int
        let padID: Int
        let stopIDs: [Int]
        let maxNewTokens: Int
        let systemPrompt: String
        let maxPostCharacters: Int
    }

    private let logger = Logger(subsystem: "FluidUse", category: "ShortReply")
    private let config: Config
    private let tokenizer: QwenBPETokenizer
    private let prefill: MLModel
    private let decode: MLModel
    private let stopIDs: Set<Int>

    /// Loads `<directory>/config.json`, the package it names, and `tokenizer.json`.
    public static func load(
        from directory: URL, prefillUnits: MLComputeUnits = .cpuAndNeuralEngine,
        decodeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> ShortReplyManager {
        let config = try JSONDecoder().decode(
            Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        var package = directory.appendingPathComponent(config.package)
        if package.pathExtension == "mlpackage" {
            package = try await KevManager.compiled(package)  // compile once, keep the .mlmodelc beside the package
        }
        return try ShortReplyManager(
            config: config, compiled: package, tokenizer: directory.appendingPathComponent("tokenizer.json"),
            prefillUnits: prefillUnits, decodeUnits: decodeUnits)
    }

    /// Models are created here, inside the actor, so the non-Sendable `MLModel`s never cross an isolation boundary.
    init(
        config: Config, compiled: URL, tokenizer: URL, prefillUnits: MLComputeUnits, decodeUnits: MLComputeUnits
    ) throws {
        self.config = config
        self.tokenizer = try QwenBPETokenizer(tokenizerJsonURL: tokenizer)
        let prefillConfiguration = MLModelConfiguration()
        prefillConfiguration.computeUnits = prefillUnits
        prefillConfiguration.functionName = "prefill"
        let decodeConfiguration = MLModelConfiguration()
        decodeConfiguration.computeUnits = decodeUnits
        decodeConfiguration.functionName = "decode"
        self.prefill = try MLModel(contentsOf: compiled, configuration: prefillConfiguration)
        self.decode = try MLModel(contentsOf: compiled, configuration: decodeConfiguration)
        self.stopIDs = Set(config.stopIDs)
    }

    /// One prompt through both functions, so the first real call pays no compile cost.
    public func warmUp() async throws {
        _ = try await draft(for: "Warm-up post.")
    }

    /// The prompt exactly as `reply.py` builds it (Qwen3 chat template, thinking disabled).
    func promptTokens(for post: String) throws -> [Int] {
        let text =
            "<|im_start|>system\n\(config.systemPrompt)<|im_end|>\n"
            + "<|im_start|>user\nPost: \(post)\nReply:<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return try tokenizer.encode(text)
    }

    /// Whitespace-normalized, links dropped, and cut (by tokens, from the end) until the full prompt fits the prefill
    /// length. The fit is checked on the re-tokenized prompt, since a cut can land mid-word or mid-merge.
    func preparePost(_ rawPost: String) throws -> (post: String, trimmed: Bool) {
        var post = Self.stripLinks(rawPost).split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard !post.isEmpty else { throw ShortReplyError.emptyPost }
        var trimmed = false
        if post.count > config.maxPostCharacters {
            post = String(post.prefix(config.maxPostCharacters))
            trimmed = true
        }
        let length = config.prefillLength
        if try promptTokens(for: post).count <= length { return (post, trimmed) }
        var keep = max(1, try tokenizer.encode(post).count - (try promptTokens(for: post).count - length) - 1)
        while keep > 0 {
            let cut =
                tokenizer.decode(Array(try tokenizer.encode(post).prefix(keep)))
                .trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "\u{FFFD}", with: "") + "…"
            if try promptTokens(for: cut).count <= length { return (cut, true) }
            keep -= 2
        }
        throw ShortReplyError.invalidAsset("prefill length too short for the system prompt")
    }

    /// True when the reply repeats a run of five or more consecutive words from the post (mirrors `reply.py`).
    static func echoes(_ post: String, _ reply: String, run: Int = 5) -> Bool {
        let postWords = post.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "'" }.map(String.init)
        let replyWords = reply.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "'" }.map(String.init)
        guard replyWords.count >= run, postWords.count >= run else { return false }
        for start in 0...(replyWords.count - run) {
            let window = Array(replyWords[start..<start + run])
            for index in 0...(postWords.count - run) where Array(postWords[index..<index + run]) == window {
                return true
            }
        }
        return false
    }

    static func stripLinks(_ text: String) -> String {
        text.replacingOccurrences(
            of: #"(https?://\S+|\b(?:pic\.x\.com|pic\.twitter\.com|t\.co|x\.com)/\S+)"#, with: "",
            options: .regularExpression)
    }

    /// `variation` 0 decodes greedily (the benchmarked behavior); higher values sample (temperature 0.7, top-p 0.9)
    /// with a seed derived from the value, so "regenerate" gives a different, repeatable reply. `avoiding` rejects
    /// replies equal to earlier drafts (up to a few attempts). A greedy reply that parrots the post is replaced by the
    /// first seeded sample that does not, as `reply.py` does.
    public func draft(
        for rawPost: String, variation: Int = 0, avoiding previous: Set<String> = []
    ) async throws -> Draft {
        if variation == 0 {
            let greedy = try await decodeDraft(for: rawPost, variation: 0)
            guard Self.echoes(greedy.post, greedy.reply) else { return greedy }
            for attempt in 1...6 {
                let sampled = try await decodeDraft(for: rawPost, variation: attempt)
                if !Self.echoes(sampled.post, sampled.reply) { return sampled }
            }
            return greedy
        }
        var last: Draft?
        for attempt in 0..<4 {
            let draft = try await decodeDraft(for: rawPost, variation: variation + attempt * 1000)
            if !previous.contains(draft.reply) { return draft }
            last = draft
        }
        return last!
    }

    private func decodeDraft(for rawPost: String, variation: Int) async throws -> Draft {
        let (post, trimmed) = try preparePost(rawPost)
        var rng = SeededGenerator(seed: UInt64(20_260_929 &+ variation))
        let tokens = try promptTokens(for: post)
        let length = config.prefillLength
        guard tokens.count <= length else { throw ShortReplyError.postTooLong(config.maxPostCharacters) }
        // Left-pad to the fixed prefill length; padded cache slots stay masked for every later query.
        let pad = length - tokens.count
        let ids = Array(repeating: config.padID, count: pad) + tokens

        let started = ContinuousClock.now
        let state = decode.makeState()
        let prefillOutput = try Self.predict(prefill, try prefillInputs(ids: ids, pad: pad), state: nil)
        try copyCache(from: prefillOutput, into: state, promptLength: length)
        var logits = try lastLogits(prefillOutput)
        let prefillSeconds = seconds(since: started)

        let decodeStarted = ContinuousClock.now
        var generated: [Int] = []
        var position = length
        while generated.count < config.maxNewTokens {
            let token = variation == 0 ? argmax(logits) : sample(logits, temperature: 0.7, topP: 0.9, using: &rng)
            if stopIDs.contains(token) { break }
            generated.append(token)
            let output = try Self.predict(
                decode, try decodeInputs(token: token, position: position, pad: pad), state: state)
            logits = try lastLogits(output)
            position += 1
        }
        let decodeSeconds = seconds(since: decodeStarted)
        let reply = ShortReplyManager.cleanReply(tokenizer.decode(generated))
        guard !reply.isEmpty else { throw ShortReplyError.emptyReply }
        let timing = Timing(
            prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds, generatedTokens: generated.count)
        logger.info(
            "reply in \(Int(timing.totalSeconds * 1000)) ms (prefill \(Int(prefillSeconds * 1000)), \(generated.count) tokens)"
        )
        return Draft(reply: reply, timing: timing, post: post, trimmed: trimmed)
    }

    /// Synchronous prediction (an async context would pick Core ML's async overload). Called only from the actor,
    /// which serializes it: Core ML's synchronous prediction is not thread-safe across callers.
    private static func predict(
        _ model: MLModel, _ input: MLFeatureProvider, state: MLState?
    ) throws -> MLFeatureProvider {
        if let state { return try model.prediction(from: input, using: state) }
        return try model.prediction(from: input)
    }

    // MARK: - Feature providers

    private func prefillInputs(ids: [Int], pad: Int) throws -> MLDictionaryFeatureProvider {
        let length = ids.count
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        for (index, id) in ids.enumerated() { inputIDs[index] = NSNumber(value: Int32(id)) }
        // Causal mask over the prompt; padded columns masked for every row.
        let mask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: length), NSNumber(value: length)], dataType: .float32)
        mask.withUnsafeMutableBytes { buffer, _ in
            let values = buffer.bindMemory(to: Float32.self)
            for row in 0..<length {
                for column in 0..<length {
                    values[row * length + column] = (column > row || column < pad) ? -1e4 : 0
                }
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIDs, "attention_mask": mask])
    }

    private func decodeInputs(token: Int, position: Int, pad: Int) throws -> MLDictionaryFeatureProvider {
        let cache = config.cacheLength
        guard position < cache else { throw ShortReplyError.postTooLong(config.maxPostCharacters) }
        let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIDs[0] = NSNumber(value: Int32(token))
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: cache)], dataType: .float32)
        mask.withUnsafeMutableBytes { buffer, _ in
            let values = buffer.bindMemory(to: Float32.self)
            for column in 0..<cache { values[column] = (column < pad || column > position) ? -1e4 : 0 }
        }
        let cachePosition = try MLMultiArray(shape: [1], dataType: .int32)
        cachePosition[0] = NSNumber(value: Int32(position))
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIDs, "attention_mask": mask, "cache_position": cachePosition,
        ])
    }

    /// Prefill returns `keys` / `values` as [layers, kvHeads, prompt, headDim] fp16; the decoder state holds
    /// [1, kvHeads, cache, headDim] fp16 per layer.
    private func copyCache(from output: MLFeatureProvider, into state: MLState, promptLength: Int) throws {
        guard let keys = output.featureValue(for: "keys")?.multiArrayValue,
            let values = output.featureValue(for: "values")?.multiArrayValue
        else { throw ShortReplyError.invalidAsset("prefill output has no keys/values") }
        let headDim = config.headDim
        let kvHeads = config.kvHeads
        let rowElements = promptLength * headDim
        for (name, source) in [("k", keys), ("v", values)] {
            // Prefill outputs are fp16 and may carry padded strides; copy head by head using the array's own strides.
            guard source.dataType == .float16, source.shape.count == 4,
                source.shape[1].intValue == kvHeads, source.shape[2].intValue == promptLength,
                source.shape[3].intValue == headDim, source.strides[3].intValue == 1,
                source.strides[2].intValue == headDim
            else {
                throw ShortReplyError.invalidAsset(
                    "unexpected prefill \(name) layout \(source.shape) / \(source.strides)")
            }
            let layerStride = source.strides[0].intValue
            let headStride = source.strides[1].intValue
            for layer in 0..<config.layers {
                // One layer's [kvHeads, prompt, headDim] as a plain (Sendable) buffer, each head's rows contiguous.
                let block: [UInt16] = source.withUnsafeBytes { origin in
                    let base = origin.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    var block = [UInt16](repeating: 0, count: kvHeads * rowElements)
                    for head in 0..<kvHeads {
                        let start = layer * layerStride + head * headStride
                        for element in 0..<rowElements { block[head * rowElements + element] = base[start + element] }
                    }
                    return block
                }
                try state.withMultiArray(for: "\(name)_cache_\(layer)") { cache in
                    guard cache.dataType == .float16, cache.strides[3].intValue == 1,
                        cache.strides[2].intValue == headDim
                    else { throw ShortReplyError.invalidAsset("unexpected \(name) cache layout \(cache.strides)") }
                    let headStride = cache.strides[1].intValue
                    cache.withUnsafeMutableBytes { destination, _ in
                        let dst = destination.baseAddress!.assumingMemoryBound(to: UInt16.self)
                        block.withUnsafeBufferPointer { src in
                            for head in 0..<kvHeads {
                                memcpy(
                                    dst.advanced(by: head * headStride),
                                    src.baseAddress!.advanced(by: head * rowElements),
                                    rowElements * MemoryLayout<UInt16>.size)
                            }
                        }
                    }
                }
            }
        }
    }

    private func lastLogits(_ output: MLFeatureProvider) throws -> MLMultiArray {
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ShortReplyError.invalidAsset("model output has no logits")
        }
        return logits
    }

    private func argmax(_ logits: MLMultiArray) -> Int {
        let count = logits.count
        return logits.withUnsafeBytes { buffer in
            switch logits.dataType {
            case .float32:
                let values = buffer.bindMemory(to: Float32.self)
                var best = 0
                for index in 1..<count where values[index] > values[best] { best = index }
                return best
            default:
                let values = buffer.bindMemory(to: Float16.self)
                var best = 0
                for index in 1..<count where values[index] > values[best] { best = index }
                return best
            }
        }
    }

    /// Temperature + nucleus sampling over the vocabulary.
    private func sample(
        _ logits: MLMultiArray, temperature: Float, topP: Float, using rng: inout SeededGenerator
    ) -> Int {
        let count = logits.count
        var scores = [Float](repeating: 0, count: count)
        logits.withUnsafeBytes { buffer in
            switch logits.dataType {
            case .float32:
                let values = buffer.bindMemory(to: Float32.self)
                for index in 0..<count { scores[index] = values[index] / temperature }
            default:
                let values = buffer.bindMemory(to: Float16.self)
                for index in 0..<count { scores[index] = Float(values[index]) / temperature }
            }
        }
        let peak = scores.max() ?? 0
        var probabilities = scores.map { exp($0 - peak) }
        let total = probabilities.reduce(0, +)
        for index in 0..<count { probabilities[index] /= total }
        // Nucleus: keep the smallest set of top tokens whose mass reaches topP (256 candidates is plenty here).
        let candidates = probabilities.indices.sorted { probabilities[$0] > probabilities[$1] }.prefix(256)
        var kept: [(Int, Float)] = []
        var mass: Float = 0
        for index in candidates {
            kept.append((index, probabilities[index]))
            mass += probabilities[index]
            if mass >= topP { break }
        }
        guard mass.isFinite, mass > 0 else { return argmax(logits) }
        var draw = Float.random(in: 0..<mass, using: &rng)
        for (index, probability) in kept {
            draw -= probability
            if draw <= 0 { return index }
        }
        return kept.last?.0 ?? argmax(logits)
    }

    private func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    /// First non-empty line, stripped of a leading "Reply:" / bullet and surrounding quotes (mirrors `reply.py`).
    static func cleanReply(_ raw: String) -> String {
        guard
            var line = raw.split(whereSeparator: \.isNewline).map({ $0.trimmingCharacters(in: .whitespaces) })
                .first(where: { !$0.isEmpty })
        else { return "" }
        if let range = line.range(of: #"^(reply\s*:\s*|[-•]\s*)"#, options: [.regularExpression, .caseInsensitive]) {
            line.removeSubrange(range)
        }
        return line.trimmingCharacters(in: CharacterSet(charactersIn: " \"'“”"))
    }
}

/// Small deterministic generator (SplitMix64) so a given variation always yields the same sampled reply.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public enum ShortReplyError: Error, LocalizedError, Sendable {
    case emptyPost
    case emptyReply
    case postTooLong(Int)
    case invalidAsset(String)

    public var errorDescription: String? {
        switch self {
        case .emptyPost: return "Select a post first."
        case .emptyReply: return "The model did not produce a reply."
        case .postTooLong(let limit): return "The post is too long (\(limit) character limit)."
        case .invalidAsset(let detail): return "Short-reply model asset problem: \(detail)"
        }
    }
}
