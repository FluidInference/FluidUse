import CoreML
import Foundation
import os

/// Writes Python from a plain-English task with Qwen2.5-Coder-0.5B-Instruct on Core ML.
///
/// Same graph split as ``ShortReplyManager``: one package holds `prefill` (stateless, fixed prompt length, Neural
/// Engine) and `decode` (stateful KV cache, GPU) over one set of weights. The host tokenizes the Qwen2.5 chat prompt,
/// left-pads it to the prefill length, copies the prefill K/V into the decoder state and decodes greedily until
/// `<|im_end|>`, streaming the text as it grows.
@available(macOS 15.0, iOS 18.0, *)
public actor CodeWriterManager {
    public struct Timing: Sendable {
        public let promptTokens: Int
        public let prefillSeconds: Double
        public let decodeSeconds: Double
        public let generatedTokens: Int
        public var totalSeconds: Double { prefillSeconds + decodeSeconds }
        public var tokensPerSecond: Double { decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0 }
    }

    public struct Completion: Sendable {
        /// Everything the model wrote (markdown, possibly with prose around the code).
        public let text: String
        /// The first ```python block, or the whole text when there is none.
        public let code: String
        /// Generated token ids, including the stop token when one was produced.
        public let tokens: [Int]
        public let timing: Timing
    }

    /// `config.json` next to the package.
    struct Config: Decodable {
        let package: String
        let prefillLength: Int
        let cacheLength: Int
        let layers: Int
        let kvHeads: Int
        let headDim: Int
        let padID: Int
        let stopIDs: [Int]
        let maxNewTokens: Int
        let systemPrompt: String
    }

    private let logger = Logger(subsystem: "FluidUse", category: "CodeWriter")
    private let config: Config
    private let tokenizer: QwenBPETokenizer
    private let prefill: MLModel
    private let decode: MLModel
    private let stopIDs: Set<Int>

    /// Loads `<directory>/config.json`, the package it names, and `tokenizer.json`.
    public static func load(
        from directory: URL, prefillUnits: MLComputeUnits = .cpuAndNeuralEngine,
        decodeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CodeWriterManager {
        let config = try JSONDecoder().decode(
            Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        var package = directory.appendingPathComponent(config.package)
        if package.pathExtension == "mlpackage" {
            package = try await KevManager.compiled(package)  // compile once, keep the .mlmodelc beside the package
        }
        return try CodeWriterManager(
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

    /// One short task through both functions, so the first real call pays no compile cost.
    public func warmUp() async throws {
        _ = try await write(task: "Write a Python function that returns 1.", maxNewTokens: 8)
    }

    /// The Qwen2.5 chat prompt (system turn from `config.json` unless overridden).
    func promptTokens(user: String, system: String?) throws -> [Int] {
        let text =
            "<|im_start|>system\n\(system ?? config.systemPrompt)<|im_end|>\n"
            + "<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
        return try tokenizer.encode(text)
    }

    /// Greedy completion of `task`. `onText` receives the full text written so far after each token (skipping
    /// states that end mid-character).
    public func write(
        task: String, system: String? = nil, maxNewTokens: Int? = nil,
        onText: (@Sendable (String) -> Void)? = nil
    ) async throws -> Completion {
        let prompt = try promptTokens(user: task, system: system)
        let length = config.prefillLength
        guard !task.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw CodeWriterError.emptyTask }
        guard prompt.count <= length else { throw CodeWriterError.taskTooLong(prompt.count, length) }
        // Left-pad to the fixed prefill length; padded cache slots stay masked for every later query.
        let pad = length - prompt.count
        let ids = Array(repeating: config.padID, count: pad) + prompt

        let started = ContinuousClock.now
        let state = decode.makeState()
        let prefillOutput = try Self.predict(prefill, try prefillInputs(ids: ids, pad: pad), state: nil)
        try copyCache(from: prefillOutput, into: state, promptLength: length)
        var logits = try lastLogits(prefillOutput)
        let prefillSeconds = seconds(since: started)

        let decodeStarted = ContinuousClock.now
        let limit = min(maxNewTokens ?? config.maxNewTokens, config.cacheLength - length)
        var generated: [Int] = []
        var position = length
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: config.cacheLength)], dataType: .float32)
        mask.withUnsafeMutableBytes { buffer, _ in
            let values = buffer.bindMemory(to: Float32.self)
            for column in 0..<config.cacheLength { values[column] = column < length ? (column < pad ? -1e4 : 0) : -1e4 }
        }
        while generated.count < limit {
            try Task.checkCancellation()
            let token = argmax(logits)
            generated.append(token)
            if stopIDs.contains(token) { break }
            if let onText {
                let text = tokenizer.decode(generated)
                if !text.hasSuffix("\u{FFFD}") { onText(text) }
            }
            guard generated.count < limit else { break }
            mask.withUnsafeMutableBytes { buffer, _ in buffer.bindMemory(to: Float32.self)[position] = 0 }
            let output = try Self.predict(
                decode, try decodeInputs(token: token, position: position, mask: mask), state: state)
            logits = try lastLogits(output)
            position += 1
        }
        let decodeSeconds = seconds(since: decodeStarted)
        let text = tokenizer.decode(generated.filter { !stopIDs.contains($0) })
        let timing = Timing(
            promptTokens: prompt.count, prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds,
            generatedTokens: generated.count)
        logger.info(
            "wrote \(generated.count) tokens in \(Int(timing.totalSeconds * 1000)) ms (prefill \(Int(prefillSeconds * 1000)) ms)"
        )
        return Completion(text: text, code: Self.extractCode(text), tokens: generated, timing: timing)
    }

    /// The first fenced code block (```python / ```py / ```), or the whole text when there is none. An unclosed
    /// block (generation cut off) runs to the end.
    public static func extractCode(_ text: String) -> String {
        guard let open = text.range(of: #"```[A-Za-z]*\n"#, options: .regularExpression) else {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let rest = text[open.upperBound...]
        let body = rest.range(of: "```").map { rest[..<$0.lowerBound] } ?? rest
        return String(body).trimmingCharacters(in: .newlines)
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

    private func decodeInputs(token: Int, position: Int, mask: MLMultiArray) throws -> MLDictionaryFeatureProvider {
        let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIDs[0] = NSNumber(value: Int32(token))
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
        else { throw CodeWriterError.invalidAsset("prefill output has no keys/values") }
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
                throw CodeWriterError.invalidAsset(
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
                    else { throw CodeWriterError.invalidAsset("unexpected \(name) cache layout \(cache.strides)") }
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
            throw CodeWriterError.invalidAsset("model output has no logits")
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

    private func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}

public enum CodeWriterError: Error, LocalizedError, Sendable {
    case emptyTask
    case taskTooLong(Int, Int)
    case invalidAsset(String)

    public var errorDescription: String? {
        switch self {
        case .emptyTask: return "Describe what the code should do."
        case .taskTooLong(let tokens, let limit): return "The task is too long (\(tokens) of \(limit) prompt tokens)."
        case .invalidAsset(let detail): return "Code-writer model asset problem: \(detail)"
        }
    }
}
