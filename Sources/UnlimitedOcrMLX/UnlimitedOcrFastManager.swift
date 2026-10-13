import CoreML
import FluidUse
import Foundation
import MLX
import os

/// Unlimited-OCR with the text decoder in MLX (~5x the Core ML decoder's tokens/s) and the page encoder in Core ML.
/// Same contract as `UnlimitedOcrManager`: pages in, one generation for the whole document, `UnlimitedOcrEvent`s out.
///
/// Bundle layout: `config.json`, `tokenizer.json`, `UnlimitedOcrPageEncoder.mlpackage`, `mlx/config.json`,
/// `mlx/decoder.safetensors`. The decoder keeps upstream's R-SWA cache (whole prefix + last 128 generated tokens), so
/// memory and speed stay flat over long documents.
public actor UnlimitedOcrFastManager {
    struct RuntimeConfig: Decodable {
        let ring: Int
        let pageTokens: Int
        let bosId: Int
        let eosId: Int
        let imageTokenId: Int
        let promptSuffixIds: [String: [Int]]
        let ngramSize: Int
        let ngramWindow: Int
        let maxLength: Int
        let encoder: String

        enum CodingKeys: String, CodingKey {
            case ring, encoder
            case pageTokens = "page_tokens"
            case bosId = "bos_id"
            case eosId = "eos_id"
            case imageTokenId = "image_token_id"
            case promptSuffixIds = "prompt_suffix_ids"
            case ngramSize = "ngram_size"
            case ngramWindow = "ngram_window"
            case maxLength = "max_length"
        }
    }

    private let config: RuntimeConfig
    private let encoder: MLModel
    private let decoder: OcrDecoder
    private let tokenDecoder: UnlimitedOcrTokenDecoder
    private nonisolated let pauseFlag = OSAllocatedUnfairLock(initialState: false)

    /// Pages per read. The R-SWA cache has no fixed buckets; 16 pages (4,373 prefix tokens) leaves ~28K tokens of the
    /// 32K context for output.
    public nonisolated let maxPages = 16

    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws
        -> UnlimitedOcrFastManager
    {
        let config = try JSONDecoder().decode(
            RuntimeConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let tokenDecoder = try UnlimitedOcrTokenDecoder(
            tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"),
            skipping: [config.bosId, config.eosId])
        let source = directory.appendingPathComponent(config.encoder)
        let compiled = directory.appendingPathComponent("compiled")
            .appendingPathComponent(source.deletingPathExtension().lastPathComponent + ".mlmodelc")
        if !FileManager.default.fileExists(atPath: compiled.path) {
            try FileManager.default.createDirectory(
                at: compiled.deletingLastPathComponent(), withIntermediateDirectories: true)
            let temporary = try await MLModel.compileModel(at: source)
            try FileManager.default.moveItem(at: temporary, to: compiled)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let encoder = try await MLModel.load(contentsOf: compiled, configuration: configuration)
        // MLX keeps freed buffers for reuse; cap that pool so long reads don't park GBs of stale cache copies.
        Memory.cacheLimit = 512 * 1024 * 1024
        let decoder = try OcrDecoder.load(directory: directory.appendingPathComponent("mlx"))
        return UnlimitedOcrFastManager(config: config, encoder: encoder, decoder: decoder, tokenDecoder: tokenDecoder)
    }

    private init(config: RuntimeConfig, encoder: MLModel, decoder: OcrDecoder, tokenDecoder: UnlimitedOcrTokenDecoder) {
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.tokenDecoder = tokenDecoder
    }

    public nonisolated func pause() { pauseFlag.withLock { $0 = true } }
    public nonisolated func resume() { pauseFlag.withLock { $0 = false } }
    public nonisolated var isPaused: Bool { pauseFlag.withLock { $0 } }

    /// One short pass through the encoder and decoder so the first real read does not pay MLX / Core ML setup.
    public func warm() throws {
        let side = UnlimitedOcrPagePreprocessor.size
        _ = try encode(UnlimitedOcrPage(pixels: [Float](repeating: 0, count: 3 * side * side)))
        let caches = (0..<decoder.config.layers).map { _ in RSWACache(ring: config.ring) }
        eval(decoder(decoder.embed([config.bosId, config.bosId]), caches: caches))
    }

    public nonisolated func read(
        pages: [UnlimitedOcrPage], prompt: String = "multi", maxNewTokens: Int? = nil
    )
        -> AsyncThrowingStream<UnlimitedOcrEvent, Error>
    {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(
                        pages: pages, prompt: prompt, maxNewTokens: maxNewTokens, emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func encode(_ page: UnlimitedOcrPage) throws -> MLXArray {
        let side = UnlimitedOcrPagePreprocessor.size as NSNumber
        let pixels = try MLMultiArray(shape: [1, 3, side, side], dataType: .float32)
        page.pixels.withUnsafeBufferPointer {
            pixels.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: $0.count)
        }
        let output = try encoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["pixels": MLFeatureValue(multiArray: pixels)]))
        guard let embeds = output.featureValue(for: "page_embeds")?.multiArrayValue else {
            throw UnlimitedOcrError.prediction("page encoder returned no page_embeds")
        }
        let rows = embeds.shape[0].intValue
        let cols = embeds.shape[1].intValue
        let rowStride = embeds.strides[0].intValue
        let colStride = embeds.strides[1].intValue
        var values = [Float](repeating: 0, count: rows * cols)
        let src = embeds.dataPointer.assumingMemoryBound(to: Float.self)
        for r in 0..<rows { for c in 0..<cols { values[r * cols + c] = src[r * rowStride + c * colStride] } }
        return MLXArray(values, [rows, cols])
    }

    private func run(
        pages: [UnlimitedOcrPage], prompt: String, maxNewTokens: Int?,
        emit: @Sendable (UnlimitedOcrEvent) -> Void
    ) async throws {
        guard !pages.isEmpty else { throw UnlimitedOcrError.invalidInput("no pages") }
        guard pages.count <= maxPages else { throw UnlimitedOcrError.invalidInput("at most \(maxPages) pages") }
        guard let suffix = config.promptSuffixIds[prompt] else {
            throw UnlimitedOcrError.invalidInput("unknown prompt \(prompt)")
        }
        resume()

        let encodeStart = Date()
        var imageRows: [MLXArray] = []
        for (index, page) in pages.enumerated() {
            let pageStart = Date()
            imageRows.append(try encode(page))
            emit(.pageEncoded(index: index, milliseconds: Date().timeIntervalSince(pageStart) * 1000))
            try Task.checkCancellation()
        }
        let encodeMs = Date().timeIntervalSince(encodeStart) * 1000

        // Prefix: BOS, page embeddings, prompt text — upstream's base-mode multi-page layout.
        let bos = decoder.embed([config.bosId])
        let dtype = bos.dtype
        let prefix = concatenated(
            [bos, concatenated(imageRows, axis: 0).asType(dtype)[.newAxis], decoder.embed(suffix)], axis: 1)
        let n = prefix.dim(1)
        var sequence = [config.bosId] + Array(repeating: config.imageTokenId, count: n - 1 - suffix.count) + suffix
        let caches = (0..<decoder.config.layers).map { _ in RSWACache(ring: config.ring) }

        let prefillStart = Date()
        var logits = decoder(prefix, caches: caches)
        eval(logits)
        let prefillMs = Date().timeIntervalSince(prefillStart) * 1000
        emit(.prefilled(tokens: n, capacity: n, milliseconds: prefillMs))

        var tokens = tokenDecoder
        var limit = config.maxLength - n
        if let maxNewTokens { limit = min(limit, maxNewTokens) }
        var produced = 0
        var stoppedAtEOS = false
        var pausedSeconds = 0.0
        let decodeStart = Date()
        for _ in 0..<limit {
            let candidates = argPartition(-logits, kth: 15)[..<16]
            let ids = candidates.asArray(Int32.self)
            let values = logits[candidates].asType(.float32).asArray(Float.self)
            let ranked = zip(values, ids).sorted { $0.0 > $1.0 }.map { Int($0.1) }
            let banned = bannedTokens(sequence)
            guard let token = ranked.first(where: { !banned.contains($0) }) else {
                throw UnlimitedOcrError.prediction("every top-16 candidate is banned")
            }
            sequence.append(token)
            produced += 1
            let text = tokens.append(token)
            if !text.isEmpty { emit(.text(text)) }
            if token == config.eosId {
                stoppedAtEOS = true
                break
            }
            try Task.checkCancellation()
            if isPaused {
                let pauseStart = Date()
                while isPaused { try await Task.sleep(nanoseconds: 50_000_000) }
                pausedSeconds += Date().timeIntervalSince(pauseStart)
            }
            logits = decoder(decoder.embed([token]), caches: caches)
        }
        Memory.clearCache()
        emit(
            .finished(
                UnlimitedOcrStats(
                    pages: pages.count, promptTokens: n, outputTokens: produced, encodeMilliseconds: encodeMs,
                    prefillMilliseconds: prefillMs,
                    decodeMilliseconds: (Date().timeIntervalSince(decodeStart) - pausedSeconds) * 1000,
                    stoppedAtEOS: stoppedAtEOS)))
    }

    /// Upstream `SlidingWindowNoRepeatNgramProcessor` (35-grams within the last 1,024 tokens, prompt included).
    private func bannedTokens(_ sequence: [Int]) -> Set<Int> {
        let size = config.ngramSize
        guard sequence.count >= size else { return [] }
        let start = max(0, sequence.count - config.ngramWindow)
        let end = sequence.count - size + 1
        guard end > start else { return [] }
        let prefixStart = sequence.count - (size - 1)
        var banned = Set<Int>()
        for i in start..<end where sequence[i] == sequence[prefixStart] {
            var same = true
            for j in 1..<(size - 1) where sequence[i + j] != sequence[prefixStart + j] {
                same = false
                break
            }
            if same { banned.insert(sequence[i + size - 1]) }
        }
        return banned
    }
}
