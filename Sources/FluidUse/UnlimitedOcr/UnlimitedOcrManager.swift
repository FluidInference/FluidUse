import CoreML
import Foundation

/// Baidu Unlimited-OCR (3B MoE, ~0.5B active) on Core ML: several document pages in, structured markdown out, in one
/// generation (upstream's "Multi page parsing." mode).
///
/// - Page encoder (SAM-B + CLIP-L + projector, fp16, GPU): one 1024 x 1024 page -> 273 embeddings.
/// - Decoder bundle (8-bit, GPU): per prefix-capacity bucket P a `prefill_P` (128 tokens per call) and a `decode_P`
///   (1 token) function sharing weights. The K/V cache is Core ML state: P prefix slots + a 128-slot ring for
///   generated tokens (upstream's R-SWA keeps the whole prefix plus the last 128 generated tokens).
/// - Host: embeddings, RoPE, masks, ring slots and upstream's 35-gram no-repeat ban.
///
/// An actor: Core ML's synchronous `prediction` is not thread-safe.
@available(macOS 15.0, iOS 18.0, *)
public actor UnlimitedOcrManager {
    private struct Bucket {
        let capacity: Int
        let prefill: MLModel
        let decode: MLModel
    }

    let config: UnlimitedOcrConfig
    private let directory: URL
    private let computeUnits: MLComputeUnits
    private let encoder: MLModel
    private let embeddings: Data  // fp16 [vocab, hidden]
    private let tokenDecoder: UnlimitedOcrTokenDecoder
    private var buckets: [Int: Bucket] = [:]

    /// Largest document, in pages, the bundle's biggest capacity bucket holds.
    public nonisolated let maxPages: Int

    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws
        -> UnlimitedOcrManager
    {
        let config = try JSONDecoder().decode(
            UnlimitedOcrConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let embeddings = try Data(
            contentsOf: directory.appendingPathComponent("embeddings.f16"), options: .alwaysMapped)
        guard embeddings.count == config.vocabSize * config.hiddenSize * 2 else {
            throw UnlimitedOcrError.invalidAsset("embeddings.f16 size")
        }
        let tokenDecoder = try UnlimitedOcrTokenDecoder(
            tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"),
            skipping: [config.bosId, config.eosId])
        let encoder = try await compiledModel(
            directory: directory, package: config.encoder, functionName: nil, computeUnits: computeUnits)
        return UnlimitedOcrManager(
            config: config, directory: directory, computeUnits: computeUnits, encoder: encoder,
            embeddings: embeddings, tokenDecoder: tokenDecoder)
    }

    private init(
        config: UnlimitedOcrConfig, directory: URL, computeUnits: MLComputeUnits, encoder: MLModel, embeddings: Data,
        tokenDecoder: UnlimitedOcrTokenDecoder
    ) {
        self.config = config
        self.directory = directory
        self.computeUnits = computeUnits
        self.encoder = encoder
        self.embeddings = embeddings
        self.tokenDecoder = tokenDecoder
        let largest = config.capacities.max() ?? 0
        maxPages = max(0, (largest - 1 - (config.promptSuffixIds["multi"]?.count ?? 0)) / config.pageTokens)
    }

    /// Compiles a package once into `compiled/` and loads it (optionally one function of a multifunction package).
    private static func compiledModel(
        directory: URL, package: String, functionName: String?, computeUnits: MLComputeUnits
    ) async throws -> MLModel {
        let source = directory.appendingPathComponent(package)
        let compiledDir = directory.appendingPathComponent("compiled")
        try FileManager.default.createDirectory(at: compiledDir, withIntermediateDirectories: true)
        let target = compiledDir.appendingPathComponent(source.deletingPathExtension().lastPathComponent + ".mlmodelc")
        if !FileManager.default.fileExists(atPath: target.path) {
            let temporary = try await MLModel.compileModel(at: source)
            try FileManager.default.moveItem(at: temporary, to: target)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.functionName = functionName
        return try await MLModel.load(contentsOf: target, configuration: configuration)
    }

    /// Smallest capacity bucket that holds a prefix of `tokens` tokens (prefill runs in whole chunks).
    func capacity(forPrefix tokens: Int) -> Int? {
        let padded = (tokens + config.chunk - 1) / config.chunk * config.chunk
        return config.capacities.sorted().first { $0 >= padded }
    }

    /// Loads the decoder functions for the buckets these page counts need (default: all) and runs one prediction
    /// through each model, so the first real read does not pay Core ML's lazy GPU setup.
    public func warm(pages: [Int]? = nil) async throws {
        let counts = pages ?? Array(1...max(1, maxPages))
        let blank = UnlimitedOcrPage(
            pixels: [Float](
                repeating: 0, count: 3 * UnlimitedOcrPagePreprocessor.size * UnlimitedOcrPagePreprocessor.size))
        _ = try encode(blank)
        var warmed = Set<Int>()
        for count in counts {
            guard let capacity = capacity(forPrefix: prefixLength(pages: count, prompt: "multi")),
                warmed.insert(capacity).inserted
            else { continue }
            let bucket = try await bucket(capacity)
            let slots = capacity + config.ring
            for (model, c) in [(bucket.prefill, config.chunk), (bucket.decode, 1)] {
                let x = try MLMultiArray(shape: [1, c as NSNumber, config.hiddenSize as NSNumber], dataType: .float16)
                let mask = try MLMultiArray(shape: [1, 1, c as NSNumber, (slots + c) as NSNumber], dataType: .float16)
                let writeIdx = try MLMultiArray(shape: [c as NSNumber], dataType: .int32)
                let outIndex = try MLMultiArray(shape: [1], dataType: .int32)
                for i in 0..<c { writeIdx.dataPointer.assumingMemoryBound(to: Int32.self)[i] = Int32(i) }
                outIndex.dataPointer.assumingMemoryBound(to: Int32.self)[0] = 0
                _ = try step(
                    model, state: model.makeState(), x: x, positions: 0..<c, mask: mask, writeIdx: writeIdx,
                    outIndex: outIndex)
            }
        }
    }

    private func bucket(_ capacity: Int) async throws -> Bucket {
        if let bucket = buckets[capacity] { return bucket }
        let prefill = try await Self.compiledModel(
            directory: directory, package: config.decoder, functionName: "prefill_P\(capacity)",
            computeUnits: computeUnits)
        let decode = try await Self.compiledModel(
            directory: directory, package: config.decoder, functionName: "decode_P\(capacity)",
            computeUnits: computeUnits)
        let bucket = Bucket(capacity: capacity, prefill: prefill, decode: decode)
        buckets[capacity] = bucket
        return bucket
    }

    func prefixLength(pages: Int, prompt: String) -> Int {
        1 + pages * config.pageTokens + (config.promptSuffixIds[prompt]?.count ?? 0)
    }

    /// Reads `pages` (from `UnlimitedOcrPagePreprocessor`) as one document. With one page the "single" prompt
    /// (`document parsing.`) can be used; the default multi-page prompt marks each page with `<PAGE>`.
    public nonisolated func read(
        pages: [UnlimitedOcrPage], prompt: String = "multi", maxNewTokens: Int? = nil
    ) -> AsyncThrowingStream<UnlimitedOcrEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(
                        pages: pages, prompt: prompt, maxNewTokens: maxNewTokens,
                        emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Generation

    private func run(
        pages: [UnlimitedOcrPage], prompt: String, maxNewTokens: Int?, emit: @Sendable (UnlimitedOcrEvent) -> Void
    ) async throws {
        guard !pages.isEmpty else { throw UnlimitedOcrError.invalidInput("no pages") }
        guard let suffix = config.promptSuffixIds[prompt] else {
            throw UnlimitedOcrError.invalidInput("unknown prompt \(prompt)")
        }
        let hidden = config.hiddenSize
        let n = prefixLength(pages: pages.count, prompt: prompt)
        guard let capacity = capacity(forPrefix: n) else {
            throw UnlimitedOcrError.invalidInput(
                "\(pages.count) pages need \(n) prefix slots; bundle holds \(maxPages) pages")
        }
        let bucket = try await bucket(capacity)

        // Prefix: BOS, the page embeddings, the prompt text.
        var promptIDs = [config.bosId]
        var prefix = [Float16](repeating: 0, count: n * hidden)
        copyEmbedding(config.bosId, into: &prefix, row: 0)
        let encodeStart = Date()
        for (index, page) in pages.enumerated() {
            let pageStart = Date()
            let embeds = try encode(page)
            try copyRows(embeds, rows: config.pageTokens, into: &prefix, startRow: 1 + index * config.pageTokens)
            promptIDs += Array(repeating: config.imageTokenId, count: config.pageTokens)
            emit(.pageEncoded(index: index, milliseconds: Date().timeIntervalSince(pageStart) * 1000))
            try Task.checkCancellation()
        }
        let encodeMs = Date().timeIntervalSince(encodeStart) * 1000
        for (i, id) in suffix.enumerated() { copyEmbedding(id, into: &prefix, row: n - suffix.count + i) }
        promptIDs += suffix

        // Prefill in chunks, then hand the cache to the decode function's state.
        let slots = capacity + config.ring
        let chunk = config.chunk
        let prefillStart = Date()
        let prefillState = bucket.prefill.makeState()
        var top: (values: [Float], ids: [Int]) = ([], [])
        let x = try MLMultiArray(shape: [1, chunk as NSNumber, hidden as NSNumber], dataType: .float16)
        let mask = try MLMultiArray(shape: [1, 1, chunk as NSNumber, (slots + chunk) as NSNumber], dataType: .float16)
        let writeIdx = try MLMultiArray(shape: [chunk as NSNumber], dataType: .int32)
        let outIndex = try MLMultiArray(shape: [1], dataType: .int32)
        for start in stride(from: 0, to: n, by: chunk) {
            let rows = min(chunk, n - start)
            let xp = x.dataPointer.assumingMemoryBound(to: Float16.self)
            prefix.withUnsafeBufferPointer { src in
                xp.update(from: src.baseAddress! + start * hidden, count: rows * hidden)
            }
            (xp + rows * hidden).update(repeating: 0, count: (chunk - rows) * hidden)
            let mp = mask.dataPointer.assumingMemoryBound(to: Float16.self)
            let width = slots + chunk
            for i in 0..<chunk {
                let row = mp + i * width
                row.update(repeating: -1e4, count: width)
                row.update(repeating: 0, count: start)  // earlier chunks
                (row + slots).update(repeating: 0, count: i + 1)  // causal within the chunk
            }
            let wp = writeIdx.dataPointer.assumingMemoryBound(to: Int32.self)
            for i in 0..<chunk { wp[i] = Int32(start + i) }
            outIndex.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(rows - 1)
            top = try step(
                bucket.prefill, state: prefillState, x: x, positions: start..<(start + chunk), mask: mask,
                writeIdx: writeIdx, outIndex: outIndex)
            try Task.checkCancellation()
        }
        let decodeState = bucket.decode.makeState()
        try copyState(from: prefillState, to: decodeState)
        let prefillMs = Date().timeIntervalSince(prefillStart) * 1000
        emit(.prefilled(tokens: n, capacity: capacity, milliseconds: prefillMs))

        // Decode: token t goes to ring slot P + t % W; the mask shows the prefix, the previous W - 1 tokens and itself.
        var tokens = tokenDecoder
        let x1 = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float16)
        let mask1 = try MLMultiArray(shape: [1, 1, 1, (slots + 1) as NSNumber], dataType: .float16)
        let m1 = mask1.dataPointer.assumingMemoryBound(to: Float16.self)
        m1.update(repeating: -1e4, count: slots + 1)
        m1.update(repeating: 0, count: n)
        m1[slots] = 0
        let write1 = try MLMultiArray(shape: [1], dataType: .int32)
        let out0 = try MLMultiArray(shape: [1], dataType: .int32)
        out0.dataPointer.assumingMemoryBound(to: Int32.self)[0] = 0
        var sequence = promptIDs
        var limit = config.maxLength - n
        if let maxNewTokens { limit = min(limit, maxNewTokens) }
        var produced = 0
        var stoppedAtEOS = false
        let decodeStart = Date()
        for t in 0..<limit {
            let banned = bannedTokens(sequence)
            guard let token = zip(top.values, top.ids).first(where: { !banned.contains($0.1) })?.1 else {
                throw UnlimitedOcrError.prediction("every top-k candidate is banned at step \(t)")
            }
            sequence.append(token)
            produced += 1
            let text = tokens.append(token)
            if !text.isEmpty { emit(.text(text)) }
            if token == config.eosId {
                stoppedAtEOS = true
                break
            }
            if t == limit - 1 { break }
            try Task.checkCancellation()
            let slot = capacity + t % config.ring
            if t > 0 { m1[capacity + (t - 1) % config.ring] = 0 }
            m1[slot] = -1e4
            write1.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(slot)
            let xp = x1.dataPointer.assumingMemoryBound(to: Float16.self)
            embeddings.withUnsafeBytes { raw in
                let table = raw.bindMemory(to: Float16.self)
                xp.update(from: table.baseAddress! + token * hidden, count: hidden)
            }
            top = try step(
                bucket.decode, state: decodeState, x: x1, positions: (n + t)..<(n + t + 1), mask: mask1,
                writeIdx: write1, outIndex: out0)
        }
        emit(
            .finished(
                UnlimitedOcrStats(
                    pages: pages.count, promptTokens: n, outputTokens: produced, encodeMilliseconds: encodeMs,
                    prefillMilliseconds: prefillMs, decodeMilliseconds: Date().timeIntervalSince(decodeStart) * 1000,
                    stoppedAtEOS: stoppedAtEOS)))
    }

    /// Page encoder call (synchronous: the async overload would send non-Sendable Core ML values off the actor).
    private func encode(_ page: UnlimitedOcrPage) throws -> MLMultiArray {
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
        return embeds
    }

    /// One decoder call; returns the merged top-k over both vocab halves, best first.
    private func step(
        _ model: MLModel, state: MLState, x: MLMultiArray, positions: Range<Int>, mask: MLMultiArray,
        writeIdx: MLMultiArray, outIndex: MLMultiArray
    ) throws -> (values: [Float], ids: [Int]) {
        let (cos, sin) = try rope(positions)
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: x), "cos": MLFeatureValue(multiArray: cos),
            "sin": MLFeatureValue(multiArray: sin), "mask": MLFeatureValue(multiArray: mask),
            "write_idx": MLFeatureValue(multiArray: writeIdx), "out_index": MLFeatureValue(multiArray: outIndex),
        ])
        let output = try model.prediction(from: inputs, using: state)
        guard let values = output.featureValue(for: "top_values")?.multiArrayValue,
            let ids = output.featureValue(for: "top_ids")?.multiArrayValue
        else { throw UnlimitedOcrError.prediction("decoder returned no top-k") }
        var merged: [(Float, Int)] = []
        let k = values.shape[1].intValue
        for half in 0..<2 {
            for j in 0..<k {
                let value = values[[half as NSNumber, j as NSNumber]].floatValue
                let id = ids[[half as NSNumber, j as NSNumber]].intValue + half * config.vocabHalf
                merged.append((value, id))
            }
        }
        merged.sort { $0.0 > $1.0 }
        return (merged.map(\.0), merged.map(\.1))
    }

    private func rope(_ positions: Range<Int>) throws -> (MLMultiArray, MLMultiArray) {
        let dim = config.headDim
        let half = dim / 2
        let count = positions.count as NSNumber
        let cos = try MLMultiArray(shape: [count, dim as NSNumber], dataType: .float16)
        let sin = try MLMultiArray(shape: [count, dim as NSNumber], dataType: .float16)
        let c = cos.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sin.dataPointer.assumingMemoryBound(to: Float16.self)
        for (row, position) in positions.enumerated() {
            for i in 0..<half {
                let angle = Double(position) / pow(config.ropeTheta, Double(2 * i) / Double(dim))
                c[row * dim + i] = Float16(Foundation.cos(angle))
                c[row * dim + half + i] = Float16(Foundation.cos(angle))
                s[row * dim + i] = Float16(Foundation.sin(angle))
                s[row * dim + half + i] = Float16(Foundation.sin(angle))
            }
        }
        return (cos, sin)
    }

    /// Upstream `SlidingWindowNoRepeatNgramProcessor`: ban the token that would repeat a 35-gram seen in the last
    /// 1,024 tokens (prompt included).
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

    private func copyEmbedding(_ id: Int, into prefix: inout [Float16], row: Int) {
        let hidden = config.hiddenSize
        embeddings.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self)
            prefix.withUnsafeMutableBufferPointer { dst in
                (dst.baseAddress! + row * hidden).update(from: table.baseAddress! + id * hidden, count: hidden)
            }
        }
    }

    private func copyRows(_ array: MLMultiArray, rows: Int, into prefix: inout [Float16], startRow: Int) throws {
        let hidden = config.hiddenSize
        guard array.shape.count == 2, array.shape[0].intValue == rows, array.shape[1].intValue == hidden else {
            throw UnlimitedOcrError.prediction("page_embeds shape \(array.shape)")
        }
        let rowStride = array.strides[0].intValue
        let colStride = array.strides[1].intValue
        prefix.withUnsafeMutableBufferPointer { dst in
            let base = dst.baseAddress! + startRow * hidden
            switch array.dataType {
            case .float32:
                let src = array.dataPointer.assumingMemoryBound(to: Float.self)
                for r in 0..<rows {
                    for c in 0..<hidden { base[r * hidden + c] = Float16(src[r * rowStride + c * colStride]) }
                }
            case .float16:
                let src = array.dataPointer.assumingMemoryBound(to: Float16.self)
                for r in 0..<rows { for c in 0..<hidden { base[r * hidden + c] = src[r * rowStride + c * colStride] } }
            default:
                break
            }
        }
    }

    private func copyState(from source: MLState, to target: MLState) throws {
        for name in ["k_cache", "v_cache"] {
            var mismatch: String?
            source.withMultiArray(for: name) { src in
                target.withMultiArray(for: name) { dst in
                    guard src.shape == dst.shape, src.strides == dst.strides, src.dataType == dst.dataType else {
                        mismatch = "\(name): \(src.shape) vs \(dst.shape)"
                        return
                    }
                    let bytes = src.count * (src.dataType == .float16 ? 2 : 4)
                    dst.dataPointer.copyMemory(from: src.dataPointer, byteCount: bytes)
                }
            }
            if let mismatch { throw UnlimitedOcrError.prediction("state copy \(mismatch)") }
        }
    }
}
