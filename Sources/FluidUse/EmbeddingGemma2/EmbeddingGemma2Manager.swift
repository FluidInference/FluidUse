@preconcurrency import CoreML
import Foundation

/// EmbeddingGemma 2 text embeddings on Core ML. The package holds fixed-shape functions sharing one set of weights,
/// so each stays on the Neural Engine: `embed_<S>` for one text of up to S tokens, and `pack_256`, which runs up to
/// eight short texts in one 256-token sequence (block-diagonal attention, positions restarting per text). Short
/// calls are bound by streaming the weights, so packing more than doubles throughput. The 262k-row token table is
/// read from a memory-mapped bf16 file instead of living in the model. Output: L2-normalized 768-d vectors.
public final class EmbeddingGemma2Manager: Sendable {
    public static let lengths = [32, 48, 64, 128, 256, 512]
    public static let hiddenSize = 512
    public static let dimension = 768
    public static let packLength = 256
    public static let packSlots = 8

    public let tokenizer: EmbeddingGemma2Tokenizer
    private let models: [Int: MLModel]
    private let packModel: MLModel?
    private let table: Data
    private let embedScale = Float(hiddenSize).squareRoot()

    init(tokenizer: EmbeddingGemma2Tokenizer, models: [Int: MLModel], packModel: MLModel?, table: Data) {
        self.tokenizer = tokenizer
        self.models = models
        self.packModel = packModel
        self.table = table
    }

    /// Loads from `EMBEDDINGGEMMA2_MODEL_DIR` when set, otherwise from the published files (downloaded once into
    /// `~/Library/Application Support/FluidUse/Models/embeddinggemma-2-coreml`, checksum-verified).
    public static func loadDefault(
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine, progress: EmbeddingGemma2ModelStore.Progress? = nil
    ) async throws -> EmbeddingGemma2Manager {
        // Before any download: the package needs multifunction support.
        guard #available(macOS 15, iOS 18, *) else {
            throw EmbeddingGemma2Error.unsupported("multifunction Core ML models need macOS 15 / iOS 18")
        }
        if let path = ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_MODEL_DIR"], !path.isEmpty {
            return try await load(from: URL(fileURLWithPath: path), computeUnits: computeUnits)
        }
        let directory = try await EmbeddingGemma2ModelStore.ensure(progress: progress)
        return try await load(from: directory, computeUnits: computeUnits)
    }

    /// Loads `EmbeddingGemma2Text.mlmodelc` (or `.mlpackage`), `embeddings.bf16`, and `tokenizer.json` from
    /// `directory`.
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> EmbeddingGemma2Manager {
        guard #available(macOS 15, iOS 18, *) else {
            throw EmbeddingGemma2Error.unsupported("multifunction Core ML models need macOS 15 / iOS 18")
        }
        let manager = FileManager.default
        let compiled = directory.appendingPathComponent("EmbeddingGemma2Text.mlmodelc")
        let package = directory.appendingPathComponent("EmbeddingGemma2Text.mlpackage")
        let modelURL: URL
        if manager.fileExists(atPath: compiled.path) {
            modelURL = compiled
        } else if manager.fileExists(atPath: package.path) {
            modelURL = try await MLModel.compileModel(at: package)
        } else {
            throw EmbeddingGemma2Error.invalidAsset("Missing EmbeddingGemma2Text.mlmodelc in \(directory.path)")
        }
        let tableURL = directory.appendingPathComponent("embeddings.bf16")
        guard let table = try? Data(contentsOf: tableURL, options: .alwaysMapped),
            table.count % (hiddenSize * 2) == 0
        else {
            throw EmbeddingGemma2Error.invalidAsset("Missing or malformed embeddings.bf16 in \(directory.path)")
        }
        let tokenizer = try EmbeddingGemma2Tokenizer(
            tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        func function(_ name: String) async throws -> MLModel {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            configuration.functionName = name
            return try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        }
        var models: [Int: MLModel] = [:]
        for length in lengths { models[length] = try await function("embed_\(length)") }
        return EmbeddingGemma2Manager(
            tokenizer: tokenizer, models: models, packModel: try await function("pack_256"), table: table)
    }

    /// Embeddings for many texts, in order. Texts that fit are packed eight to a 256-token call; up to
    /// `maxInFlight` calls run concurrently (Core ML's async prediction is thread-safe).
    public static let maxInFlight = 8

    public func embed(_ texts: [String], prompt: EmbeddingGemma2Prompt = .none) async throws -> [[Float]] {
        let tokenized = texts.map { tokenizer.encode(prompt.apply(to: $0), maxLength: Self.lengths.last!) }
        var bins: [[Int]] = []
        var singles: [Int] = []
        var current: [Int] = []
        var used = 0
        for (index, ids) in tokenized.enumerated() {
            guard packModel != nil, ids.count <= Self.packLength else {
                singles.append(index)
                continue
            }
            if used + ids.count > Self.packLength || current.count == Self.packSlots {
                bins.append(current)
                current = []
                used = 0
            }
            current.append(index)
            used += ids.count
        }
        if !current.isEmpty { bins.append(current) }
        // Each job is one model call: a bin of packed texts or one long text.
        let jobs: [[Int]] = bins + singles.map { [$0] }
        let packedJobs = bins.count
        return try await withThrowingTaskGroup(of: [(Int, [Float])].self) { group in
            var result = [[Float]](repeating: [], count: texts.count)
            var next = 0
            func addJob() {
                guard next < jobs.count else { return }
                let job = jobs[next]
                let packed = next < packedJobs
                next += 1
                group.addTask {
                    if packed {
                        return zip(job, try await self.embedPacked(job.map { tokenized[$0] })).map { ($0, $1) }
                    }
                    return [(job[0], try await self.embed(ids: tokenized[job[0]]))]
                }
            }
            for _ in 0..<min(Self.maxInFlight, jobs.count) { addJob() }
            for try await pairs in group {
                for (index, vector) in pairs { result[index] = vector }
                addJob()
            }
            return result
        }
    }

    /// One `pack_256` call for up to eight token sequences that fit in 256 tokens together.
    private func embedPacked(_ sequences: [[Int32]]) async throws -> [[Float]] {
        guard let packModel else { throw EmbeddingGemma2Error.predictionFailed("no packed function") }
        let length = Self.packLength
        let embeds = try MLMultiArray(
            shape: [1, NSNumber(value: length), NSNumber(value: Self.hiddenSize)], dataType: .float16)
        let bias = try MLMultiArray(
            shape: [1, 1, NSNumber(value: length), NSNumber(value: length)], dataType: .float16)
        let positions = try MLMultiArray(shape: [NSNumber(value: length), 1], dataType: .float16)
        let pool = try MLMultiArray(
            shape: [NSNumber(value: Self.packSlots), NSNumber(value: length)], dataType: .float16)
        let embedPointer = embeds.dataPointer.assumingMemoryBound(to: Float16.self)
        let biasPointer = bias.dataPointer.assumingMemoryBound(to: Float16.self)
        let positionPointer = positions.dataPointer.assumingMemoryBound(to: Float16.self)
        let poolPointer = pool.dataPointer.assumingMemoryBound(to: Float16.self)
        biasPointer.update(repeating: -10_000, count: length * length)
        positionPointer.update(repeating: 0, count: length)
        poolPointer.update(repeating: 0, count: Self.packSlots * length)
        var offset = 0
        for (slot, ids) in sequences.enumerated() {
            writeEmbeddings(ids, to: embedPointer + offset * Self.hiddenSize)
            let weight = Float16(1 / Float(ids.count))
            for row in offset..<(offset + ids.count) {
                for column in offset..<(offset + ids.count) { biasPointer[row * length + column] = 0 }
                positionPointer[row] = Float16(row - offset)
                poolPointer[slot * length + row] = weight
            }
            offset += ids.count
        }
        // Padding rows attend to themselves so their softmax stays finite; the pool ignores them.
        writeEmbeddings(
            [Int32](repeating: tokenizer.padId, count: length - offset), to: embedPointer + offset * Self.hiddenSize)
        for row in offset..<length { biasPointer[row * length + row] = 0 }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: embeds), "attention_bias": MLFeatureValue(multiArray: bias),
            "positions": MLFeatureValue(multiArray: positions), "pool": MLFeatureValue(multiArray: pool),
        ])
        let output = try await packModel.prediction(from: input)
        guard let matrix = output.featureValue(for: "embedding")?.multiArrayValue,
            matrix.count == Self.packSlots * Self.dimension
        else { throw EmbeddingGemma2Error.predictionFailed("missing packed embedding output") }
        // [8, 768] or [1, 8, 768]: the row stride is the second-to-last one.
        let rowStride = matrix.strides[matrix.strides.count - 2].intValue
        return (0..<sequences.count).map { slot in
            (0..<Self.dimension).map { column in
                let index = slot * rowStride + column
                return matrix.dataType == .float16
                    ? Float(matrix.dataPointer.assumingMemoryBound(to: Float16.self)[index])
                    : matrix.dataPointer.assumingMemoryBound(to: Float.self)[index]
            }
        }
    }

    /// Token rows (bf16 → × √512 → fp16) for `ids`, written contiguously at `destination`.
    private func writeEmbeddings(_ ids: [Int32], to destination: UnsafeMutablePointer<Float16>) {
        let rows = table.count / (Self.hiddenSize * 2)
        table.withUnsafeBytes { raw in
            let bf16 = raw.bindMemory(to: UInt16.self)
            for (position, id) in ids.enumerated() {
                let row = min(max(Int(id), 0), rows - 1) * Self.hiddenSize
                let target = destination + position * Self.hiddenSize
                for column in 0..<Self.hiddenSize {
                    target[column] = Float16(Float(bitPattern: UInt32(bf16[row + column]) << 16) * embedScale)
                }
            }
        }
    }

    /// L2-normalized embedding of `text` with the task prefix `prompt`; text past 512 tokens is dropped.
    public func embed(_ text: String, prompt: EmbeddingGemma2Prompt = .none) async throws -> [Float] {
        try await embed(ids: tokenizer.encode(prompt.apply(to: text), maxLength: Self.lengths.last!))
    }

    private func embed(ids: [Int32]) async throws -> [Float] {
        let padded = ids + [Int32](repeating: tokenizer.padId, count: max(Self.lengths.last!, ids.count) - ids.count)
        let length = Self.lengths.first { $0 >= ids.count } ?? Self.lengths.last!
        return try await run(length: length, count: ids.count) { destination in
            writeEmbeddings(Array(padded.prefix(length)), to: destination)
        }
    }

    /// Embedding of one audio window: `<bos> <|audio> tokens <audio|> <eos>` through the text model, where `tokens`
    /// are the first `count` rows of an `EmbeddingGemma2Audio` output ([1, 250, 512], fp16 or fp32).
    public func embed(audioTokens tokens: MLMultiArray, count: Int) async throws -> [Float] {
        let total = count + 4
        guard let length = Self.lengths.first(where: { $0 >= total }) else {
            throw EmbeddingGemma2Error.predictionFailed("\(count) audio tokens do not fit")
        }
        let tokenStride = tokens.strides[tokens.strides.count - 2].intValue
        return try await run(length: length, count: total) { destination in
            writeEmbeddings([bosId, Self.audioStartId], to: destination)
            let target = destination + 2 * Self.hiddenSize
            for row in 0..<count {
                for column in 0..<Self.hiddenSize {
                    let index = row * tokenStride + column
                    target[row * Self.hiddenSize + column] =
                        tokens.dataType == .float16
                        ? tokens.dataPointer.assumingMemoryBound(to: Float16.self)[index]
                        : Float16(tokens.dataPointer.assumingMemoryBound(to: Float.self)[index])
                }
            }
            writeEmbeddings(
                [Self.audioEndId, eosId] + [Int32](repeating: tokenizer.padId, count: length - total),
                to: destination + (2 + count) * Self.hiddenSize)
        }
    }

    static let audioStartId: Int32 = 256_000  // <|audio>
    static let audioEndId: Int32 = 258_883  // <audio|>
    private var bosId: Int32 { tokenizer.bosId }
    private var eosId: Int32 { tokenizer.eosId }

    /// One `embed_<length>` call: `fill` writes `length` rows of fp16 inputs; the first `count` are attended.
    private func run(
        length: Int, count: Int, fill: (UnsafeMutablePointer<Float16>) throws -> Void
    ) async throws -> [Float] {
        guard let model = models[length] else { throw EmbeddingGemma2Error.predictionFailed("no model for \(length)") }
        let embeds = try MLMultiArray(
            shape: [1, NSNumber(value: length), NSNumber(value: Self.hiddenSize)], dataType: .float16)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float16)
        let maskPointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        try fill(embeds.dataPointer.assumingMemoryBound(to: Float16.self))
        for position in 0..<length { maskPointer[position] = position < count ? 1 : 0 }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: embeds), "attention_mask": MLFeatureValue(multiArray: mask),
        ])
        let output = try await model.prediction(from: input)
        guard let vector = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw EmbeddingGemma2Error.predictionFailed("missing embedding output")
        }
        guard vector.count == Self.dimension else {
            throw EmbeddingGemma2Error.predictionFailed("embedding has \(vector.count) values")
        }
        if vector.dataType == .float16 {
            let pointer = vector.dataPointer.assumingMemoryBound(to: Float16.self)
            return (0..<Self.dimension).map { Float(pointer[$0]) }
        }
        let pointer = vector.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: Self.dimension))
    }
}
