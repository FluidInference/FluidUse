@preconcurrency import CoreML
import Foundation

/// Kev in one call per request: the state and every question run through a single `fused_S*_P*` function, the
/// questions packed end to end (each restarting from the state) instead of one row per question. Questions longer
/// than a lane (128 tokens) or with more than 16 options fall back to `KevManager`'s rows.
@available(macOS 15.0, iOS 18.0, *)
public final class KevFastManager: Sendable {
    struct Shape: Sendable {
        let hidden: Int
        let rotary: Int
        let theta: Double
        let convTail: Int
    }

    struct Branch {
        let ids: [Int]
        let decide: Int
        let options: [Int]
    }

    static let lane = 128
    static let readouts = 16
    static let maxOptions = 16

    private let rows: KevManager
    private let compiled: URL
    private let computeUnits: MLComputeUnits
    private let shape: Shape
    private let embeddings: Data
    private let padID: Int
    /// State bucket -> packed buckets, from the package's function names.
    private let buckets: [Int: [Int]]
    private let stateBuckets: [Int]
    private let largestPacked: Int
    private let functions = FunctionCache()

    actor FunctionCache {
        private var models: [String: MLModel] = [:]

        func model(_ name: String, at url: URL, units: MLComputeUnits) async throws -> MLModel {
            if let model = models[name] { return model }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units
            configuration.functionName = name
            let model = try await MLModel.load(contentsOf: url, configuration: configuration)
            models[name] = model
            return model
        }
    }

    /// `directory` holds the row buckets `KevManager` loads plus `fused/` with `KevFused.mlmodelc` (or `.mlpackage`)
    /// and its `config.json`.
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> KevFastManager {
        let rows = try await KevManager.load(from: directory, computeUnits: computeUnits)
        let fused = directory.appendingPathComponent("fused")
        var compiled = fused.appendingPathComponent("KevFused.mlmodelc")
        if !FileManager.default.fileExists(atPath: compiled.path) {
            compiled = try await MLModel.compileModel(at: fused.appendingPathComponent("KevFused.mlpackage"))
        }
        guard
            let config = try JSONSerialization.jsonObject(
                with: Data(contentsOf: fused.appendingPathComponent("config.json"))) as? [String: Any],
            let kernel = config["conv_kernel"] as? Int, let hidden = config["hidden_size"] as? Int,
            let rotary = config["rotary_dim"] as? Int, let theta = (config["rope_theta"] as? NSNumber)?.doubleValue,
            let pad = config["pad_id"] as? Int, let vocab = config["vocab_size"] as? Int
        else { throw KevError.invalidAsset("fused/config.json is missing fields") }
        // the embedding table ships once, in one of the row bucket folders
        guard
            let table = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .map({ $0.resolvingSymlinksInPath().appendingPathComponent("embeddings.f16") })
                .first(where: { FileManager.default.fileExists(atPath: $0.path) })
        else { throw KevError.invalidAsset("no row bucket folder has embeddings.f16") }
        let embeddings = try Data(contentsOf: table, options: .alwaysMapped)
        guard embeddings.count == vocab * hidden * 2 else { throw KevError.invalidAsset("embeddings.f16 size") }
        var buckets: [Int: [Int]] = [:]
        let suffix = "_B\(readouts)_K\(maxOptions)"
        for name in try await MLModelAsset(url: compiled).functionNames
        where name.hasPrefix("fused_S") && name.hasSuffix(suffix) {
            let numbers = name.dropLast(suffix.count).split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            guard numbers.count == 2 else { continue }
            buckets[numbers[0], default: []].append(numbers[1])
        }
        guard !buckets.isEmpty else {
            throw KevError.invalidAsset("no fused_S*_P* functions in \(compiled.lastPathComponent)")
        }
        return KevFastManager(
            rows: rows, compiled: compiled, computeUnits: computeUnits,
            shape: Shape(hidden: hidden, rotary: rotary, theta: theta, convTail: kernel - 1), embeddings: embeddings,
            padID: pad, buckets: buckets.mapValues { $0.sorted() })
    }

    init(
        rows: KevManager, compiled: URL, computeUnits: MLComputeUnits, shape: Shape, embeddings: Data, padID: Int,
        buckets: [Int: [Int]]
    ) {
        self.rows = rows
        self.compiled = compiled
        self.computeUnits = computeUnits
        self.shape = shape
        self.embeddings = embeddings
        self.padID = padID
        self.buckets = buckets
        self.stateBuckets = buckets.keys.sorted()
        self.largestPacked = buckets.values.compactMap(\.last).min() ?? 0
    }

    /// Loads every function (or those of `stateBuckets`) and runs it once, so no request pays for loading or for the
    /// first GPU dispatch.
    public func warm(stateBuckets: [Int]? = nil) async throws {
        for s in stateBuckets ?? self.stateBuckets {
            for p in buckets[s] ?? [] {
                _ = try await run(
                    stateIDs: [padID], stateLen: s, branches: [Branch(ids: [padID], decide: 0, options: [0])],
                    starts: [0], packedLen: p)
            }
        }
    }

    public func answer(
        state: String, questions: [KevQuestion], maxStateTokens: Int = KevManager.evaluationMaxStateTokens
    ) async throws -> [KevAnswer] {
        let special = rows.special
        let stateIDs = [special.state] + Array(try rows.userTokens(state).prefix(maxStateTokens - 1))
        var branches: [Branch] = []
        for question in questions {
            var ids = [special.question] + (try rows.userTokens(question.instruction))
            var ends: [Int] = []
            for text in question.optionTexts {
                ids += [special.option] + (try rows.userTokens(text)) + [special.closeOption]
                ends.append(ids.count - 1)
            }
            ids.append(special.decide)
            branches.append(Branch(ids: ids, decide: ids.count - 1, options: ends))
        }
        var answers = [KevAnswer?](repeating: nil, count: questions.count)
        let fits = branches.indices.filter {
            branches[$0].ids.count <= Self.lane && branches[$0].options.count <= Self.maxOptions
        }
        guard let stateLen = stateBuckets.first(where: { stateIDs.count <= $0 }) else {
            for index in answers.indices {
                answers[index] = try await rows.answer(stateIDs: stateIDs, question: questions[index])
            }
            return answers.compactMap { $0 }
        }
        for index in answers.indices where !fits.contains(index) {
            answers[index] = try await rows.answer(stateIDs: stateIDs, question: questions[index])
        }
        for group in Self.packGroups(fits.map { branches[$0].ids.count }, packedLen: largestPacked) {
            let indices = group.map { fits[$0] }
            let starts = Self.laneStarts(indices.map { branches[$0].ids.count })
            let extent = zip(starts, indices).map { $0 + branches[$1].ids.count }.max() ?? 1
            guard let packedLen = buckets[stateLen]?.first(where: { extent <= $0 }) else {
                throw KevError.invalidAsset("no packed bucket of \(extent) tokens for state bucket \(stateLen)")
            }
            let probabilities = try await run(
                stateIDs: stateIDs, stateLen: stateLen, branches: indices.map { branches[$0] }, starts: starts,
                packedLen: packedLen)
            for (slot, index) in indices.enumerated() {
                answers[index] = KevAnswer(
                    keys: questions[index].keys, probabilities: probabilities[slot],
                    tokens: stateIDs.count + branches[index].ids.count)
            }
        }
        return answers.compactMap { $0 }
    }

    /// Start offsets of questions packed in order, a question that would cross a lane moved to the next lane.
    static func laneStarts(_ lengths: [Int]) -> [Int] {
        var starts: [Int] = []
        var start = 0
        for n in lengths {
            if start / lane != (start + n - 1) / lane { start = (start / lane + 1) * lane }
            starts.append(start)
            start += n
        }
        return starts
    }

    /// Greedy in-order groups of question indices that fit one call (`packedLen` tokens, readout count).
    static func packGroups(_ lengths: [Int], packedLen: Int) -> [[Int]] {
        var groups: [[Int]] = []
        var current: [Int] = []
        for (index, _) in lengths.enumerated() {
            let candidate = current + [index]
            let starts = laneStarts(candidate.map { lengths[$0] })
            let extent = starts.last! + lengths[index]
            if !current.isEmpty && (extent > packedLen || candidate.count > readouts) {
                groups.append(current)
                current = [index]
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { groups.append(current) }
        return groups
    }

    private func model(state: Int, packed: Int) async throws -> MLModel {
        let name = "fused_S\(state)_P\(packed)_B\(Self.readouts)_K\(Self.maxOptions)"
        do {
            return try await functions.model(name, at: compiled, units: computeUnits)
        } catch {
            throw KevError.invalidAsset("function \(name): \(error.localizedDescription)")
        }
    }

    private func run(
        stateIDs: [Int], stateLen: Int, branches: [Branch], starts: [Int], packedLen: Int
    )
        async throws -> [[Float]]
    {
        let model = try await model(state: stateLen, packed: packedLen)
        let total = stateLen + packedLen
        let n = stateIDs.count
        var tokens = [Int](repeating: padID, count: total)
        var positions = Array(0..<stateLen) + [Int](repeating: n, count: packedLen)
        for i in 0..<n { tokens[i] = stateIDs[i] }
        let valid = try half([stateLen])
        let tail = try half([shape.convTail, stateLen])
        let segment = try half([packedLen, packedLen])
        let lagKeep = try half([shape.convTail, packedLen])
        let lagTail = try half([shape.convTail, packedLen, shape.convTail])
        let decide = try half([Self.readouts, packedLen])
        let options = try half([Self.readouts, Self.maxOptions, packedLen])
        let mask = try half([Self.readouts, Self.maxOptions])
        let validPointer = valid.dataPointer.assumingMemoryBound(to: Float16.self)
        let tailPointer = tail.dataPointer.assumingMemoryBound(to: Float16.self)
        let segmentPointer = segment.dataPointer.assumingMemoryBound(to: Float16.self)
        let keepPointer = lagKeep.dataPointer.assumingMemoryBound(to: Float16.self)
        let lagTailPointer = lagTail.dataPointer.assumingMemoryBound(to: Float16.self)
        let decidePointer = decide.dataPointer.assumingMemoryBound(to: Float16.self)
        let optionPointer = options.dataPointer.assumingMemoryBound(to: Float16.self)
        let maskPointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<n { validPointer[i] = 1 }
        for j in 0..<shape.convTail where n - shape.convTail + j >= 0 {
            tailPointer[j * stateLen + n - shape.convTail + j] = 1
        }
        for i in 0..<packedLen { segmentPointer[i * packedLen + i] = 1 }
        let lags = shape.convTail
        for (b, (branch, start)) in zip(branches, starts).enumerated() {
            let count = branch.ids.count
            for p in 0..<count {
                tokens[stateLen + start + p] = branch.ids[p]
                positions[stateLen + start + p] = n + p
                for j in 0...p { segmentPointer[(start + p) * packedLen + start + j] = 1 }
                for s in 1...lags {
                    if p >= s {
                        keepPointer[(s - 1) * packedLen + start + p] = 1
                    } else {
                        lagTailPointer[((s - 1) * packedLen + start + p) * lags + lags + p - s] = 1
                    }
                }
            }
            decidePointer[b * packedLen + start + branch.decide] = 1
            for (slot, index) in branch.options.enumerated() {
                optionPointer[(b * Self.maxOptions + slot) * packedLen + start + index] = 1
                maskPointer[b * Self.maxOptions + slot] = 1
            }
        }
        let (cos, sin) = try rope(positions)
        let output = try await model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "hidden": try embed(tokens), "cos": cos, "sin": sin, "valid": valid, "tail_onehot": tail,
                "segment": segment, "lag_keep": lagKeep, "lag_tail": lagTail, "decide_onehot": decide,
                "option_onehot": options, "option_mask": mask,
            ]))
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw KevError.invalidOutput("fused logits")
        }
        let strides = logits.strides.map(\.intValue)
        let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
        return branches.enumerated().map { b, branch in
            var values = (0..<branch.options.count).map { pointer[b * strides[0] + $0 * strides[1]] }
            let peak = values.max() ?? 0
            values = values.map { exp($0 - peak) }
            let sum = values.reduce(0, +)
            return values.map { $0 / sum }
        }
    }

    /// fp16 embedding rows copied straight from the table.
    private func embed(_ tokens: [Int]) throws -> MLMultiArray {
        let array = try half([1, tokens.count, shape.hidden])
        let destination = array.dataPointer.assumingMemoryBound(to: Float16.self)
        embeddings.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self).baseAddress!
            for (position, token) in tokens.enumerated() {
                (destination + position * shape.hidden).update(from: table + token * shape.hidden, count: shape.hidden)
            }
        }
        return array
    }

    private func rope(_ positions: [Int]) throws -> (MLMultiArray, MLMultiArray) {
        let cos = try half([positions.count, shape.rotary])
        let sin = try half([positions.count, shape.rotary])
        let c = cos.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sin.dataPointer.assumingMemoryBound(to: Float16.self)
        let halfDim = shape.rotary / 2
        for (row, position) in positions.enumerated() {
            for i in 0..<halfDim {
                let angle = Double(position) / pow(shape.theta, Double(2 * i) / Double(shape.rotary))
                let cv = Float16(Foundation.cos(angle))
                let sv = Float16(Foundation.sin(angle))
                c[row * shape.rotary + i] = cv
                c[row * shape.rotary + i + halfDim] = cv
                s[row * shape.rotary + i] = sv
                s[row * shape.rotary + i + halfDim] = sv
            }
        }
        return (cos, sin)
    }

    private func half(_ dims: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: .float16)
        array.dataPointer.initializeMemory(as: Float16.self, repeating: 0, count: array.count)
        return array
    }
}
