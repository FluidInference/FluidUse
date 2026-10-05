@preconcurrency import CoreML
import Foundation

/// Decision 2.0 (vLLM Semantic Router: Kai 0.6B on Qwen3, Eos 0.8B on Qwen3.5) on Core ML: every question of a
/// request in one call. The questions' rows share a token prefix (the context); it runs once, and each question's
/// suffix continues from it exactly as in its own upstream row. Long requests are split into chunks that each repeat
/// the prefix, on whichever function needs the fewest milliseconds.
///
/// `directory` is a FluidInference/decision-2.0-*-coreml snapshot (`Decision2ModelStore.ensure`).
@available(macOS 15.0, iOS 18.0, *)
public final class Decision2Manager: Sendable {
    static let maxOptions = 255
    static let maskValue: Float16 = -1e4
    static let chunk = 64
    static let lags = 3

    enum Backbone: Sendable { case qwen3, qwen35 }

    struct Function: Sendable {
        let name: String
        /// Qwen3: L = packed tokens, S = 0. Qwen3.5: S = prefix tokens, L = packed question tokens.
        let s: Int
        let l: Int
        let n: Int
        let costMs: Double
    }

    struct Row {
        let ids: [Int]
        let candidates: [Int]
        let query: Int
    }

    public let modelName: String
    private let backbone: Backbone
    private let compiled: URL
    private let computeUnits: MLComputeUnits
    private let tokenizer: QwenBPETokenizer
    private let padID: Int
    private let rotaryDim: Int
    private let ropeTheta: Double
    private let scoreBias: [Int: [Double]]
    private let functions: [Function]
    private let cache = FunctionCache()

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

    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> Decision2Manager {
        guard
            let config = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("coreml_config.json"))) as? [String: Any],
            let modelName = config["model_name"] as? String, let package = config["package"] as? String,
            let costs = config["functions"] as? [String: NSNumber], let pad = config["pad_token_id"] as? Int
        else { throw Decision2Error.invalidAsset("coreml_config.json is missing fields") }
        let backbone: Backbone = (config["backbone"] as? String) == "qwen3_5" ? .qwen35 : .qwen3
        var rotary = 0
        var theta = 0.0
        if backbone == .qwen35 {
            guard let rope = config["rope"] as? [String: Any], let dim = rope["rotary_dim"] as? Int,
                let base = (rope["rope_theta"] as? NSNumber)?.doubleValue
            else { throw Decision2Error.invalidAsset("coreml_config.json has no rope") }
            rotary = dim
            theta = base
        }
        var functions: [Function] = []
        for (name, cost) in costs {
            var dims: [Character: Int] = [:]
            for part in name.split(separator: "_") { dims[part.first!] = Int(part.dropFirst()) }
            if backbone == .qwen3, let l = dims["L"], let n = dims["N"] {
                functions.append(Function(name: name, s: 0, l: l, n: n, costMs: cost.doubleValue))
            } else if backbone == .qwen35, let s = dims["S"], let c = dims["C"], let n = dims["N"] {
                functions.append(Function(name: name, s: s, l: c, n: n, costMs: cost.doubleValue))
            } else {
                throw Decision2Error.invalidAsset("unexpected function name \(name)")
            }
        }
        var bias: [Int: [Double]] = [:]
        let biasURL = directory.appendingPathComponent("score_bias.json")
        if FileManager.default.fileExists(atPath: biasURL.path) {
            guard let report = try JSONSerialization.jsonObject(with: Data(contentsOf: biasURL)) as? [String: Any],
                let offsets = report["offsets"] as? [String: [NSNumber]]
            else { throw Decision2Error.invalidAsset("score_bias.json") }
            for (levels, values) in offsets { bias[Int(levels)!] = values.map(\.doubleValue) }
        }
        var compiled = directory.appendingPathComponent(
            (package as NSString).deletingPathExtension + ".mlmodelc")
        if !FileManager.default.fileExists(atPath: compiled.path) {
            compiled = try await KevManager.compiled(directory.appendingPathComponent(package))
        }
        return Decision2Manager(
            modelName: modelName, backbone: backbone, compiled: compiled, computeUnits: computeUnits,
            tokenizer: try QwenBPETokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json")),
            padID: pad, rotaryDim: rotary, ropeTheta: theta, scoreBias: bias,
            functions: functions.sorted { ($0.s, $0.l) < ($1.s, $1.l) })
    }

    init(
        modelName: String, backbone: Backbone, compiled: URL, computeUnits: MLComputeUnits,
        tokenizer: QwenBPETokenizer, padID: Int, rotaryDim: Int, ropeTheta: Double, scoreBias: [Int: [Double]],
        functions: [Function]
    ) {
        self.modelName = modelName
        self.backbone = backbone
        self.compiled = compiled
        self.computeUnits = computeUnits
        self.tokenizer = tokenizer
        self.padID = padID
        self.rotaryDim = rotaryDim
        self.ropeTheta = ropeTheta
        self.scoreBias = scoreBias
        self.functions = functions
    }

    /// Loads every function and runs it once, so no request pays for loading or for the first GPU dispatch. Largest
    /// first: switching functions costs a re-setup on the next call, so the small, common ones are left hot.
    public func warm() async throws {
        let row = Row(ids: [padID, padID, padID], candidates: [1], query: 2)
        for function in functions.reversed() { _ = try await run([row], on: function) }
    }

    public func answer(state: OrderedJSON, questions: [(id: String, question: Decision2Question)]) async throws
        -> Decision2Result
    {
        let rows = try questions.map { try encode(state: state, question: $0.question) }
        var logits = [[Double]](repeating: [], count: rows.count)
        let calls = try plan(rows)
        for (function, group) in calls {
            let out = try await run(group.map { rows[$0] }, on: function)
            for (slot, index) in group.enumerated() { logits[index] = out[slot] }
        }
        let answers = zip(questions, logits).map { item, values -> Decision2Answer in
            let options = item.question.options
            var z = values
            if case .score = item.question, let offsets = scoreBias[z.count] {
                z = zip(z, offsets).map { $0 + $1 }
            }
            let peak = z.max() ?? 0
            let e = z.map { exp($0 - peak) }
            let sum = e.reduce(0, +)
            return Decision2Answer(
                id: item.id, type: item.question.taskType, keys: options.map(\.key), probabilities: e.map { $0 / sum })
        }
        return Decision2Result(answers: answers, inputTokens: rows.reduce(0) { $0 + $1.ids.count }, calls: calls.count)
    }

    /// The token ids of one question's upstream row (for parity checks).
    public func tokens(state: OrderedJSON, question: Decision2Question) throws -> [Int] {
        try encode(state: state, question: question).ids
    }

    // MARK: Prompt (upstream decision_model.segments / encode)

    func encode(state: OrderedJSON, question: Decision2Question) throws -> Row {
        let prefix =
            "Context:\n\(state.decisionPayload)\n\nTask type: \(question.taskType)\nQuestion:\n"
            + "\(question.instructions.decisionPayload)\nOptions:"
        var ids = try tokenizer.encode(prefix)
        var candidates: [Int] = []
        for option in question.options {
            let body = OrderedJSON.object([("key", .string(option.key)), ("description", option.description)])
            let piece = try tokenizer.encode("\n<option>\n" + body.canonical + "\n</option>")
            guard !piece.isEmpty else { throw Decision2Error.invalidQuestion("empty option") }
            ids += piece
            candidates.append(ids.count - 1)
        }
        ids += try tokenizer.encode(
            "\n\nSelect the single option best supported by the context and instructions.\nDecision:")
        return Row(ids: ids, candidates: candidates, query: ids.count - 1)
    }

    // MARK: Planning

    /// Common token prefix of the rows, cut before the first option endpoint and the last token of the shortest.
    static func prefixLength(_ rows: [Row]) -> Int {
        let limit = min(rows.map { $0.candidates[0] }.min()!, rows.map { $0.ids.count }.min()! - 1)
        let first = rows[0].ids
        for i in 0..<limit where rows.contains(where: { $0.ids[i] != first[i] }) { return i }
        return limit
    }

    func fits(_ rows: [Row], _ function: Function) -> Bool {
        let p = Self.prefixLength(rows)
        let options = rows.reduce(0) { $0 + $1.candidates.count }
        let suffix = rows.reduce(0) { $0 + $1.ids.count - p }
        switch backbone {
        case .qwen3: return p + suffix <= function.l && options <= function.n
        case .qwen35: return p <= function.s && suffix <= function.l && options <= function.n
        }
    }

    /// One call in the smallest function that fits, or greedy in-order chunks on the cheapest function.
    func plan(_ rows: [Row]) throws -> [(Function, [Int])] {
        var best: (Double, [(Function, [Int])])?
        candidates: for function in functions {
            var groups: [[Int]] = []
            var group: [Int] = []
            for index in rows.indices {
                if fits((group + [index]).map { rows[$0] }, function) {
                    group.append(index)
                    continue
                }
                guard !group.isEmpty, fits([rows[index]], function) else { continue candidates }
                groups.append(group)
                group = [index]
            }
            groups.append(group)
            let cost = Double(groups.count) * function.costMs
            if best == nil || cost < best!.0 { best = (cost, groups.map { (function, $0) }) }
        }
        guard let best else {
            throw Decision2Error.tooLong("a question exceeds the largest function \(functions.last?.name ?? "")")
        }
        return best.1
    }

    // MARK: Inputs and prediction

    func run(_ rows: [Row], on function: Function) async throws -> [[Double]] {
        let model: MLModel
        do {
            model = try await cache.model(function.name, at: compiled, units: computeUnits)
        } catch {
            throw Decision2Error.invalidAsset("function \(function.name): \(error.localizedDescription)")
        }
        let (features, owners) = try backbone == .qwen3 ? qwen3Inputs(rows, function) : qwen35Inputs(rows, function)
        let output = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw Decision2Error.invalidOutput("logits")
        }
        var per = [[Double]](repeating: [], count: rows.count)
        for (slot, owner) in owners.enumerated() { per[owner].append(logits[slot].doubleValue) }
        return per
    }

    /// Kai: [1, L] ids and positions, additive [1, 1, L, L] mask (a suffix sees the prefix and itself, causally).
    func qwen3Inputs(_ rows: [Row], _ function: Function) throws -> ([String: Any], [Int]) {
        let L = function.l
        let p = Self.prefixLength(rows)
        var ids = Array(rows[0].ids[0..<p])
        var positions = Array(0..<p)
        var segment = [Int](repeating: -1, count: p)
        var candidates: [Int32] = []
        var queries: [Int32] = []
        var owners: [Int] = []
        for (j, row) in rows.enumerated() {
            let shift = ids.count - p
            ids += row.ids[p...]
            positions += Array(p..<row.ids.count)
            segment += [Int](repeating: j, count: row.ids.count - p)
            for c in row.candidates {
                candidates.append(Int32(c < p ? c : c + shift))
                queries.append(Int32(row.query + shift))
                owners.append(j)
            }
        }
        let t = ids.count
        let inputIDs = try array([1, L], .int32)
        let positionIDs = try array([1, L], .int32)
        let mask = try array([1, 1, L, L], .float16)
        let idp = inputIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        let posp = positionIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<L {
            idp[i] = Int32(i < t ? ids[i] : padID)
            posp[i] = Int32(i < t ? positions[i] : 0)
        }
        let m = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        m.update(repeating: Self.maskValue, count: L * L)
        for i in 0..<L {
            m[i * L + i] = 0
            guard i < t else { continue }
            for j in 0...i where segment[j] == -1 || segment[j] == segment[i] { m[i * L + j] = 0 }
        }
        return (
            [
                "input_ids": inputIDs, "position_ids": positionIDs, "mask": mask,
                "cand_idx": try indices(candidates, function.n), "query_idx": try indices(queries, function.n),
            ], owners
        )
    }

    /// Eos: prefix right-padded to S, suffixes packed in C, plus the masks that make every suffix restart from the
    /// prefix's recurrent state and conv history (see the model card's "How it works").
    func qwen35Inputs(_ rows: [Row], _ function: Function) throws -> ([String: Any], [Int]) {
        let S = function.s
        let C = function.l
        let T = S + C
        let lags = Self.lags
        let p = Self.prefixLength(rows)
        let inputIDs = try array([1, T], .int32)
        let valid = try array([S], .float16)
        let tail = try array([lags, S], .float16)
        let segment = try array([C, C], .float16)
        let keep = try array([lags, C], .float16)
        let lagTail = try array([lags, C, lags], .float16)
        let idp = inputIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        let vp = valid.dataPointer.assumingMemoryBound(to: Float16.self)
        let tp = tail.dataPointer.assumingMemoryBound(to: Float16.self)
        let sp = segment.dataPointer.assumingMemoryBound(to: Float16.self)
        let kp = keep.dataPointer.assumingMemoryBound(to: Float16.self)
        let lp = lagTail.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<T { idp[i] = Int32(padID) }
        var positions = [Int](repeating: p, count: T)
        for i in 0..<p {
            idp[i] = Int32(rows[0].ids[i])
            positions[i] = i
            vp[i] = 1
        }
        for i in 0..<lags where p - lags + i >= 0 { tp[i * S + p - lags + i] = 1 }
        for i in 0..<C { sp[i * C + i] = 1 }
        var candidates: [Int32] = []
        var queries: [Int32] = []
        var owners: [Int] = []
        var start = 0
        for (j, row) in rows.enumerated() {
            let n = row.ids.count - p
            for k in 0..<n {
                idp[S + start + k] = Int32(row.ids[p + k])
                positions[S + start + k] = p + k
                for i in 0...k { sp[(start + k) * C + start + i] = 1 }
                for s in 1...lags {
                    if k >= s {
                        kp[(s - 1) * C + start + k] = 1
                    } else {
                        lp[((s - 1) * C + start + k) * lags + lags + k - s] = 1
                    }
                }
            }
            for c in row.candidates {
                candidates.append(Int32(start + c - p))
                queries.append(Int32(start + row.query - p))
                owners.append(j)
            }
            start += n
        }
        // chunked delta-rule masks, from the segment matrix
        let chunks = C / Self.chunk
        let segChunks = try array([chunks, Self.chunk, Self.chunk], .float16)
        let cont = try array([C], .float16)
        let lastSeg = try array([chunks, Self.chunk], .float16)
        let scp = segChunks.dataPointer.assumingMemoryBound(to: Float16.self)
        let cp = cont.dataPointer.assumingMemoryBound(to: Float16.self)
        let lsp = lastSeg.dataPointer.assumingMemoryBound(to: Float16.self)
        for c in 0..<chunks {
            let base = c * Self.chunk
            let last = base + Self.chunk - 1
            for a in 0..<Self.chunk {
                for b in 0..<Self.chunk { scp[(c * Self.chunk + a) * Self.chunk + b] = sp[(base + a) * C + base + b] }
                lsp[c * Self.chunk + a] = sp[last * C + base + a]
            }
        }
        for i in 0..<C {
            var first = i
            for j in 0..<C where sp[i * C + j] != 0 {
                first = j
                break
            }
            cp[i] = first < (i / Self.chunk) * Self.chunk ? 1 : 0
        }
        let (cos, sin) = try rope(positions)
        return (
            [
                "input_ids": inputIDs, "cos": cos, "sin": sin, "valid": valid, "tail_onehot": tail,
                "segment": segment, "lag_keep": keep, "lag_tail": lagTail, "seg_chunks": segChunks, "cont": cont,
                "last_seg": lastSeg, "cand_idx": try indices(candidates, function.n),
                "query_idx": try indices(queries, function.n),
            ], owners
        )
    }

    /// Text-only M-RoPE tables (all three axes share the position), computed in Double as upstream does in float64.
    private func rope(_ positions: [Int]) throws -> (MLMultiArray, MLMultiArray) {
        let cos = try array([positions.count, rotaryDim], .float16)
        let sin = try array([positions.count, rotaryDim], .float16)
        let c = cos.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sin.dataPointer.assumingMemoryBound(to: Float16.self)
        let half = rotaryDim / 2
        for (row, position) in positions.enumerated() {
            for i in 0..<half {
                let angle = Double(position) * (1.0 / pow(ropeTheta, Double(2 * i) / Double(rotaryDim)))
                let cv = Float16(Foundation.cos(angle))
                let sv = Float16(Foundation.sin(angle))
                c[row * rotaryDim + i] = cv
                c[row * rotaryDim + i + half] = cv
                s[row * rotaryDim + i] = sv
                s[row * rotaryDim + i + half] = sv
            }
        }
        return (cos, sin)
    }

    private func indices(_ values: [Int32], _ n: Int) throws -> MLMultiArray {
        guard values.count <= n else { throw Decision2Error.tooLong("\(values.count) options > \(n) slots") }
        let out = try array([n], .int32)
        let pointer = out.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, v) in values.enumerated() { pointer[i] = v }
        return out
    }

    private func array(_ dims: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: type)
        switch type {
        case .int32: array.dataPointer.initializeMemory(as: Int32.self, repeating: 0, count: array.count)
        default: array.dataPointer.initializeMemory(as: Float16.self, repeating: 0, count: array.count)
        }
        return array
    }
}
