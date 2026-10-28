@preconcurrency import CoreML
import Foundation

/// A Vela 3.0 decision question: pick one option, or a yes / no statement (answered as the probability of yes).
public enum Vela3Question: Sendable {
    case choice(id: String, instructions: String, options: [(name: String, description: String)])
    case yesNo(id: String, instructions: String)

    var id: String {
        switch self {
        case .choice(let id, _, _), .yesNo(let id, _): id
        }
    }
}

public struct Vela3Decision: Sendable {
    public let id: String
    public let names: [String]
    public let probabilities: [Double]
    public let answer: String
    public func probability(_ name: String) -> Double? { names.firstIndex(of: name).map { probabilities[$0] } }
    /// For a yes / no question: the probability of yes.
    public var yes: Double? { probability("yes") }
}

public struct Vela3Span: Sendable {
    public let label: String
    /// Unicode scalar offsets into the text the span is in.
    public let start: Int
    public let end: Int
    public let text: String
    public let probability: Double
}

public struct Vela3Timing: Sendable {
    /// Core ML encoder time and the whole call (tokenize, encode, heads, decode), in milliseconds.
    public let encoderMs: Double
    public let totalMs: Double
    /// The encoder function (sequence length) of each pass.
    public let buckets: [Int]
}

public struct Vela3DecisionResult: Sendable {
    public let decisions: [Vela3Decision]
    public let timing: Vela3Timing
    public subscript(id: String) -> Vela3Decision? { decisions.first { $0.id == id } }
}

public struct Vela3SpanResult: Sendable {
    public let spans: [Vela3Span]
    public let threshold: Double
    /// Encoder passes the target text was read in (1 unless it was longer than one pass's budget).
    public let windows: Int
    public let timing: Vela3Timing
}

public enum Vela3Error: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case invalidRequest(String)
    case tooLong(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let r): "Invalid Vela 3.0 asset: \(r)"
        case .invalidRequest(let r): "Invalid Vela 3.0 request: \(r)"
        case .tooLong(let r): "Vela 3.0 request too long: \(r)"
        }
    }
}

/// Vela 3.0 vela3-fast (vLLM Semantic Router × KR Labs; UltraBERT-380M encoder) on Core ML, from
/// [FluidInference/vela-3.0-coreml](https://huggingface.co/FluidInference/vela-3.0-coreml): decisions over options you
/// name (the decision backbone) and the two label sets of the release's `span_sets` adapter, personal information (17
/// types) and claims of an answer its context does not support. One encoder pass per question on the GPU (the Neural
/// Engine is slower for this encoder at every length and drifts further from the fp32 answers). Rendering, readout,
/// thresholds, the span decoder and long-input handling reproduce the release's `vela3_encoder.py` / `vela3_schema.py`
/// with a 1,024-token pass: a decision longer than that throws `tooLong`; for spans the other part (the context of an
/// unsupported-claim check) is cut first, then the target text is read in overlapping windows (the release's own rule,
/// as the Python runtime of the snapshot; the PyTorch release windows at 8,192 tokens).
@available(macOS 15.0, iOS 18.0, *)
public final class Vela3Manager: Sendable {
    struct Markers: Sendable {
        let q, o, e, sep: Int
        let seg: [String: Int]
    }

    /// A trained label set with its schema prefix (`<bos> ([E] label: description)*`) tokenized once.
    struct SpanSet: Sendable {
        let names: [String]
        let role: String
        let prefix: [Int]
        let ePos: [Int]
    }

    private let tokenizer: Vela3Tokenizer
    private let markers: Markers
    private let bos: Int
    private let eos: Int
    private let decisionHeads: Vela2Heads
    private let spanHeads: Vela2Heads
    private let decisionModels: [(length: Int, model: MLModel)]
    private let spanModels: [(length: Int, model: MLModel)]
    private let pii: SpanSet
    private let unsupported: SpanSet
    /// The set adapter's span temperature, PII thresholds by target length, and the unsupported-claim threshold.
    private let spanTemperature: Double
    private let piiBuckets: [(maxTokens: Int, threshold: Double)]
    private let unsupportedThreshold: Double

    public static let decisionLengths = [128, 256, 512, 1024]
    public static let spanLengths = [512, 1024]
    /// The release's long-input constants: the other part keeps at least this many tokens before the target is cut,
    /// and windows overlap by up to `overlap` tokens (a quarter of the window at most).
    static let shrinkFloor = 64
    static let overlap = 512

    public static func load(from directory: URL) async throws -> Vela3Manager {
        func json(_ name: String) throws -> [String: Any] {
            guard let o = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent(name))) as? [String: Any]
            else { throw Vela3Error.invalidAsset(name) }
            return o
        }
        let config = try json("config.json")
        let vela = try json("vela3_config.json")
        guard let decision = config["decision"] as? [String: Any], let marker = decision["marker_ids"] as? [String: Int],
            let bos = config["bos_token_id"] as? Int, let eos = config["eos_token_id"] as? Int
        else { throw Vela3Error.invalidAsset("config.json is missing fields") }
        func m(_ k: String) throws -> Int {
            guard let v = marker[k] else { throw Vela3Error.invalidAsset("marker \(k)") }
            return v
        }
        guard let span = vela["span"] as? [String: Any], let sets = span["router_label_sets"] as? [String: [String]],
            let schemas = span["schemas"] as? [String: [String: Any]], let setAdapter = span["set_adapter"] as? [String: Any],
            let temperature = setAdapter["router_set_temperature"] as? Double,
            let buckets = setAdapter["pii_buckets"] as? [[String: Any]], !buckets.isEmpty,
            let byset = setAdapter["thresholds_by_set"] as? [String: Double], let halu = byset["halu"]
        else { throw Vela3Error.invalidAsset("vela3_config.json is missing the span_sets settings") }
        let piiBuckets = try buckets.map { b -> (Int, Double) in
            guard let t = b["threshold"] as? Double, let n = b["max_tokens"] as? NSNumber else {
                throw Vela3Error.invalidAsset("vela3_config.json pii_buckets entry \(b)")
            }
            return (n.intValue, t)
        }
        let tokenizer = try Vela3Tokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let e = try m("[E]")
        func spanSet(_ name: String) throws -> SpanSet {
            guard let names = sets[name], !names.isEmpty, Set(names).count == names.count, let schema = schemas[name],
                let labels = schema["labels"] as? [String: String], let over = schema["over"] as? String
            else { throw Vela3Error.invalidAsset("vela3_config.json label set \(name)") }
            var prefix = [bos]
            var ePos: [Int] = []
            for n in names {
                let d = labels[n] ?? ""
                ePos.append(prefix.count)
                prefix.append(e)
                prefix += tokenizer.encode(d.isEmpty ? n : "\(n): \(d)")
            }
            return SpanSet(names: names, role: over, prefix: prefix, ePos: ePos)
        }
        func heads(_ file: String, span: Bool) throws -> Vela2Heads {
            do {
                return try Vela2Heads(weights: try safetensors(directory.appendingPathComponent(file)), span: span)
            } catch let error as Vela3Error {
                throw error
            } catch {
                throw Vela3Error.invalidAsset("\(file): \(error.localizedDescription)")
            }
        }
        let decisionHeads = try heads("heads_decision.safetensors", span: false)
        let spanHeads = try heads("heads_span_sets.safetensors", span: true)
        guard decisionHeads.hidden == spanHeads.hidden else {
            throw Vela3Error.invalidAsset("decision and span heads disagree on the hidden size")
        }
        let decisionModels = try await models(directory.appendingPathComponent("Vela3Decision.mlpackage"), lengths: decisionLengths)
        let spanModels = try await models(directory.appendingPathComponent("Vela3SpanSets.mlpackage"), lengths: spanLengths)
        return Vela3Manager(
            tokenizer: tokenizer,
            markers: Markers(
                q: try m("[Q]"), o: try m("[O]"), e: e, sep: try m("[SEP_SCHEMA]"),
                seg: ["user": try m("[SEG_user]"), "context": try m("[SEG_context]"), "answer": try m("[SEG_answer]")]),
            bos: bos, eos: eos, decisionHeads: decisionHeads, spanHeads: spanHeads, decisionModels: decisionModels,
            spanModels: spanModels, pii: try spanSet("pii"), unsupported: try spanSet("halu"), spanTemperature: temperature,
            piiBuckets: piiBuckets, unsupportedThreshold: halu)
    }

    private static func models(_ package: URL, lengths: [Int]) async throws -> [(length: Int, model: MLModel)] {
        let compiled = try await KevManager.compiled(package)
        var out: [(Int, MLModel)] = []
        for length in lengths {
            let configuration = MLModelConfiguration()
            configuration.functionName = "L\(length)"
            configuration.computeUnits = .cpuAndGPU
            out.append((length, try await MLModel.load(contentsOf: compiled, configuration: configuration)))
        }
        return out
    }

    /// fp32 tensors of a safetensors file by name.
    static func safetensors(_ url: URL) throws -> [String: [Float]] {
        let data = try Data(contentsOf: url)
        let name = url.lastPathComponent
        guard data.count >= 8 else { throw Vela3Error.invalidAsset("\(name) is too short") }
        let length = data.prefix(8).enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << (8 * UInt64($1.offset))) }
        guard length <= UInt64(data.count - 8) else { throw Vela3Error.invalidAsset("\(name) header runs past the file") }
        let base = 8 + Int(length)
        guard let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<base)) as? [String: Any] else {
            throw Vela3Error.invalidAsset("\(name) header")
        }
        var out: [String: [Float]] = [:]
        for (tensor, value) in header where tensor != "__metadata__" {
            guard let meta = value as? [String: Any], meta["dtype"] as? String == "F32", let range = meta["data_offsets"] as? [Int],
                range.count == 2, let shape = meta["shape"] as? [Int]
            else { throw Vela3Error.invalidAsset("\(name): \(tensor) is not an F32 tensor") }
            let (a, b) = (range[0], range[1])
            guard a >= 0, a <= b, b <= data.count - base, (b - a) == 4 * shape.reduce(1, *) else {
                throw Vela3Error.invalidAsset("\(name): \(tensor) offsets \(range) do not match its shape \(shape)")
            }
            out[tensor] = data.subdata(in: (base + a)..<(base + b)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        return out
    }

    init(
        tokenizer: Vela3Tokenizer, markers: Markers, bos: Int, eos: Int, decisionHeads: Vela2Heads, spanHeads: Vela2Heads,
        decisionModels: [(length: Int, model: MLModel)], spanModels: [(length: Int, model: MLModel)], pii: SpanSet,
        unsupported: SpanSet, spanTemperature: Double, piiBuckets: [(maxTokens: Int, threshold: Double)], unsupportedThreshold: Double
    ) {
        self.tokenizer = tokenizer
        self.markers = markers
        self.bos = bos
        self.eos = eos
        self.decisionHeads = decisionHeads
        self.spanHeads = spanHeads
        self.decisionModels = decisionModels
        self.spanModels = spanModels
        self.pii = pii
        self.unsupported = unsupported
        self.spanTemperature = spanTemperature
        self.piiBuckets = piiBuckets
        self.unsupportedThreshold = unsupportedThreshold
    }

    /// Runs every encoder function once so no request pays for the first dispatch.
    public func warm() async throws {
        for (length, model) in decisionModels.reversed() {
            _ = try await encode(ids: [bos, eos], model: model, length: length, dim: decisionHeads.hidden)
        }
        for (length, model) in spanModels.reversed() {
            _ = try await encode(ids: [bos, eos], model: model, length: length, dim: spanHeads.hidden)
        }
    }

    // MARK: decisions (vela3_encoder.DecisionEncoder)

    /// Answer each question about `state` (the release reads a JSON state as compact JSON; pass that text here).
    public func decide(state: String, questions: [Vela3Question]) async throws -> Vela3DecisionResult {
        let started = Date()
        var seen = Set<String>()
        for q in questions {
            guard seen.insert(q.id).inserted else { throw Vela3Error.invalidRequest("duplicate question id \(q.id)") }
            if case .choice(let id, _, let options) = q {
                guard options.count >= 2 else { throw Vela3Error.invalidRequest("choice \(id) needs at least 2 options") }
                guard Set(options.map(\.name)).count == options.count else {
                    throw Vela3Error.invalidRequest("choice \(id) has duplicate option names")
                }
            }
        }
        let stateIDs = tokenizer.encode(state)
        var decisions: [Vela3Decision] = []
        var encoderMs = 0.0
        var used: [Int] = []
        for q in questions {
            let rec = decisionRecord(q, stateIDs: stateIDs)
            guard let (length, model) = decisionModels.first(where: { rec.ids.count <= $0.length }) else {
                throw Vela3Error.tooLong("\(rec.ids.count) tokens exceed the largest encoder function L\(Self.decisionLengths.last!)")
            }
            let t = Date()
            let hidden = try await encode(ids: rec.ids, model: model, length: length, dim: decisionHeads.hidden)
            encoderMs += Date().timeIntervalSince(t) * 1000
            used.append(length)
            let logits = decisionHeads.run(hidden: hidden, dim: decisionHeads.hidden, rec: rec).0.map(Double.init)
            let m = logits.max() ?? 0
            let e = logits.map { exp($0 - m) }
            let s = e.reduce(0, +)
            let p = e.map { $0 / s }
            let names = rec.questions[0].names
            var best = 0
            for i in 1..<p.count where p[i] > p[best] { best = i }
            decisions.append(Vela3Decision(id: q.id, names: names, probabilities: p, answer: names[best]))
        }
        return Vela3DecisionResult(
            decisions: decisions, timing: Vela3Timing(encoderMs: encoderMs, totalMs: Date().timeIntervalSince(started) * 1000, buckets: used))
    }

    /// The encoder input of a decision question (for parity checks).
    public func decisionSequence(state: String, question: Vela3Question) -> [Int] {
        decisionRecord(question, stateIDs: tokenizer.encode(state)).ids
    }

    /// `[BOS] [Q] instructions [O] key_1: description_1 [O] key_2 ... [SEP_SCHEMA] [SEG_user] state [EOS]`; a yes / no
    /// question is the 2-option choice {no, yes}. The readout pools [SEG_user] .. the end of the state.
    private func decisionRecord(_ q: Vela3Question, stateIDs: [Int]) -> Vela2Manager.Record {
        let (instructions, options): (String, [(name: String, description: String)]) =
            switch q {
            case .choice(_, let text, let options): (text, options)
            case .yesNo(_, let text): (text, [("no", ""), ("yes", "")])
            }
        var ids = [bos, markers.q] + tokenizer.encode(instructions)
        var optPos: [Int] = []
        for o in options {
            optPos.append(ids.count)
            ids.append(markers.o)
            ids += tokenizer.encode(o.description.isEmpty ? o.name : "\(o.name): \(o.description)")
        }
        ids.append(markers.sep)
        let ps = ids.count
        ids.append(markers.seg["user"]!)
        ids += stateIDs
        let pe = ids.count
        ids.append(eos)
        let rec = Vela2Manager.QuestionRec(id: q.id, qPos: 1, optPos: optPos, names: options.map(\.name), absPos: nil, pool: (ps, pe))
        return Vela2Manager.Record(ids: ids, questions: [rec], span: nil)
    }

    // MARK: spans (vela3_schema, span_sets adapter)

    /// Personal information in `text` (17 types: names, e-mail addresses, phone numbers, street addresses, card numbers, …).
    public func pii(_ text: String, threshold: Double? = nil) async throws -> Vela3SpanResult {
        try await spans(set: pii, parts: [("user", text)], threshold: threshold) { tokens in
            self.piiBuckets.first { tokens <= $0.maxTokens }?.threshold ?? self.piiBuckets[self.piiBuckets.count - 1].threshold
        }
    }

    /// The words of `answer` that `context` does not support.
    public func unsupported(context: String, answer: String, threshold: Double? = nil) async throws -> Vela3SpanResult {
        try await spans(set: unsupported, parts: [("context", context), ("answer", answer)], threshold: threshold) { _ in
            self.unsupportedThreshold
        }
    }

    /// Token ids and Unicode-scalar offsets of a text (for parity checks).
    public func tokenize(_ text: String) -> (ids: [Int], offsets: [(Int, Int)]) {
        let t = tokenizer.tokens(Array(text.unicodeScalars))
        return (t.map(\.id), t.map { ($0.start, $0.end) })
    }

    /// The encoder inputs (one per window) of a PII or unsupported-claim call (for parity checks).
    public func spanSequences(pii text: String) throws -> [[Int]] { try plan(set: pii, parts: [("user", text)]).passes.map { $0.rec.ids } }
    public func spanSequences(context: String, answer: String) throws -> [[Int]] {
        try plan(set: unsupported, parts: [("context", context), ("answer", answer)]).passes.map { $0.rec.ids }
    }

    struct Part {
        let role: String
        let scalars: [Unicode.Scalar]
        let tokens: [Vela3Tokenizer.Token]
        var ids: [Int] { tokens.map(\.id) }
    }

    struct Plan {
        /// One encoder pass per window, with the target words (indices into `words`) it reads.
        let passes: [(rec: Vela2Manager.Record, words: [Int])]
        let words: [(Int, Int)]
        let target: [Unicode.Scalar]
        let targetTokens: Int
    }

    private func spans(
        set: SpanSet, parts: [(role: String, text: String)], threshold: Double?, defaultThreshold: (Int) -> Double
    ) async throws -> Vela3SpanResult {
        let started = Date()
        let plan = try self.plan(set: set, parts: parts)
        // per word: logits averaged over the windows that read it (-30 where none did), as predict_raw
        var acc = [[Double]](repeating: [Double](repeating: 0, count: set.names.count), count: plan.words.count)
        var cnt = [Int](repeating: 0, count: plan.words.count)
        var encoderMs = 0.0
        var used: [Int] = []
        for pass in plan.passes {
            guard let (length, model) = spanModels.first(where: { pass.rec.ids.count <= $0.length }) else {
                throw Vela3Error.tooLong("\(pass.rec.ids.count) tokens exceed L\(Self.spanLengths.last!)")
            }
            let t = Date()
            let hidden = try await encode(ids: pass.rec.ids, model: model, length: length, dim: spanHeads.hidden)
            encoderMs += Date().timeIntervalSince(t) * 1000
            used.append(length)
            let logits = spanHeads.run(hidden: hidden, dim: spanHeads.hidden, rec: pass.rec).1
            for (row, w) in zip(logits, pass.words) {
                for k in 0..<row.count { acc[w][k] += Double(row[k]) }
                cnt[w] += 1
            }
        }
        let probs = zip(acc, cnt).map { row, c in
            row.map { c > 0 ? 1 / (1 + exp(-($0 / Double(c)) / spanTemperature)) : 1 / (1 + exp(30 / spanTemperature)) }
        }
        let thr = threshold ?? defaultThreshold(plan.targetTokens)
        let decoded = plan.words.isEmpty ? [] : Vela2Text.decodeSpans(probs, offsets: plan.words, text: plan.target, threshold: thr)
        let out = decoded.map {
            Vela3Span(
                label: set.names[$0.label], start: $0.start, end: $0.end,
                text: String(String.UnicodeScalarView(plan.target[$0.start..<$0.end])), probability: $0.probability)
        }
        return Vela3SpanResult(
            spans: out, threshold: thr, windows: plan.passes.count,
            timing: Vela3Timing(encoderMs: encoderMs, totalMs: Date().timeIntervalSince(started) * 1000, buckets: used))
    }

    /// The release's `_assemble` / `assemble` / `_windows` for one span question in a 1,024-token pass: cut the part
    /// the question is not over (to `shrinkFloor`, then to 1 token), then, if the target still does not fit, read it in
    /// windows of the room left, overlapping by min(`overlap`, window / 4) tokens.
    func plan(set: SpanSet, parts raw: [(role: String, text: String)]) throws -> Plan {
        let parts = raw.map { p -> Part in
            let scalars = Array(p.text.unicodeScalars)
            return Part(role: p.role, scalars: scalars, tokens: tokenizer.tokens(scalars))
        }
        guard let target = parts.first(where: { $0.role == set.role }) else { throw Vela3Error.invalidRequest("no \(set.role) part") }
        let (words, first) = Vela2Text.wordFirstTokens(target.scalars, offsets: target.tokens.map { ($0.start, $0.end) })
        let maxLength = Self.spanLengths.last!
        let fixed = set.prefix.count + 1 + 1 + parts.count  // prefix (with <bos>), [SEP_SCHEMA], <eos>, one [SEG] per part
        let t = parts.firstIndex { $0.role == set.role }!
        let shrink = parts.firstIndex { $0.role != set.role }
        /// Part lengths after `assemble`'s cuts for a target of `targetCount` tokens: the other part first (to the
        /// floor, then to 1 token), then the target. Each window is re-assembled this way, so a short last window
        /// gives the other part its room back.
        func fit(_ targetCount: Int) -> [Int] {
            var lengths = parts.map { $0.tokens.count }
            lengths[t] = targetCount
            func over() -> Int { fixed + lengths.reduce(0, +) - maxLength }
            if let s = shrink {
                if over() > 0 { lengths[s] = max(min(lengths[s], Self.shrinkFloor), lengths[s] - over()) }
                if over() > 0 { lengths[s] = max(1, lengths[s] - over()) }
            }
            if over() > 0 { lengths[t] = max(1, lengths[t] - over()) }
            return lengths
        }
        let n = target.tokens.count
        let lengths = fit(n)
        guard fixed + lengths.reduce(0, +) <= maxLength else {
            throw Vela3Error.tooLong("the label schema alone does not fit in \(maxLength) tokens")
        }
        let cut = lengths[t] < n
        let partIDs = parts.map(\.ids)
        func record(_ a: Int, _ b: Int) -> (Vela2Manager.Record, [Int]) {
            let lengths = fit(b - a)
            var ids = set.prefix
            ids.append(markers.sep)
            var span: Vela2Manager.SpanRec?
            var sel: [Int] = []
            for (k, p) in parts.enumerated() {
                let start = ids.count
                ids.append(markers.seg[p.role]!)
                if k == t {
                    ids += partIDs[k][a..<b]
                    sel = first.indices.filter { first[$0] >= a && first[$0] < b }
                    span = Vela2Manager.SpanRec(
                        id: set.role, role: p.role, names: set.names, ePos: set.ePos, wPos: sel.map { start + 1 + first[$0] - a },
                        wOffsets: sel.map { words[$0] }, tokenCount: b - a)
                } else {
                    ids += partIDs[k].prefix(lengths[k])
                }
            }
            ids.append(eos)
            return (Vela2Manager.Record(ids: ids, questions: [], span: span), sel)
        }
        var passes: [(rec: Vela2Manager.Record, words: [Int])] = []
        if !cut {
            passes.append(record(0, n))
        } else {
            let window = lengths[t]
            let overlap = min(Self.overlap, window / 4)
            let stride = max(1, window - overlap)
            var a = 0
            while a < max(1, n - overlap) {
                let b = min(n, a + window)
                passes.append(record(a, b))
                if b >= n { break }
                a += stride
            }
        }
        return Plan(passes: passes, words: words, target: target.scalars, targetTokens: n)
    }

    // MARK: encoder

    private static func predict(_ model: MLModel, _ input: MLFeatureProvider, options: MLPredictionOptions) throws -> MLFeatureProvider {
        try model.prediction(from: input, options: options)
    }

    private func encode(ids: [Int], model: MLModel, length: Int, dim: Int) async throws -> [Float] {
        let input = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let ip = input.dataPointer.assumingMemoryBound(to: Int32.self)
        let mp = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<length {
            ip[i] = i < ids.count ? Int32(ids[i]) : 0
            mp[i] = i < ids.count ? 1 : 0
        }
        // The hidden states go into a buffer we allocate (output backing), with a synchronous prediction. Letting Core ML
        // allocate the output, the first GPU prediction on a cold compile cache aborted in MPSGraphTensorData
        // (`shape.count = 0 != strides.count = 3`) in about 1 of 3 cold runs (macOS 27.0); with the backing, 0 of 20.
        let hidden = try MLMultiArray(shape: [1, NSNumber(value: length), NSNumber(value: dim)], dataType: .float16)
        let options = MLPredictionOptions()
        options.outputBackings = ["hidden": hidden]
        _ = try Self.predict(model, MLDictionaryFeatureProvider(dictionary: ["input_ids": input, "attention_mask": mask]), options: options)
        let strides = hidden.strides.map(\.intValue)
        var h = [Float](repeating: 0, count: ids.count * dim)
        let src = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        for t in 0..<ids.count {
            for c in 0..<dim { h[t * dim + c] = Float(src[t * strides[1] + c * strides[2]]) }
        }
        return h
    }
}
