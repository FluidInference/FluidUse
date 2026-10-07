import Accelerate
@preconcurrency import CoreML
import Foundation

/// A typed part of a Vela 2.0 request: `user` (the request), `context` (a source document) or `answer`.
public struct Vela2Part: Sendable {
    public let role: String
    public let text: String
    public init(_ role: String, _ text: String) {
        self.role = role
        self.text = text
    }
}

/// A Vela 2.0 question over one or more parts (the engine's `predict()` contract).
public enum Vela2Question: Sendable {
    /// Pick one option (`name`, `description`); `abstain` adds the `[ABS]` endpoint (on by default, as upstream).
    case choice(id: String, text: String, options: [(name: String, description: String)], over: [String], abstain: Bool = true)
    /// Mark spans of one part with these labels.
    case span(id: String, text: String, labels: [(name: String, description: String)], over: String)

    var id: String {
        switch self {
        case .choice(let id, _, _, _, _), .span(let id, _, _, _): id
        }
    }
}

public struct Vela2ChoiceAnswer: Sendable {
    public let id: String
    public let names: [String]
    /// Calibrated softmax over the real options.
    public let probabilities: [Double]
    public let answer: String
    public let abstainProbability: Double?
    public func probability(_ name: String) -> Double? { names.firstIndex(of: name).map { probabilities[$0] } }
}

public struct Vela2Span: Sendable {
    public let label: String
    /// Unicode scalar offsets into the part's text.
    public let start: Int
    public let end: Int
    public let text: String
    public let probability: Double
}

public struct Vela2SpanAnswer: Sendable {
    public let id: String
    public let role: String
    public let spans: [Vela2Span]
    public let threshold: Double
    /// The highest label probability of any word (upstream's yes/no view of a span question).
    public let maxProbability: Double
}

public struct Vela2Result: Sendable {
    public let choices: [Vela2ChoiceAnswer]
    public let spans: [Vela2SpanAnswer]
    /// Core ML encoder time and the whole request (tokenize, encode, heads, decode), in milliseconds.
    public let encoderMs: Double
    public let totalMs: Double
    /// Encoder calls (one per sequence; each extra span question is its own sequence) and their buckets.
    public let buckets: [Int]
    /// Whether every encoder call of this request ran on the Neural Engine bucket(s).
    public let onNeuralEngine: Bool
    public subscript(choice id: String) -> Vela2ChoiceAnswer? { choices.first { $0.id == id } }
    public subscript(span id: String) -> Vela2SpanAnswer? { spans.first { $0.id == id } }
}

public enum Vela2Error: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case invalidRequest(String)
    case tooLong(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let r): "Invalid Vela 2.0 asset: \(r)"
        case .invalidRequest(let r): "Invalid Vela 2.0 request: \(r)"
        case .tooLong(let r): "Vela 2.0 request too long: \(r)"
        }
    }
}

/// Vela 2.0 0.3B (vLLM Semantic Router × KR Labs; ModernBERT encoder) on Core ML. Every question of a request is read
/// in one encoder pass (span questions beyond the first get their own pass, as upstream); short sequences run on the
/// Neural Engine, longer ones on the GPU. Tokenization, assembly, heads, calibration and the span decoder reproduce
/// the release's `vela2_inference.py`.
@available(macOS 15.0, iOS 18.0, *)
public final class Vela2Manager: Sendable {
    struct Markers: Sendable {
        let q, o, abs, sep, e: Int
        let seg: [String: Int]
    }

    struct Calibration: Sendable {
        let temperature: [String: Double]
        let thresholds: [String: Double]
        let piiTypes: Set<String>
        let piiAnchors: [(tokens: Double, threshold: Double)]
        let sparseK: Int
        let sparseThreshold: Double
        let probeThreshold: Double
    }

    public let modelName: String
    private let tokenizer: Vela2Tokenizer
    private let markers: Markers
    private let bos: Int
    private let eos: Int
    private let calibration: Calibration
    private let heads: Vela2Heads
    private let buckets: [(length: Int, model: MLModel)]
    private let partCache = PartCache()
    /// Buckets up to this length run on the Neural Engine, longer ones on the GPU.
    public let aneMaxLength: Int

    public static func load(from directory: URL, aneMaxLength: Int? = nil) async throws -> Vela2Manager {
        func json(_ name: String) throws -> [String: Any] {
            guard let o = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent(name))) as? [String: Any]
            else { throw Vela2Error.invalidAsset(name) }
            return o
        }
        let config = try json("coreml_config.json")
        let cal = try json("calibration.json")
        guard let marker = config["marker_ids"] as? [String: Int], let package = config["package"] as? String,
            let functions = config["functions"] as? [String: Any], let bos = config["bos_token_id"] as? Int,
            let eos = config["eos_token_id"] as? Int
        else { throw Vela2Error.invalidAsset("coreml_config.json is missing fields") }
        let aneMax = aneMaxLength ?? (config["ane_max_length"] as? Int ?? 0)
        var compiled = directory.appendingPathComponent((package as NSString).deletingPathExtension + ".mlmodelc")
        if !FileManager.default.fileExists(atPath: compiled.path) {
            compiled = try await KevManager.compiled(directory.appendingPathComponent(package))
        }
        var buckets: [(Int, MLModel)] = []
        for name in functions.keys {
            guard let length = Int(name.dropFirst()) else { continue }
            let configuration = MLModelConfiguration()
            configuration.functionName = name
            configuration.computeUnits = length <= aneMax ? .cpuAndNeuralEngine : .cpuAndGPU
            buckets.append((length, try await MLModel.load(contentsOf: compiled, configuration: configuration)))
        }
        guard let temps = cal["temperature"] as? [String: Double], let thr = cal["thresholds"] as? [String: Double],
            let types = cal["pii_types"] as? [String], let rule = cal["pii_length_rule"] as? [String: Any],
            let anchors = rule["anchors"] as? [[String: Double]], let gate = cal["pii_sparse_gate"] as? [String: Any]
        else { throw Vela2Error.invalidAsset("calibration.json is missing fields") }
        func m(_ k: String) throws -> Int {
            guard let v = marker[k] else { throw Vela2Error.invalidAsset("marker \(k)") }
            return v
        }
        return Vela2Manager(
            modelName: config["model_name"] as? String ?? "Vela-2.0-0.3B",
            tokenizer: try Vela2Tokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json")),
            markers: Markers(
                q: try m("[Q]"), o: try m("[O]"), abs: try m("[ABS]"), sep: try m("[SEP_SCHEMA]"), e: try m("[E]"),
                seg: ["user": try m("[SEG_user]"), "context": try m("[SEG_context]"), "answer": try m("[SEG_answer]")]),
            bos: bos, eos: eos,
            calibration: Calibration(
                temperature: temps, thresholds: thr, piiTypes: Set(types),
                piiAnchors: anchors.map { ($0["n_tokens"]!, $0["threshold"]!) }, sparseK: (gate["K"] as? Int) ?? -1,
                sparseThreshold: (gate["t_sparse"] as? Double) ?? 0, probeThreshold: (gate["probe_threshold"] as? Double) ?? 0.5),
            heads: try Vela2Heads(directory: directory), buckets: buckets.sorted { $0.0 < $1.0 }, aneMaxLength: aneMax)
    }

    init(
        modelName: String, tokenizer: Vela2Tokenizer, markers: Markers, bos: Int, eos: Int, calibration: Calibration,
        heads: Vela2Heads, buckets: [(length: Int, model: MLModel)], aneMaxLength: Int
    ) {
        self.aneMaxLength = aneMaxLength
        self.modelName = modelName
        self.tokenizer = tokenizer
        self.markers = markers
        self.bos = bos
        self.eos = eos
        self.calibration = calibration
        self.heads = heads
        self.buckets = buckets
    }

    /// Runs every bucket once so no request pays for loading or for the first dispatch (largest first: the small,
    /// common bucket is left hot).
    public func warm() async throws {
        for (length, model) in buckets.reversed() {
            _ = try await encode(ids: [bos, eos] + [Int](repeating: bos, count: max(0, min(length, 8) - 2)), model: model, length: length)
        }
    }

    // MARK: predict

    public func predict(parts: [Vela2Part], questions: [Vela2Question]) async throws -> Vela2Result {
        let started = Date()
        try Self.validate(questions)
        let rows = Self.rows(questions)
        var choices: [Vela2ChoiceAnswer] = []
        var spans: [Vela2SpanAnswer] = []
        var encoderMs = 0.0
        var used: [Int] = []
        for row in rows {
            let rec = try assemble(parts: parts, questions: row)
            guard let (length, model) = buckets.first(where: { rec.ids.count <= $0.length }) else {
                throw Vela2Error.tooLong("\(rec.ids.count) tokens exceed the largest bucket \(buckets.last?.length ?? 0)")
            }
            let t = Date()
            let hidden = try await encode(ids: rec.ids, model: model, length: length)
            encoderMs += Date().timeIntervalSince(t) * 1000
            used.append(length)
            let (optLogits, spanLogits) = heads.run(hidden: hidden, dim: heads.hidden, rec: rec)
            var k = 0
            for q in rec.questions {
                let n = q.optPos.count
                let lg = optLogits[k..<(k + n + (q.absPos == nil ? 0 : 1))].map(Double.init)
                k += n + (q.absPos == nil ? 0 : 1)
                choices.append(choiceAnswer(q, Array(lg)))
            }
            if let span = rec.span {
                spans.append(spanAnswer(span, logits: spanLogits, parts: parts))
            }
        }
        let byChoice = Dictionary(uniqueKeysWithValues: choices.map { ($0.id, $0) })
        let bySpan = Dictionary(uniqueKeysWithValues: spans.map { ($0.id, $0) })
        return Vela2Result(
            choices: questions.compactMap { byChoice[$0.id] }, spans: questions.compactMap { bySpan[$0.id] },
            encoderMs: encoderMs, totalMs: Date().timeIntervalSince(started) * 1000, buckets: used,
            onNeuralEngine: !used.isEmpty && used.allSatisfy { $0 <= aneMaxLength })
    }

    /// Token ids and Unicode-scalar offsets of a text (for parity checks).
    public func tokenize(_ text: String) -> (ids: [Int], offsets: [(Int, Int)]) {
        let t = tokenizer.tokens(Array(text.unicodeScalars))
        return (t.map(\.id), t.map { ($0.start, $0.end) })
    }

    /// The encoder input of each sequence a request runs as (for parity checks).
    public func sequences(parts: [Vela2Part], questions: [Vela2Question]) throws -> [[Int]] {
        try Self.validate(questions)
        return try Self.rows(questions).map { try assemble(parts: parts, questions: $0).ids }
    }

    /// Upstream's request rules: unique ids, 2…255 choice options, 1…255 span labels.
    static func validate(_ questions: [Vela2Question]) throws {
        var seen = Set<String>()
        for q in questions {
            guard seen.insert(q.id).inserted else { throw Vela2Error.invalidRequest("duplicate question id \(q.id)") }
            switch q {
            case .choice(let id, _, let options, let over, _):
                guard (2...255).contains(options.count) else {
                    throw Vela2Error.invalidRequest("choice \(id) needs 2 to 255 options, got \(options.count)")
                }
                guard !over.isEmpty else { throw Vela2Error.invalidRequest("choice \(id) is over no part") }
            case .span(let id, _, let labels, _):
                guard (1...255).contains(labels.count) else {
                    throw Vela2Error.invalidRequest("span \(id) needs 1 to 255 labels, got \(labels.count)")
                }
            }
        }
    }

    /// Choice questions and the first span question share a sequence; each further span question gets its own.
    static func rows(_ questions: [Vela2Question]) -> [[Vela2Question]] {
        var spans: [Vela2Question] = []
        var first: [Vela2Question] = []
        for q in questions {
            if case .span = q { spans.append(q) } else { first.append(q) }
        }
        if let s = spans.first { first.append(s) }
        var rows: [[Vela2Question]] = first.isEmpty ? [] : [first]
        for s in spans.dropFirst() { rows.append([s]) }
        return rows
    }

    // MARK: assembly (schema_lib.assemble, eval path)

    struct PartTokens: Sendable {
        let scalars: [Unicode.Scalar]
        let ids: [Int]
        let offsets: [(Int, Int)]
        /// For a span target: word character spans and each word's first token index.
        var words: [(Int, Int)] = []
        var first: [Int] = []
    }

    struct QuestionRec: Sendable {
        let id: String
        let qPos: Int
        let optPos: [Int]
        let names: [String]
        let absPos: Int?
        let pool: (Int, Int)
    }

    struct SpanRec: Sendable {
        let id: String
        let role: String
        let names: [String]
        let ePos: [Int]
        let wPos: [Int]
        let wOffsets: [(Int, Int)]
        let tokenCount: Int
    }

    struct Record: Sendable {
        let ids: [Int]
        let questions: [QuestionRec]
        let span: SpanRec?
    }

    func tokens(_ text: String) -> [Int] { tokenizer.encode(text) }

    private func partTokens(_ part: Vela2Part, words: Bool) -> PartTokens {
        if let cached = partCache.lookup(part.text, words) { return cached }
        let scalars = Array(part.text.unicodeScalars)
        let toks = tokenizer.tokens(scalars)
        var p = PartTokens(scalars: scalars, ids: toks.map(\.id), offsets: toks.map { ($0.start, $0.end) })
        if words { (p.words, p.first) = Vela2Text.wordFirstTokens(scalars, offsets: p.offsets) }
        partCache.store(part.text, words, p)
        return p
    }

    private static func optionText(_ name: String, _ description: String) -> String {
        description.isEmpty ? name : "\(name): \(description)"
    }

    func assemble(parts: [Vela2Part], questions: [Vela2Question]) throws -> Record {
        let roles = parts.map(\.role)
        guard Set(roles).count == roles.count else { throw Vela2Error.invalidRequest("one part per role") }
        let spanRole: String? = questions.lazy.compactMap { if case .span(_, _, _, let over) = $0 { over } else { nil } }.first
        var ids = [bos]
        var pending: [(id: String, qPos: Int, optPos: [Int], names: [String], absPos: Int?, over: [String])] = []
        var spanInfo: (id: String, names: [String], ePos: [Int])?
        for q in questions {
            guard case .choice(let id, let text, let options, let over, let abstain) = q else { continue }
            let qPos = ids.count
            ids.append(markers.q)
            ids += tokens(text)
            var optPos: [Int] = []
            for o in options {
                optPos.append(ids.count)
                ids.append(markers.o)
                ids += tokens(Self.optionText(o.name, o.description))
            }
            var absPos: Int?
            if abstain {
                absPos = ids.count
                ids.append(markers.abs)
            }
            pending.append((id, qPos, optPos, options.map(\.name), absPos, over))
        }
        for q in questions {
            guard case .span(let id, _, let labels, _) = q else { continue }
            var ePos: [Int] = []
            for l in labels {
                ePos.append(ids.count)
                ids.append(markers.e)
                ids += tokens(Self.optionText(l.name, l.description))
            }
            spanInfo = (id, labels.map(\.name), ePos)
        }
        ids.append(markers.sep)
        var ranges: [String: (Int, Int, Int)] = [:]
        var spanPart: PartTokens?
        for part in parts {
            guard let seg = markers.seg[part.role] else { throw Vela2Error.invalidRequest("unknown role \(part.role)") }
            let p = partTokens(part, words: part.role == spanRole)
            let start = ids.count
            ids.append(seg)
            ids += p.ids
            ranges[part.role] = (start, ids.count, p.ids.count)
            if part.role == spanRole { spanPart = p }
        }
        ids.append(eos)
        let recs = try pending.map { q -> QuestionRec in
            let ordered = roles.filter { q.over.contains($0) }
            guard let first = ordered.first, let last = ordered.last, let a = ranges[first], let b = ranges[last] else {
                throw Vela2Error.invalidRequest("question \(q.id) is over \(q.over) but the parts are \(roles)")
            }
            return QuestionRec(id: q.id, qPos: q.qPos, optPos: q.optPos, names: q.names, absPos: q.absPos, pool: (a.0, b.1))
        }
        var span: SpanRec?
        if let info = spanInfo, let role = spanRole, let p = spanPart, let range = ranges[role] {
            var wPos: [Int] = []
            var wOff: [(Int, Int)] = []
            for (k, f) in p.first.enumerated() where f < range.2 {
                wPos.append(range.0 + 1 + f)
                wOff.append(p.words[k])
            }
            span = SpanRec(id: info.id, role: role, names: info.names, ePos: info.ePos, wPos: wPos, wOffsets: wOff, tokenCount: p.ids.count)
        } else if spanInfo != nil {
            throw Vela2Error.invalidRequest("span question over a missing part")
        }
        return Record(ids: ids, questions: recs, span: span)
    }

    // MARK: encoder

    private func encode(ids: [Int], model: MLModel, length: Int) async throws -> [Float] {
        let input = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let ip = input.dataPointer.assumingMemoryBound(to: Int32.self)
        let mp = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<length {
            ip[i] = i < ids.count ? Int32(ids[i]) : 0
            mp[i] = i < ids.count ? 1 : 0
        }
        let out = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": input, "attention_mask": mask]))
        guard let hidden = out.featureValue(for: "hidden")?.multiArrayValue else {
            throw Vela2Error.invalidAsset("encoder returned no hidden states")
        }
        let d = heads.hidden
        let strides = hidden.strides.map(\.intValue)
        var h = [Float](repeating: 0, count: ids.count * d)
        let src = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        for t in 0..<ids.count {
            for c in 0..<d { h[t * d + c] = Float(src[t * strides[1] + c * strides[2]]) }
        }
        return h
    }

    // MARK: calibration

    private func choiceAnswer(_ q: QuestionRec, _ lg: [Double]) -> Vela2ChoiceAnswer {
        let n = q.names.count
        let temp = calibration.temperature["choice"] ?? 1
        func softmax(_ x: [Double]) -> [Double] {
            let m = x.max() ?? 0
            let e = x.map { exp($0 - m) }
            let s = e.reduce(0, +)
            return e.map { $0 / s }
        }
        let p = softmax(lg[0..<n].map { $0 / temp })
        var best = 0
        for i in 0..<n where lg[i] > lg[best] { best = i }
        let abstain = q.absPos == nil ? nil : softmax(lg.map { $0 / temp }).last
        return Vela2ChoiceAnswer(id: q.id, names: q.names, probabilities: p, answer: q.names[best], abstainProbability: abstain)
    }

    private func spanAnswer(_ span: SpanRec, logits: [[Float]], parts: [Vela2Part]) -> Vela2SpanAnswer {
        let tSpan = calibration.temperature["span"] ?? 1
        let probs = logits.map { row in row.map { 1 / (1 + exp(-Double($0) / tSpan)) } }
        let text = Array(parts.first { $0.role == span.role }!.text.unicodeScalars)
        let isPII = span.id == "pii" || (!span.names.isEmpty && Set(span.names).isSubset(of: calibration.piiTypes))
        var thr: Double
        if isPII {
            thr = piiThreshold(Double(span.tokenCount))
            if calibration.sparseK >= 0, thr < calibration.sparseThreshold, !span.wOffsets.isEmpty,
                Vela2Text.decodeSpans(probs, offsets: span.wOffsets, text: text, threshold: calibration.probeThreshold).count
                    <= calibration.sparseK
            {
                thr = calibration.sparseThreshold
            }
        } else if span.id == "halu" || span.id == "toxic" {
            thr = calibration.thresholds["span:\(span.id)"] ?? 0.5
        } else {
            thr = calibration.thresholds["span:\(span.id)"] ?? calibration.thresholds["span:*"] ?? 0.5
        }
        let decoded = span.wOffsets.isEmpty ? [] : Vela2Text.decodeSpans(probs, offsets: span.wOffsets, text: text, threshold: thr)
        let out = decoded.map {
            Vela2Span(
                label: span.names[$0.label], start: $0.start, end: $0.end,
                text: String(String.UnicodeScalarView(text[$0.start..<$0.end])), probability: $0.probability)
        }
        let maxP = probs.map { $0.max() ?? 0 }.max() ?? 0
        return Vela2SpanAnswer(id: span.id, role: span.role, spans: out, threshold: thr, maxProbability: maxP)
    }

    /// `pii_length_threshold`: log(threshold) piecewise linear in log(tokens) between anchors, constant outside.
    private func piiThreshold(_ tokens: Double) -> Double {
        let a = calibration.piiAnchors
        let x = log(max(tokens, 1))
        let xs = a.map { log($0.tokens) }
        let ys = a.map { log($0.threshold) }
        if x <= xs[0] { return exp(ys[0]) }
        if x >= xs[xs.count - 1] { return exp(ys[ys.count - 1]) }
        for i in 1..<xs.count where x <= xs[i] {
            return exp(ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]))
        }
        return exp(ys[ys.count - 1])
    }

    final class PartCache: @unchecked Sendable {
        private var entries: [String: PartTokens] = [:]
        private let lock = NSLock()
        func lookup(_ text: String, _ words: Bool) -> PartTokens? {
            lock.lock()
            defer { lock.unlock() }
            return entries[(words ? "w|" : "t|") + text]
        }
        func store(_ text: String, _ words: Bool, _ p: PartTokens) {
            lock.lock()
            defer { lock.unlock() }
            if entries.count > 256 { entries.removeAll() }
            entries[(words ? "w|" : "t|") + text] = p
        }
    }
}

/// The release's readout heads (`modeling_vela2._Vela2Forward.heads`), fp32 on the CPU.
@available(macOS 15.0, iOS 18.0, *)
struct Vela2Heads: Sendable {
    let hidden: Int
    let proj: Int
    let w: [String: [Float]]
    let tau: Float
    let tauSpan: Float

    init(directory: URL) throws {
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("heads.json"))) as? [String: [String: Any]] ?? [:]
        let blob = try Data(contentsOf: directory.appendingPathComponent("heads.bin"))
        var w: [String: [Float]] = [:]
        blob.withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float.self)
            for (name, meta) in index {
                let offset = meta["offset"] as! Int
                let count = (meta["shape"] as! [Int]).reduce(1, *)
                w[name] = Array(f[offset..<(offset + count)])
            }
        }
        guard let n = w["norm.weight"], let p = w["w_o.bias"] else { throw Vela2Error.invalidAsset("heads.bin") }
        self.hidden = n.count
        self.proj = p.count
        self.w = w
        self.tau = exp(w["log_tau"]![0])
        self.tauSpan = exp(w["log_tau_span"]![0])
    }

    private func layerNorm(_ x: ArraySlice<Float>) -> [Float] {
        let n = Float(x.count)
        let mean = x.reduce(0, +) / n
        var v: Float = 0
        for a in x { v += (a - mean) * (a - mean) }
        let inv = 1 / (v / n + 1e-5).squareRoot()
        let g = w["norm.weight"]!, b = w["norm.bias"]!
        return x.enumerated().map { (i, a) in (a - mean) * inv * g[i] + b[i] }
    }

    private func linear(_ name: String, _ x: [Float]) -> [Float] {
        let W = w[name + ".weight"]!, b = w[name + ".bias"]!
        var y = b
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(b.count), Int32(x.count), 1, W, Int32(x.count), x, 1, 1, &y, 1)
        return y
    }

    private func normalize(_ x: [Float]) -> [Float] {
        let n = max(x.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-12)
        return x.map { $0 / n }
    }

    private func row(_ h: [Float], _ t: Int) -> ArraySlice<Float> { h[(t * hidden)..<((t + 1) * hidden)] }

    /// Option logits (all questions, options then [ABS], in order) and span logits [word][label].
    func run(hidden h: [Float], dim: Int, rec: Vela2Manager.Record) -> ([Float], [[Float]]) {
        var logits: [Float] = []
        for q in rec.questions {
            let (ps, pe) = q.pool
            var pooled = [Float](repeating: 0, count: hidden)
            for t in ps..<pe { for c in 0..<hidden { pooled[c] += h[t * hidden + c] } }
            pooled = pooled.map { $0 / Float(max(pe - ps, 1)) }
            let v = normalize(linear("w_p", layerNorm(pooled[...])))
            let uq = linear("w_q", layerNorm(row(h, q.qPos)))
            for pos in q.optPos + (q.absPos.map { [$0] } ?? []) {
                let ho = Array(row(h, pos))
                let u = normalize(zip(linear("w_o", layerNorm(ho[...])), uq).map(+))
                var cosine: Float = 0
                for i in 0..<proj { cosine += u[i] * v[i] }
                let hidden1 = linear("cls_mlp.0", ho).map { max($0, 0) }
                let mlp = linear("cls_mlp.3", hidden1)[0]
                logits.append(cosine / tau + mlp)
            }
        }
        var spans: [[Float]] = []
        if let span = rec.span {
            let e = span.ePos.map { normalize(linear("w_e", layerNorm(row(h, $0)))) }
            for p in span.wPos {
                let t = normalize(linear("w_t", layerNorm(row(h, p))))
                spans.append(e.map { ev in zip(t, ev).reduce(0) { $0 + $1.0 * $1.1 } / tauSpan })
            }
        }
        return (logits, spans)
    }
}
