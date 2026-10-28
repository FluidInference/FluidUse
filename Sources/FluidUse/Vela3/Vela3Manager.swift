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
/// types) and claims of an answer its context does not support. One encoder pass per question, on the GPU (the Neural
/// Engine is slower for this encoder at every length and drifts further from the fp32 answers). Rendering, readout,
/// thresholds and the span decoder reproduce the release's `vela3_encoder.py` / `vela3_schema.py`; inputs are limited to
/// 1,024 tokens per pass (the release windows longer span inputs).
@available(macOS 15.0, iOS 18.0, *)
public final class Vela3Manager: Sendable {
    struct Markers: Sendable {
        let q, o, e, sep: Int
        let seg: [String: Int]
    }

    struct SpanSet: Sendable {
        let names: [String]
        let descriptions: [String]
        let role: String
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
            let buckets = setAdapter["pii_buckets"] as? [[String: Any]],
            let byset = setAdapter["thresholds_by_set"] as? [String: Double], let halu = byset["halu"]
        else { throw Vela3Error.invalidAsset("vela3_config.json is missing the span_sets settings") }
        func spanSet(_ name: String) throws -> SpanSet {
            guard let names = sets[name], let schema = schemas[name], let labels = schema["labels"] as? [String: String],
                let over = schema["over"] as? String
            else { throw Vela3Error.invalidAsset("vela3_config.json label set \(name)") }
            return SpanSet(names: names, descriptions: names.map { labels[$0] ?? "" }, role: over)
        }
        let decisionModels = try await models(directory.appendingPathComponent("Vela3Decision.mlpackage"), lengths: decisionLengths)
        let spanModels = try await models(directory.appendingPathComponent("Vela3SpanSets.mlpackage"), lengths: spanLengths)
        return Vela3Manager(
            tokenizer: try Vela3Tokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json")),
            markers: Markers(
                q: try m("[Q]"), o: try m("[O]"), e: try m("[E]"), sep: try m("[SEP_SCHEMA]"),
                seg: ["user": try m("[SEG_user]"), "context": try m("[SEG_context]"), "answer": try m("[SEG_answer]")]),
            bos: bos, eos: eos,
            decisionHeads: try Vela2Heads(weights: try safetensors(directory.appendingPathComponent("heads_decision.safetensors"))),
            spanHeads: try Vela2Heads(weights: try safetensors(directory.appendingPathComponent("heads_span_sets.safetensors"))),
            decisionModels: decisionModels, spanModels: spanModels,
            pii: try spanSet("pii"), unsupported: try spanSet("halu"), spanTemperature: temperature,
            piiBuckets: buckets.compactMap { b in
                guard let t = b["threshold"] as? Double, let n = b["max_tokens"] as? NSNumber else { return nil }
                return (n.intValue, t)
            }, unsupportedThreshold: halu)
    }

    private static func models(_ package: URL, lengths: [Int]) async throws -> [(length: Int, model: MLModel)] {
        var compiled = package.deletingPathExtension().appendingPathExtension("mlmodelc")
        if !FileManager.default.fileExists(atPath: compiled.path) { compiled = try await KevManager.compiled(package) }
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
        guard data.count >= 8 else { throw Vela3Error.invalidAsset(url.lastPathComponent) }
        let headerLength = data.prefix(8).enumerated().reduce(0) { $0 | (Int($1.element) << (8 * $1.offset)) }
        guard 8 + headerLength <= data.count,
            let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + headerLength))) as? [String: Any]
        else { throw Vela3Error.invalidAsset("\(url.lastPathComponent) header") }
        var out: [String: [Float]] = [:]
        for (name, value) in header where name != "__metadata__" {
            guard let meta = value as? [String: Any], meta["dtype"] as? String == "F32", let range = meta["data_offsets"] as? [Int],
                range.count == 2
            else { throw Vela3Error.invalidAsset("\(url.lastPathComponent): \(name) is not an F32 tensor") }
            let bytes = data.subdata(in: (8 + headerLength + range[0])..<(8 + headerLength + range[1]))
            out[name] = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
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
        for (length, model) in (decisionModels + spanModels).reversed() {
            _ = try await encode(ids: [bos, eos], model: model, length: length, dim: decisionHeads.hidden)
        }
    }

    // MARK: decisions (vela3_encoder.DecisionEncoder)

    /// Answer each question about `state` (the release reads a JSON state as compact JSON; pass that text here).
    public func decide(state: String, questions: [Vela3Question]) async throws -> Vela3DecisionResult {
        let started = Date()
        var seen = Set<String>()
        for q in questions {
            guard seen.insert(q.id).inserted else { throw Vela3Error.invalidRequest("duplicate question id \(q.id)") }
            if case .choice(let id, _, let options) = q, options.count < 2 {
                throw Vela3Error.invalidRequest("choice \(id) needs at least 2 options")
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
            self.piiBuckets.first { tokens <= $0.maxTokens }?.threshold ?? self.piiBuckets.last?.threshold ?? 0.5
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

    /// The encoder input of a PII or unsupported-claim call (for parity checks).
    public func spanSequence(pii text: String) -> [Int] { spanRecord(set: pii, parts: [("user", text)]).0.ids }
    public func spanSequence(context: String, answer: String) -> [Int] {
        spanRecord(set: unsupported, parts: [("context", context), ("answer", answer)]).0.ids
    }

    private func spans(
        set: SpanSet, parts: [(role: String, text: String)], threshold: Double?, defaultThreshold: (Int) -> Double
    ) async throws -> Vela3SpanResult {
        let started = Date()
        let (rec, target) = spanRecord(set: set, parts: parts)
        guard let span = rec.span else { throw Vela3Error.invalidRequest("no \(set.role) part") }
        guard let (length, model) = spanModels.first(where: { rec.ids.count <= $0.length }) else {
            throw Vela3Error.tooLong("\(rec.ids.count) tokens exceed the largest encoder function L\(Self.spanLengths.last!)")
        }
        let t = Date()
        let hidden = try await encode(ids: rec.ids, model: model, length: length, dim: spanHeads.hidden)
        let encoderMs = Date().timeIntervalSince(t) * 1000
        let logits = spanHeads.run(hidden: hidden, dim: spanHeads.hidden, rec: rec).1
        let probs = logits.map { row in row.map { 1 / (1 + exp(-Double($0) / spanTemperature)) } }
        let thr = threshold ?? defaultThreshold(span.tokenCount)
        let decoded = span.wOffsets.isEmpty ? [] : Vela2Text.decodeSpans(probs, offsets: span.wOffsets, text: target, threshold: thr)
        let out = decoded.map {
            Vela3Span(
                label: span.names[$0.label], start: $0.start, end: $0.end,
                text: String(String.UnicodeScalarView(target[$0.start..<$0.end])), probability: $0.probability)
        }
        return Vela3SpanResult(
            spans: out, threshold: thr,
            timing: Vela3Timing(encoderMs: encoderMs, totalMs: Date().timeIntervalSince(started) * 1000, buckets: [length]))
    }

    /// `<bos> ([E] label: description)* [SEP_SCHEMA] ([SEG_role] part)* <eos>` (vela3_schema.assemble with one span
    /// question; its text is not part of the input), with the target part's word units for the span readout.
    private func spanRecord(set: SpanSet, parts: [(role: String, text: String)]) -> (Vela2Manager.Record, [Unicode.Scalar]) {
        var ids = [bos]
        var ePos: [Int] = []
        for (name, description) in zip(set.names, set.descriptions) {
            ePos.append(ids.count)
            ids.append(markers.e)
            ids += tokenizer.encode(description.isEmpty ? name : "\(name): \(description)")
        }
        ids.append(markers.sep)
        var span: Vela2Manager.SpanRec?
        var target: [Unicode.Scalar] = []
        for part in parts {
            let scalars = Array(part.text.unicodeScalars)
            let toks = tokenizer.tokens(scalars)
            let start = ids.count
            ids.append(markers.seg[part.role]!)
            ids += toks.map(\.id)
            if part.role == set.role {
                let offsets = toks.map { ($0.start, $0.end) }
                let (words, first) = Vela2Text.wordFirstTokens(scalars, offsets: offsets)
                span = Vela2Manager.SpanRec(
                    id: set.role, role: part.role, names: set.names, ePos: ePos, wPos: first.map { start + 1 + $0 }, wOffsets: words,
                    tokenCount: toks.count)
                target = scalars
            }
        }
        ids.append(eos)
        return (Vela2Manager.Record(ids: ids, questions: [], span: span), target)
    }

    // MARK: encoder

    private func encode(ids: [Int], model: MLModel, length: Int, dim: Int) async throws -> [Float] {
        let input = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let ip = input.dataPointer.assumingMemoryBound(to: Int32.self)
        let mp = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<length {
            ip[i] = i < ids.count ? Int32(ids[i]) : 0
            mp[i] = i < ids.count ? 1 : 0
        }
        let out = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": input, "attention_mask": mask]))
        guard let hidden = out.featureValue(for: "hidden")?.multiArrayValue, hidden.dataType == .float16 else {
            throw Vela3Error.invalidAsset("encoder returned no fp16 hidden states")
        }
        let strides = hidden.strides.map(\.intValue)
        var h = [Float](repeating: 0, count: ids.count * dim)
        let src = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        for t in 0..<ids.count {
            for c in 0..<dim { h[t * dim + c] = Float(src[t * strides[1] + c * strides[2]]) }
        }
        return h
    }
}
