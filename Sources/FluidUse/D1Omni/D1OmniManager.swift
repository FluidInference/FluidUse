@preconcurrency import CoreML
import Foundation

/// One d1-omni question in the Decision Index schema.
public enum D1OmniQuestion: Sendable {
    /// Named options; `description` may be empty.
    case choice(String, options: [(name: String, description: String)])
    /// Yes or no; probabilities come back as [yes, no]. `criteria` describes each side.
    case noul(String, criteria: (yes: String, no: String)? = nil)
    /// 2 to 10 ordered levels, lowest first.
    case score(String, levels: [String])

    var typeIndex: Int32 {
        switch self {
        case .choice: 0
        case .score: 1
        case .noul: 2
        }
    }

    var instructions: String {
        switch self {
        case .choice(let text, _), .noul(let text, _), .score(let text, _): text
        }
    }

    /// Option texts in the model's order (a noul is read as [false, true]), as upstream `render_options`.
    var optionTexts: [String] {
        switch self {
        case .choice(_, let options):
            options.map { $0.description.isEmpty ? $0.name : "\($0.name): \($0.description)" }
        case .score(_, let levels):
            levels.enumerated().map { "level \($0.offset): \($0.element)" }
        case .noul(_, let criteria):
            [
                "false: " + (criteria.map(\.no) ?? "no, the statement does not hold"),
                "true: " + (criteria.map(\.yes) ?? "yes, the statement holds"),
            ]
        }
    }

    var temperatureKeys: [String] {
        let type: String
        switch self {
        case .choice: type = "choice"
        case .score: type = "score"
        case .noul: type = "noul"
        }
        let k = optionTexts.count
        let bucket = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
        return ["\(type):\(bucket)", type]
    }
}

public struct D1OmniAnswer: Sendable {
    /// Calibrated probabilities in option order ([yes, no] for a noul).
    public let probabilities: [Float]
    public let selectedIndex: Int
    public let tokenCount: Int
    /// Core ML prediction time only.
    public let predictionMilliseconds: Double
}

public enum D1OmniError: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case tooLong(needed: Int, budget: Int)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid d1-omni asset: \(reason)"
        case .tooLong(let needed, let budget):
            "Prompt needs \(needed) tokens; exceeds the largest token budget (\(budget))"
        case .invalidOutput(let reason): "Invalid d1-omni output: \(reason)"
        }
    }
}

/// Text-only d1-omni-600M (Liquid AI) on Core ML: trunk + decision head in one fixed-shape call per bucket, the
/// answer read off the option markers with zero generated tokens.
///
/// The directory holds `tokenizer.json`, `config.json` and `d1-omni-text.mlpackage` (or its compiled `.mlmodelc`), a
/// multifunction package whose functions `L{tokens}_K{options}_B{batch}` share one copy of the weights; each call
/// uses the smallest bucket that fits. Use `D1OmniModelStore.ensure()` to download it.
@available(macOS 15.0, iOS 18.0, *)
public actor D1OmniManager {
    private struct Bucket {
        let length: Int
        let markers: Int
        let batch: Int
        let model: MLModel
        let inputIds: MLMultiArray
        let attentionMask: MLMultiArray
        let markerPos: MLMultiArray
        let markerMask: MLMultiArray
        let qtype: MLMultiArray
        let features: MLDictionaryFeatureProvider
    }

    private enum Delimiter {
        static let state = "<|reserved_7|>"
        static let question = "<|reserved_8|>"
        static let option = "<|reserved_9|>"
        static let optionEnd = "<|reserved_10|>"
        static let decide = "<|reserved_11|>"
        static let marker = "<|mask|>"
    }

    public let tokenizer: D1OmniTokenizer
    /// Rows per Core ML call; `answer(states:question:)` takes up to this many states.
    public nonisolated let batch: Int
    /// Bucket lengths, shortest first.
    public nonisolated let lengths: [Int]
    private let buckets: [Bucket]
    private let temperatures: [String: Double]
    private let maxLength: Int
    private let ids: (state: Int, question: Int, option: Int, optionEnd: Int, decide: Int, marker: Int)

    private init(
        tokenizer: D1OmniTokenizer, buckets: [Bucket], temperatures: [String: Double], maxLength: Int
    ) throws {
        self.tokenizer = tokenizer
        self.buckets = buckets
        batch = buckets.first?.batch ?? 1
        lengths = buckets.map(\.length)
        self.temperatures = temperatures
        self.maxLength = maxLength
        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.id(for: token) else { throw D1OmniError.invalidAsset("Missing \(token)") }
            return id
        }
        ids = (
            try id(Delimiter.state), try id(Delimiter.question), try id(Delimiter.option),
            try id(Delimiter.optionEnd), try id(Delimiter.decide), try id(Delimiter.marker)
        )
    }

    /// Loads the functions for `batch` rows per call and, if given, exactly `options` marker slots.
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .all, batch: Int = 1, options: Int? = nil
    ) async throws -> D1OmniManager {
        let tokenizer = try D1OmniTokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw D1OmniError.invalidAsset("config.json")
        }
        let temperatures = (config["temperatures"] as? [String: Double]) ?? [:]
        let maxLength = (config["max_length"] as? Int) ?? 16384

        var compiled = directory.appendingPathComponent("d1-omni-text.mlmodelc")
        if !FileManager.default.fileExists(atPath: compiled.path) {
            compiled = try await Self.compiled(directory.appendingPathComponent("d1-omni-text.mlpackage"))
        }
        var buckets: [Bucket] = []
        for name in try await MLModelAsset(url: compiled).functionNames {
            let numbers = name.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            guard name.hasPrefix("L"), numbers.count == 3, numbers[2] == batch,
                options.map({ $0 == numbers[1] }) ?? true
            else { continue }
            let (l, k, b) = (numbers[0], numbers[1], numbers[2])
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            configuration.functionName = name
            let model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
            let rows = NSNumber(value: b)
            let inputIds = try MLMultiArray(shape: [rows, NSNumber(value: l)], dataType: .int32)
            let attentionMask = try MLMultiArray(shape: [rows, NSNumber(value: l)], dataType: .int32)
            let markerPos = try MLMultiArray(shape: [rows, NSNumber(value: k)], dataType: .int32)
            let markerMask = try MLMultiArray(shape: [rows, NSNumber(value: k)], dataType: .int32)
            let qtype = try MLMultiArray(shape: [rows], dataType: .int32)
            let features = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": inputIds, "attention_mask": attentionMask, "marker_pos": markerPos,
                "marker_mask": markerMask, "qtype": qtype,
            ])
            buckets.append(
                Bucket(
                    length: l, markers: k, batch: b, model: model, inputIds: inputIds, attentionMask: attentionMask,
                    markerPos: markerPos, markerMask: markerMask, qtype: qtype, features: features))
        }
        guard !buckets.isEmpty else {
            throw D1OmniError.invalidAsset("No L*_K*_B\(batch) function in \(compiled.lastPathComponent)")
        }
        buckets.sort { ($0.length, $0.markers) < ($1.length, $1.markers) }
        return try D1OmniManager(
            tokenizer: tokenizer, buckets: buckets, temperatures: temperatures, maxLength: maxLength)
    }

    /// Compiles `package` once, next to it; later loads reuse the compiled model.
    static func compiled(_ package: URL) async throws -> URL {
        let destination = package.deletingPathExtension().appendingPathExtension("mlmodelc")
        let manager = FileManager.default
        if manager.fileExists(atPath: destination.path) { return destination }
        let temporary = try await MLModel.compileModel(at: package)
        do {
            try manager.moveItem(at: temporary, to: destination)
            return destination
        } catch {
            return temporary
        }
    }

    /// Token ids and marker positions, as upstream `prompt.encode` (text state, calibrated path).
    public func encode(state: String, question: D1OmniQuestion) throws -> (ids: [Int], markers: [Int]) {
        let options = question.optionTexts
        let budget = max(96, min(options.count * 24 + 32, maxLength / 2))
        let perOption = max(2, (budget - 3 * options.count) / options.count)
        var tokens = [ids.question] + (try tokenizer.encode(try escape(question.instructions)))
        tokens = Array(tokens.prefix(max(16, budget)))
        var markers: [Int] = []
        for text in options {
            markers.append(tokens.count + 1)
            tokens += [ids.option, ids.marker]
            tokens += try tokenizer.encode(try escape(" " + text)).prefix(perOption)
            tokens.append(ids.optionEnd)
        }
        tokens.append(ids.decide)
        let room = max(0, maxLength - tokens.count - 2)
        let stateIds = [ids.state] + (try tokenizer.encode(try escape(state))).prefix(room)
        let sequence = Array(([tokenizer.bosTokenId] + stateIds + tokens).prefix(maxLength))
        return (sequence, markers.map { $0 + 1 + stateIds.count })
    }

    public func answer(state: String, question: D1OmniQuestion) throws -> D1OmniAnswer {
        guard let answer = try answer(states: [state], question: question).first else {
            throw D1OmniError.invalidOutput("empty batch")
        }
        return answer
    }

    /// One Core ML call for up to `batch` states and the same question; the bucket fits the longest prompt.
    /// `predictionMilliseconds` is the shared call time.
    public func answer(states: [String], question: D1OmniQuestion) throws -> [D1OmniAnswer] {
        guard !states.isEmpty, states.count <= batch else {
            throw D1OmniError.invalidOutput("\(states.count) states for a batch of \(batch)")
        }
        let encoded = try states.map { try encode(state: $0, question: question) }
        let longest = encoded.map(\.ids.count).max() ?? 0
        let markerCount = encoded[0].markers.count
        guard let bucket = buckets.first(where: { $0.length >= longest && $0.markers >= markerCount }) else {
            throw D1OmniError.tooLong(needed: longest, budget: buckets.map(\.length).max() ?? 0)
        }
        for row in 0..<bucket.batch {
            let (sequence, markers) = row < encoded.count ? encoded[row] : ([], [])
            fill(bucket.inputIds, row: row, with: sequence)
            fill(bucket.attentionMask, row: row, with: Array(repeating: 1, count: sequence.count))
            fill(bucket.markerPos, row: row, with: markers)
            fill(bucket.markerMask, row: row, with: Array(repeating: 1, count: markers.count))
            bucket.qtype.dataPointer.assumingMemoryBound(to: Int32.self)[row] = question.typeIndex
        }

        let started = DispatchTime.now().uptimeNanoseconds
        let output = try bucket.model.prediction(from: bucket.features)
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw D1OmniError.invalidOutput("missing logits")
        }
        let temperature = question.temperatureKeys.lazy.compactMap { self.temperatures[$0] }.first ?? 1
        return try encoded.enumerated().map { row, item in
            var scores = (0..<item.markers.count).map {
                Double(logits[row * bucket.markers + $0].floatValue) / temperature
            }
            let peak = scores.max() ?? 0
            scores = scores.map { exp($0 - peak) }
            let total = scores.reduce(0, +)
            var probabilities = scores.map { Float($0 / total) }
            if case .noul = question { probabilities.reverse() }
            guard probabilities.allSatisfy(\.isFinite) else { throw D1OmniError.invalidOutput("non-finite logits") }
            let selected = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
            return D1OmniAnswer(
                probabilities: probabilities, selectedIndex: selected, tokenCount: item.ids.count,
                predictionMilliseconds: milliseconds)
        }
    }

    /// `<|name|>` -> `<¦name¦>`, so caller text cannot emit a delimiter or marker token.
    private func escape(_ text: String) throws -> String {
        guard text.contains("<|") else { return text }
        let escaper = try NSRegularExpression(pattern: #"<\|([A-Za-z0-9_]+)\|>"#)
        let range = NSRange(text.startIndex..., in: text)
        return escaper.stringByReplacingMatches(in: text, range: range, withTemplate: "<¦$1¦>")
    }

    private func fill(_ array: MLMultiArray, row: Int, with values: [Int]) {
        let width = array.shape.count > 1 ? array.shape[1].intValue : array.count
        let pointer = array.dataPointer.assumingMemoryBound(to: Int32.self) + row * width
        for index in 0..<width {
            pointer[index] = index < values.count ? Int32(values[index]) : 0
        }
    }
}
