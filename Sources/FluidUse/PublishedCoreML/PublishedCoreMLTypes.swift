import Foundation

/// A named option. Its position in the list is the option order the model sees.
public struct DecisionOption: Sendable, Equatable {
    public let name: String
    public let description: String?

    public init(_ name: String, _ description: String? = nil) {
        self.name = name
        self.description = description
    }
}

// MARK: - System One (Kev 0.5B/0.6B, Decision 1.0 Kai/Lex)

/// One typed System One question.
public struct SystemOneQuestion: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        /// Yes/no probability, with optional descriptions of each outcome.
        case noul(falseCriterion: String?, trueCriterion: String?)
        /// One named option.
        case choice([DecisionOption])
        /// Ordered levels; the answer is the expected level index.
        case score([String])
    }

    public let id: String
    public let instructions: String
    public let kind: Kind

    public init(id: String, instructions: String, kind: Kind) {
        self.id = id
        self.instructions = instructions
        self.kind = kind
    }

    public static func noul(
        _ id: String, _ instructions: String, falseCriterion: String? = nil, trueCriterion: String? = nil
    ) -> SystemOneQuestion {
        SystemOneQuestion(
            id: id, instructions: instructions,
            kind: .noul(falseCriterion: falseCriterion, trueCriterion: trueCriterion))
    }

    public static func choice(_ id: String, _ instructions: String, options: [DecisionOption]) -> SystemOneQuestion {
        SystemOneQuestion(id: id, instructions: instructions, kind: .choice(options))
    }

    public static func score(_ id: String, _ instructions: String, levels: [String]) -> SystemOneQuestion {
        SystemOneQuestion(id: id, instructions: instructions, kind: .score(levels))
    }

    var json: PublishedJSON {
        var members: [PublishedJSON.Member] = []
        switch kind {
        case .noul(let falseCriterion, let trueCriterion):
            members = [.init("type", "noul"), .init("instructions", .string(instructions))]
            let criteria = [("false", falseCriterion), ("true", trueCriterion)].compactMap { key, value in
                value.map { PublishedJSON.Member(key, .string($0)) }
            }
            if !criteria.isEmpty { members.append(.init("criteria", .object(criteria))) }
        case .choice(let options):
            members = [
                .init("type", "choice"), .init("instructions", .string(instructions)),
                .init(
                    "criteria",
                    .object(options.map { .init($0.name, $0.description.map(PublishedJSON.string) ?? .null) })),
            ]
        case .score(let levels):
            members = [
                .init("type", "score"), .init("instructions", .string(instructions)),
                .init("criteria", .array(levels.map(PublishedJSON.string))),
            ]
        }
        return .object(members)
    }
}

/// A System One request. Kev packages accept one question per call; Kai and Lex accept up to 128.
public struct SystemOneRequest: Sendable, Equatable {
    /// Text, or a JSON object/array that the runtime renders as labelled text.
    public var state: PublishedJSON
    public var questions: [SystemOneQuestion]
    /// `nil` uses the session model's `systemOneModelName`.
    public var model: String?

    public init(state: PublishedJSON, questions: [SystemOneQuestion], model: String? = nil) {
        self.state = state
        self.questions = questions
        self.model = model
    }

    public init(state: String, questions: [SystemOneQuestion], model: String? = nil) {
        self.init(state: .string(state), questions: questions, model: model)
    }

    func json(defaultModel: String) throws -> PublishedJSON {
        guard !questions.isEmpty else { throw PublishedCoreMLError.invalidRequest("Provide at least one question") }
        guard Set(questions.map(\.id)).count == questions.count else {
            throw PublishedCoreMLError.invalidRequest("Question IDs must be unique")
        }
        return .object([
            .init("model", .string(model ?? defaultModel)),
            .init("state", state),
            .init("questions", .object(questions.map { .init($0.id, $0.json) })),
        ])
    }
}

public struct DecisionProbability: Sendable, Equatable {
    public let key: String
    public let probability: Double
}

/// A typed System One answer. Kev rounds every reported value to two decimals, as upstream does.
public enum SystemOneAnswer: Sendable, Equatable {
    case noul(probabilityTrue: Double)
    /// `probabilities` follow the request's option order.
    case choice(selected: String, confidence: Double, probabilities: [DecisionProbability])
    /// `expectedLevel` is the probability-weighted level index; `probabilities` follow level order.
    case score(expectedLevel: Double, confidence: Double, probabilities: [Double])
}

public struct SystemOneResponse: Sendable, Equatable {
    public let model: String
    /// Answers in request question order.
    public let answers: [(id: String, answer: SystemOneAnswer)]
    public let inputTokens: Int
    public let outputTokens: Int

    public subscript(id: String) -> SystemOneAnswer? { answers.first { $0.id == id }?.answer }

    public static func == (lhs: SystemOneResponse, rhs: SystemOneResponse) -> Bool {
        lhs.model == rhs.model && lhs.inputTokens == rhs.inputTokens && lhs.outputTokens == rhs.outputTokens
            && lhs.answers.map(\.id) == rhs.answers.map(\.id) && lhs.answers.map(\.answer) == rhs.answers.map(\.answer)
    }

    init(json data: Data, request: SystemOneRequest) throws {
        let root = try PublishedCoreMLDecoding.object(data)
        guard let model = root["model"] as? String, let answers = root["answers"] as? [String: Any],
            let usage = root["usage"] as? [String: Any], let inputTokens = usage["input_tokens"] as? Int,
            let outputTokens = usage["output_tokens"] as? Int
        else { throw PublishedCoreMLError.invalidResponse("System One response lacks model, answers, or usage") }
        self.model = model
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.answers = try request.questions.map { question in
            guard let answer = answers[question.id] as? [String: Any] else {
                throw PublishedCoreMLError.invalidResponse("No answer for \(question.id)")
            }
            return (question.id, try Self.answer(answer, for: question))
        }
    }

    private static func answer(_ answer: [String: Any], for question: SystemOneQuestion) throws -> SystemOneAnswer {
        let invalid = PublishedCoreMLError.invalidResponse("Malformed answer for \(question.id)")
        switch question.kind {
        case .noul:
            guard answer["type"] as? String == "noul", let value = answer["noul"] as? Double else { throw invalid }
            return .noul(probabilityTrue: value)
        case .choice(let options):
            guard answer["type"] as? String == "choice", let selected = answer["choice"] as? String,
                let confidence = answer["confidence"] as? Double,
                let distribution = answer["probabilities"] as? [String: Double]
            else { throw invalid }
            let probabilities = try options.map { option in
                guard let value = distribution[option.name] else { throw invalid }
                return DecisionProbability(key: option.name, probability: value)
            }
            return .choice(selected: selected, confidence: confidence, probabilities: probabilities)
        case .score(let levels):
            guard answer["type"] as? String == "score", let score = answer["score"] as? Double,
                let confidence = answer["confidence"] as? Double,
                let distribution = answer["probabilities"] as? [String: Double]
            else { throw invalid }
            let probabilities = try levels.indices.map { index in
                guard let value = distribution[String(index)] else { throw invalid }
                return value
            }
            return .score(expectedLevel: score, confidence: confidence, probabilities: probabilities)
        }
    }
}

// MARK: - Constrained schema (LFM2.5-350M-RLCD)

/// One required field of a closed, flat decision schema.
public struct ConstrainedField: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case oneOf([String])
        case boolean
    }

    public let name: String
    public let kind: Kind
    /// Rendered into the prompt with the schema, as the RLCD runtime does.
    public let description: String?

    public init(name: String, kind: Kind, description: String? = nil) {
        self.name = name
        self.kind = kind
        self.description = description
    }

    public static func oneOf(_ name: String, _ values: [String], description: String? = nil) -> ConstrainedField {
        ConstrainedField(name: name, kind: .oneOf(values), description: description)
    }

    public static func boolean(_ name: String, description: String? = nil) -> ConstrainedField {
        ConstrainedField(name: name, kind: .boolean, description: description)
    }

    /// The runtime puts `json.dumps(schema)` in the prompt, so member order follows RLCD's own schemas.
    static func schema(_ fields: [ConstrainedField]) throws -> PublishedJSON {
        guard !fields.isEmpty, Set(fields.map(\.name)).count == fields.count else {
            throw PublishedCoreMLError.invalidRequest("Provide at least one uniquely named field")
        }
        let properties = fields.map { field -> PublishedJSON.Member in
            var members: [PublishedJSON.Member]
            switch field.kind {
            case .oneOf: members = [.init("type", "string")]
            case .boolean: members = [.init("type", "boolean")]
            }
            if let description = field.description { members.append(.init("description", .string(description))) }
            if case .oneOf(let values) = field.kind {
                members.append(.init("enum", .array(values.map(PublishedJSON.string))))
            }
            return .init(field.name, .object(members))
        }
        return .object([
            .init("type", "object"), .init("properties", .object(properties)),
            .init("required", .array(fields.map { .string($0.name) })), .init("additionalProperties", false),
        ])
    }
}

public enum ConstrainedValue: Sendable, Equatable {
    case string(String)
    case bool(Bool)
}

/// Full-value log-likelihood of one allowed value.
public struct ConstrainedCandidate: Sendable, Equatable {
    public let value: ConstrainedValue
    public let logLikelihood: Double
}

public struct ConstrainedDecision: Sendable, Equatable {
    /// Selected values in field order.
    public let values: [(field: String, value: ConstrainedValue)]
    /// Every scored candidate per field, in field order.
    public let candidates: [(field: String, scores: [ConstrainedCandidate])]
    public let branches: Int
    public let modelCalls: Int

    public subscript(field: String) -> ConstrainedValue? { values.first { $0.field == field }?.value }

    public static func == (lhs: ConstrainedDecision, rhs: ConstrainedDecision) -> Bool {
        lhs.values.map(\.field) == rhs.values.map(\.field) && lhs.values.map(\.value) == rhs.values.map(\.value)
            && lhs.candidates.map(\.field) == rhs.candidates.map(\.field)
            && lhs.candidates.map(\.scores) == rhs.candidates.map(\.scores) && lhs.branches == rhs.branches
            && lhs.modelCalls == rhs.modelCalls
    }

    init(json data: Data, fields: [ConstrainedField]) throws {
        let root = try PublishedCoreMLDecoding.object(data)
        guard let text = root["text"] as? String, let scores = root["scores"] as? [String: Any],
            let branches = root["branches"] as? Int, let modelCalls = root["model_calls"] as? Int
        else { throw PublishedCoreMLError.invalidResponse("Constrained response lacks text, scores, or counts") }
        let selected = try PublishedCoreMLDecoding.object(Data(text.utf8))
        self.branches = branches
        self.modelCalls = modelCalls
        values = try fields.map { field in
            guard let value = Self.value(selected[field.name], kind: field.kind) else {
                throw PublishedCoreMLError.invalidResponse("No valid value for \(field.name)")
            }
            return (field.name, value)
        }
        candidates = try fields.map { field in
            guard let entries = scores[field.name] as? [[String: Any]] else {
                throw PublishedCoreMLError.invalidResponse("No candidate scores for \(field.name)")
            }
            return (
                field.name,
                try entries.map { entry in
                    guard let value = Self.value(entry["value"], kind: field.kind),
                        let score = entry["log_likelihood"] as? Double
                    else { throw PublishedCoreMLError.invalidResponse("Malformed candidate for \(field.name)") }
                    return ConstrainedCandidate(value: value, logLikelihood: score)
                }
            )
        }
    }

    private static func value(_ raw: Any?, kind: ConstrainedField.Kind) -> ConstrainedValue? {
        switch kind {
        case .oneOf(let allowed):
            guard let text = raw as? String, allowed.contains(text) else { return nil }
            return .string(text)
        case .boolean:
            guard let number = raw as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
            return .bool(number.boolValue)
        }
    }
}

// MARK: - Label classification (Jeff)

/// Independent sigmoid label scores; they do not sum to one.
public struct LabelClassification: Sendable, Equatable {
    public let labels: [String]
    public let probabilities: [Double]
    public let selectedIndex: Int
    public var selectedLabel: String { labels[selectedIndex] }

    init(json data: Data, labels: [String]) throws {
        let root = try PublishedCoreMLDecoding.object(data)
        guard root["labels"] as? [String] == labels, let probabilities = root["probabilities"] as? [Double],
            probabilities.count == labels.count, let selected = root["selected_index"] as? Int,
            labels.indices.contains(selected)
        else { throw PublishedCoreMLError.invalidResponse("Malformed classification response") }
        self.labels = labels
        self.probabilities = probabilities
        self.selectedIndex = selected
    }
}

// MARK: - NanoJev

/// One NanoJev question.
public struct NanoJevQuestion: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case choice([DecisionOption])
        case boolean
        case score([String])
    }

    public let id: String
    public let instructions: String
    public let kind: Kind

    public init(id: String, instructions: String, kind: Kind) {
        self.id = id
        self.instructions = instructions
        self.kind = kind
    }

    func request(state: PublishedJSON) -> PublishedJSON {
        var question: [PublishedJSON.Member]
        switch kind {
        case .choice(let options):
            question = [
                .init("type", "choice"), .init("instructions", .string(instructions)),
                .init("criteria", .object(options.map { .init($0.name, .string($0.description ?? $0.name)) })),
            ]
        case .boolean:
            question = [.init("type", "boolean"), .init("instructions", .string(instructions))]
        case .score(let levels):
            question = [
                .init("type", "score"), .init("instructions", .string(instructions)),
                .init("criteria", .array(levels.map(PublishedJSON.string))),
            ]
        }
        let entry: PublishedJSON = .object([
            .init("id", "request"), .init("state", state), .init("questions", .object([.init(id, .object(question))])),
        ])
        return .object([.init("states", .array([entry]))])
    }
}

public struct NanoJevDecision: Sendable, Equatable {
    public let type: String
    public let candidateIDs: [String]
    public let probabilities: [Double]
    public let selectedID: String

    init(json data: Data) throws {
        let root = try PublishedCoreMLDecoding.object(data)
        guard let type = root["type"] as? String, let candidateIDs = root["candidate_ids"] as? [String],
            let probabilities = root["probabilities"] as? [Double], probabilities.count == candidateIDs.count,
            let selectedID = root["selected_id"] as? String, candidateIDs.contains(selectedID)
        else { throw PublishedCoreMLError.invalidResponse("Malformed NanoJev response") }
        self.type = type
        self.candidateIDs = candidateIDs
        self.probabilities = probabilities
        self.selectedID = selectedID
    }
}

enum PublishedCoreMLDecoding {
    static func object(_ data: Data) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PublishedCoreMLError.invalidResponse("Expected a JSON object")
        }
        return object
    }
}

// MARK: - Typed entry points

extension PublishedCoreMLManager {
    /// Answer typed questions with Kev or Decision 1.0 Kai/Lex.
    public func evaluate(_ request: SystemOneRequest) async throws -> SystemOneResponse {
        guard let name = model.systemOneModelName else { throw familyError("System One") }
        let reply = try await evaluate(try request.json(defaultModel: name).encoded())
        return try SystemOneResponse(json: reply, request: request)
    }

    /// Choose one value per field with LFM2.5-350M-RLCD's full-value likelihood scoring.
    public func constrained(context: String, fields: [ConstrainedField]) async throws -> ConstrainedDecision {
        guard model.family == .constrainedSchema else { throw familyError("constrained schema") }
        let request: PublishedJSON = .object([
            .init("context", .string(context)), .init("schema", try ConstrainedField.schema(fields)),
        ])
        return try ConstrainedDecision(json: try await evaluate(try request.encoded()), fields: fields)
    }

    /// Score 1–8 labels for one text with Jeff.
    public func classify(
        text: String, labels: [String], name: String = "", description: String = ""
    ) async throws -> LabelClassification {
        guard model.family == .labelClassification else { throw familyError("label classification") }
        let request: PublishedJSON = .object([
            .init("text", .string(text)), .init("labels", .array(labels.map(PublishedJSON.string))),
            .init("name", .string(name)), .init("description", .string(description)),
        ])
        return try LabelClassification(json: try await evaluate(try request.encoded()), labels: labels)
    }

    /// Answer one question with the local NanoJev conversion.
    public func decide(state: PublishedJSON, question: NanoJevQuestion) async throws -> NanoJevDecision {
        guard model.family == .nanoJev else { throw familyError("NanoJev") }
        return try NanoJevDecision(json: try await evaluate(try question.request(state: state).encoded()))
    }

    private nonisolated func familyError(_ family: String) -> PublishedCoreMLError {
        .invalidRequest("\(model.rawValue) does not serve \(family) requests")
    }
}
