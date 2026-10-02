import Foundation

/// A typed question in Clef's schema (same contract as Jev / SystemOne).
public enum ClefQuestion: Sendable, Equatable {
    /// Yes / no. `criteria` optionally describes the `true` and `false` options.
    case noul(instructions: String?, criteria: [String: String] = [:])
    /// Named options: option id → description. Options are scored in sorted-id order, like Clef.
    case choice(instructions: String?, criteria: [String: String])
    /// Ordered options indexed from 0; the answer is the expected index.
    case score(instructions: String?, criteria: [String])

    var typeName: String {
        switch self {
        case .noul: return "noul"
        case .choice: return "choice"
        case .score: return "score"
        }
    }

    /// 0 = noul, 1 = choice, 2 = score (the head's type embedding index).
    var typeIndex: Int {
        switch self {
        case .noul: return 0
        case .choice: return 1
        case .score: return 2
        }
    }

    var instructions: String? {
        switch self {
        case .noul(let i, _), .choice(let i, _): return i
        case .score(let i, _): return i
        }
    }

    /// (option id, description) in Clef's scoring order.
    func options() -> [(id: String, description: String?)] {
        switch self {
        case .noul(_, let criteria):
            var merged = [
                "true": "The proposition is true or the answer is yes.",
                "false": "The proposition is false or the answer is no.",
            ]
            merged.merge(criteria) { _, new in new }
            return [("true", merged["true"]), ("false", merged["false"])]
        case .choice(_, let criteria):
            return criteria.keys.sorted().map { ($0, criteria[$0]) }
        case .score(_, let criteria):
            return criteria.enumerated().map { (String($0.offset), $0.element) }
        }
    }
}

/// One question's answer: probability per option id in Clef's option order.
public struct ClefAnswer: Sendable, Equatable {
    public let questionID: String
    public let optionIDs: [String]
    public let logits: [Float]
    public let probabilities: [Float]

    /// Highest-probability option id (questions always have at least one option; the encoder rejects empty ones).
    public var choice: String {
        guard let best = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) else { return "" }
        return optionIDs[best]
    }
    /// For `noul` questions: probability of `true`.
    public var noul: Float? { optionIDs.firstIndex(of: "true").map { probabilities[$0] } }
    /// For `score` questions: expected index.
    public var score: Float {
        zip(probabilities.indices, probabilities).reduce(0) { $0 + Float($1.0) * $1.1 }
    }
}

public struct ClefVisionResult: Sendable {
    public let answers: [ClefAnswer]
    public let inputTokens: Int
    public let imageTokens: Int
    public let visionMilliseconds: Double
    public let languageMilliseconds: Double
    public let headMilliseconds: Double
    public var totalMilliseconds: Double { visionMilliseconds + languageMilliseconds + headMilliseconds }
}

public enum ClefVisionError: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case invalidInput(String)
    case tooLong(tokens: Int, bucket: Int)
    case tooManyFields(questions: Int, options: Int)
    case checksumMismatch(String)
    case download(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let m): return "Invalid asset: \(m)"
        case .invalidInput(let m): return "Invalid input: \(m)"
        case .tooLong(let tokens, let bucket): return "Record needs \(tokens) tokens; the largest bucket holds \(bucket)"
        case .tooManyFields(let q, let o): return "Record has \(q) questions / \(o) options; the head holds 16 / 64"
        case .checksumMismatch(let m): return "Checksum mismatch: \(m)"
        case .download(let m): return "Download failed: \(m)"
        }
    }
}

/// Constants from the bundle's `config.json`.
struct ClefVisionConfig: Sendable {
    let imageTokenID: Int
    let visionStartID: Int
    let visionEndID: Int
    let padID: Int
    let vocabSize: Int
    let hiddenSize: Int
    let rotaryDim: Int
    let ropeTheta: Double
    let mropeSection: [Int]
    let lmBuckets: [Int]
    let headLength: Int
    let headMaxQuestions: Int
    let headMaxOptions: Int
    let vision: VisionShape
    let image: ImageProcessing

    struct VisionShape: Sendable {
        let patches: Int  // 784
        let patchSize: Int  // 16
        let mergeSize: Int  // 2
        let temporalPatch: Int  // 2
        let hidden: Int  // 768
        let headDim: Int  // 64
        let depthPositions: Int  // 2304 -> 48 per side
        let ropeTheta: Double
        var patchDim: Int { 3 * temporalPatch * patchSize * patchSize }
        var gridSide: Int { Int(Double(depthPositions).squareRoot()) }
    }

    struct ImageProcessing: Sendable {
        let minPixels: Int
        let maxPixels: Int
        let mean: [Float]
        let std: [Float]
        let rescale: Float
    }

    init(json: [String: Any]) throws {
        func int(_ key: String, in dict: [String: Any]) throws -> Int {
            guard let value = (dict[key] as? NSNumber)?.intValue else { throw ClefVisionError.invalidAsset("config.json: \(key)") }
            return value
        }
        guard let tokenizer = json["tokenizer"] as? [String: Any], let embeddings = json["embeddings"] as? [String: Any],
            let shape = embeddings["shape"] as? [NSNumber], shape.count == 2,
            let packages = json["packages"] as? [String: Any], let lm = packages["lm"] as? [[String: Any]],
            let visionPkg = packages["vision"] as? [String: Any],
            let visionCfg = json["vision_config"] as? [String: Any], let rope = json["text_rope"] as? [String: Any],
            let image = json["image_processor"] as? [String: Any], let mean = image["image_mean"] as? [NSNumber],
            let std = image["image_std"] as? [NSNumber], let section = rope["mrope_section"] as? [NSNumber]
        else { throw ClefVisionError.invalidAsset("config.json is missing sections") }
        // one head package per LM bucket (list), or a single package (object) in older manifests
        let heads: [[String: Any]]
        if let list = packages["head"] as? [[String: Any]] { heads = list } else if let one = packages["head"] as? [String: Any] { heads = [one] } else {
            throw ClefVisionError.invalidAsset("config.json: packages.head")
        }
        guard let head = heads.first else { throw ClefVisionError.invalidAsset("config.json: no head package") }
        imageTokenID = try int("image_token_id", in: json)
        visionStartID = try int("vision_start_token_id", in: json)
        visionEndID = try int("vision_end_token_id", in: json)
        padID = try int("pad_token_id", in: tokenizer)
        vocabSize = shape[0].intValue
        hiddenSize = shape[1].intValue
        let headDim = try int("head_dim", in: json["text_config_summary"] as? [String: Any] ?? ["head_dim": 256])
        let partial = (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        rotaryDim = Int(Double(headDim) * partial)
        ropeTheta = (rope["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000_000
        mropeSection = section.map(\.intValue)
        lmBuckets = try lm.map { try int("length", in: $0) }.sorted()
        headLength = try int("length", in: head)
        headMaxQuestions = try int("max_questions", in: head)
        headMaxOptions = try int("max_options", in: head)
        let visionRope = (visionCfg["rope_parameters"] as? [String: Any])?["rope_theta"] as? NSNumber
        vision = VisionShape(
            patches: try int("patches", in: visionPkg), patchSize: try int("patch_size", in: visionCfg),
            mergeSize: try int("spatial_merge_size", in: visionCfg), temporalPatch: try int("temporal_patch_size", in: visionCfg),
            hidden: try int("hidden_size", in: visionCfg), headDim: try int("hidden_size", in: visionCfg) / (try int("num_heads", in: visionCfg)),
            depthPositions: try int("num_position_embeddings", in: visionCfg), ropeTheta: visionRope?.doubleValue ?? 10_000)
        self.image = ImageProcessing(
            minPixels: try int("min_pixels", in: image), maxPixels: try int("max_pixels", in: image),
            mean: mean.map(\.floatValue), std: std.map(\.floatValue),
            rescale: (image["rescale_factor"] as? NSNumber)?.floatValue ?? 1 / 255)
    }
}
