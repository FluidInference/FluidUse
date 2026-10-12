import CoreGraphics
import Foundation

/// A System 1 question kind, as JEV's `POST /v1/decide`.
public enum JevKind: Sendable, Equatable {
    /// Yes / no: probabilities for `["false", "true"]`.
    case noul
    /// Ordered 0...5: probabilities for `["0", ..., "5"]`.
    case score
    /// Pick one of 2...256 options, in the given order.
    case choice([String])

    var name: String {
        switch self {
        case .noul: return "noul"
        case .score: return "score"
        case .choice: return "choice"
        }
    }
}

/// One question about the shared state.
public struct JevQuestion: Sendable, Equatable {
    public let kind: JevKind
    public let text: String

    public init(_ kind: JevKind, _ text: String) {
        self.kind = kind
        self.text = text
    }
}

/// The state: text and images, in order (JEV's `state` list of strings and `{"image": ...}` parts).
public enum JevStatePart: Sendable {
    case text(String)
    case image(CGImage)
}

/// Calibrated probabilities for one question.
public struct JevDecision: Sendable, Equatable {
    public let options: [String]
    public let probabilities: [Float]

    /// Most likely option.
    public var choice: String {
        guard let best = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) else { return "" }
        return options[best]
    }
    /// P(true) for a `noul` question.
    public var yes: Float? { options.firstIndex(of: "true").map { probabilities[$0] } }
    /// Expected value for a `score` question.
    public var expectedScore: Float { zip(probabilities.indices, probabilities).reduce(0) { $0 + Float($1.0) * $1.1 } }
}

public struct JevResult: Sendable {
    public let decisions: [JevDecision]
    /// Prompt tokens of each question's row (the image is shared).
    public let inputTokens: [Int]
    public let visionMilliseconds: Double
    public let decoderMilliseconds: Double
    public let totalMilliseconds: Double
}

public enum JevError: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case invalidInput(String)
    case tooLong(tokens: Int, bucket: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let m): return "Invalid asset: \(m)"
        case .invalidInput(let m): return "Invalid input: \(m)"
        case .tooLong(let tokens, let bucket): return "Prompt needs \(tokens) tokens; the loaded bucket holds \(bucket)"
        }
    }
}

/// The bundle's `config.json` and `head.json`.
struct JevConfig: Sendable {
    let hidden: Int
    let rotaryDim: Int
    let ropeTheta: Double
    let mropeSection: [Int]
    let padID: Int
    let vocab: Int
    let buckets: [Int]
    let imageTokenID: Int
    let visionStartID: Int
    let visionEndID: Int
    let parts: [String]
    let visionPackages: [(file: String, patches: Int)]
    let posEmbed: String
    let headRows: String
    let headCount: Int
    /// Image preprocessing and vision-tower host math are shared with clef-vision; `patches` is the largest bucket
    /// (`clef(patches:)` gives the config for another one).
    let clef: ClefVisionConfig

    // head.json
    let headIDs: [Int]
    let bias: [Float]
    let ranges: [String: (Int, Int)]
    let temperatures: [String: Float]
    let labels: [String]

    init(manifest json: [String: Any], head: [String: Any]) throws {
        func int(_ key: String, _ dict: [String: Any] = [:]) throws -> Int {
            guard let v = ((dict.isEmpty ? json : dict)[key] as? NSNumber)?.intValue else {
                throw JevError.invalidAsset("config.json: \(key)")
            }
            return v
        }
        guard let buckets = json["buckets"] as? [NSNumber], let parts = json["parts"] as? [String],
            let theta = (json["rope_theta"] as? NSNumber)?.doubleValue,
            let section = json["mrope_section"] as? [NSNumber], let vision = json["vision"] as? [String: Any],
            let packages = vision["packages"] as? [[String: Any]], let posEmbed = vision["pos_embed"] as? String,
            let image = json["image_processor"] as? [String: Any], let mean = image["image_mean"] as? [NSNumber],
            let std = image["image_std"] as? [NSNumber], let headInfo = json["head"] as? [String: Any],
            let rows = headInfo["rows"] as? String
        else { throw JevError.invalidAsset("config.json is missing sections") }
        hidden = try int("hidden_size")
        rotaryDim = try int("rotary_dim")
        ropeTheta = theta
        mropeSection = section.map(\.intValue)
        padID = try int("pad_id")
        vocab = try int("vocab_size")
        self.buckets = buckets.map(\.intValue).sorted()
        imageTokenID = try int("image_token_id")
        visionStartID = try int("vision_start_token_id")
        visionEndID = try int("vision_end_token_id")
        self.parts = parts
        visionPackages = try packages.map {
            guard let file = $0["file"] as? String else { throw JevError.invalidAsset("config.json: vision file") }
            return (file, try int("patches", $0))
        }.sorted { $0.patches < $1.patches }
        self.posEmbed = posEmbed
        headRows = rows
        headCount = try int("count", headInfo)
        let headDim = try int("hidden_size", vision) / (try int("num_heads", vision))
        clef = ClefVisionConfig(
            imageTokenID: imageTokenID, visionStartID: visionStartID, visionEndID: visionEndID, padID: padID,
            vocabSize: vocab, hiddenSize: hidden, rotaryDim: rotaryDim, ropeTheta: theta,
            mropeSection: mropeSection, lmBuckets: self.buckets, headLength: 0, headMaxQuestions: 0,
            headMaxOptions: 0,
            vision: .init(
                patches: visionPackages.last?.patches ?? 0, patchSize: try int("patch_size", vision),
                mergeSize: try int("spatial_merge_size", vision), temporalPatch: try int("temporal_patch_size", vision),
                hidden: try int("hidden_size", vision), headDim: headDim,
                depthPositions: try int("num_position_embeddings", vision), ropeTheta: 10_000),
            image: .init(
                minPixels: try int("min_pixels", image), maxPixels: try int("max_pixels", image),
                mean: mean.map(\.floatValue), std: std.map(\.floatValue),
                rescale: (image["rescale_factor"] as? NSNumber)?.floatValue ?? 1 / 255))

        guard let ids = head["ids"] as? [NSNumber], let bias = head["bias"] as? [NSNumber],
            let ranges = head["ranges"] as? [String: [NSNumber]],
            let temps = head["temperatures"] as? [String: NSNumber],
            let labels = head["labels"] as? [String]
        else { throw JevError.invalidAsset("head.json is missing sections") }
        headIDs = ids.map(\.intValue)
        self.bias = bias.map(\.floatValue)
        self.ranges = ranges.mapValues { ($0[0].intValue, $0[1].intValue) }
        temperatures = temps.mapValues(\.floatValue)
        self.labels = labels
        guard headIDs.count == headCount else { throw JevError.invalidAsset("head.json ids vs head_rows count") }
    }

    /// Rows of the head table and per-option bias for one question (`serve_decide.s1_pass`): noul and score read
    /// their verbalizer slots; choice reads the trained A...P slots, then the remaining option labels with no bias.
    func decisionRows(_ kind: JevKind) throws -> (rows: [Int], bias: [Float]) {
        guard let (lo, hi) = ranges[kind.name] else { throw JevError.invalidAsset("head.json: no \(kind.name) range") }
        guard case .choice(let options) = kind else {
            return (Array(lo..<hi), Array(bias[lo..<hi]))
        }
        let trained = hi - lo
        let rows = (0..<options.count).map { $0 < trained ? lo + $0 : bias.count + ($0 - trained) }
        let optionBias = (0..<options.count).map { $0 < trained ? bias[lo + $0] : 0 }
        return (rows, optionBias)
    }
}

extension JevConfig {
    func clef(patches: Int) -> ClefVisionConfig {
        let v = clef.vision
        return ClefVisionConfig(
            imageTokenID: clef.imageTokenID, visionStartID: clef.visionStartID, visionEndID: clef.visionEndID,
            padID: clef.padID, vocabSize: clef.vocabSize, hiddenSize: clef.hiddenSize, rotaryDim: clef.rotaryDim,
            ropeTheta: clef.ropeTheta, mropeSection: clef.mropeSection, lmBuckets: clef.lmBuckets, headLength: 0,
            headMaxQuestions: 0, headMaxOptions: 0,
            vision: .init(
                patches: patches, patchSize: v.patchSize, mergeSize: v.mergeSize, temporalPatch: v.temporalPatch,
                hidden: v.hidden, headDim: v.headDim, depthPositions: v.depthPositions, ropeTheta: v.ropeTheta),
            image: clef.image)
    }
}

extension ClefVisionConfig {
    init(
        imageTokenID: Int, visionStartID: Int, visionEndID: Int, padID: Int, vocabSize: Int, hiddenSize: Int,
        rotaryDim: Int, ropeTheta: Double, mropeSection: [Int], lmBuckets: [Int], headLength: Int,
        headMaxQuestions: Int, headMaxOptions: Int, vision: VisionShape, image: ImageProcessing
    ) {
        self.imageTokenID = imageTokenID
        self.visionStartID = visionStartID
        self.visionEndID = visionEndID
        self.padID = padID
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.rotaryDim = rotaryDim
        self.ropeTheta = ropeTheta
        self.mropeSection = mropeSection
        self.lmBuckets = lmBuckets
        self.headLength = headLength
        self.headMaxQuestions = headMaxQuestions
        self.headMaxOptions = headMaxOptions
        self.vision = vision
        self.image = image
    }
}
