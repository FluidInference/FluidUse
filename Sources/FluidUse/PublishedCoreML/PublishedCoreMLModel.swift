import Foundation

/// Published sub-1B Core ML runtimes whose preprocessing is provided by their conversion toolkits.
public enum PublishedCoreMLModel: String, CaseIterable, Sendable {
    case kev05 = "kev-0-5b"
    case kev06 = "kev-0.6b"
    case kai = "decision-1.0-kai"
    case lex = "decision-1.0-lex"
    case lfm350 = "lfm2-5-350m-rlcd"
    case jeff
    case nanojev

    /// Request/answer contract served by the model's published runtime.
    public enum Family: Sendable {
        /// Typed `noul`/`choice`/`score` questions (Kev, Decision 1.0 Kai/Lex).
        case systemOne
        /// Closed flat JSON schema with enum and boolean fields (LFM2.5-350M-RLCD).
        case constrainedSchema
        /// Independent sigmoid label scores (Jeff).
        case labelClassification
        /// One `choice`/`boolean`/`score` question in NanoJev's native format.
        case nanoJev
    }

    public var repository: String { "FluidInference/\(rawValue)-coreml" }

    public var family: Family {
        switch self {
        case .kev05, .kev06, .kai, .lex: .systemOne
        case .lfm350: .constrainedSchema
        case .jeff: .labelClassification
        case .nanojev: .nanoJev
        }
    }

    /// Precisions accepted by the published runtime; the first is the default.
    public var precisions: [String] {
        switch self {
        case .kev05: ["fp16", "e8"]
        case .kev06, .kai, .lex, .jeff: ["fp16", "w8"]
        case .lfm350, .nanojev: ["fp16"]
        }
    }

    /// The `model` value a System One request must carry. Kai and Lex reject any other name.
    public var systemOneModelName: String? {
        switch self {
        case .kev05: "kev-0.5b"
        case .kev06: "kev-0.6b"
        case .kai: "Decision-1.0-Kai"
        case .lex: "Decision-1.0-Lex"
        case .lfm350, .jeff, .nanojev: nil
        }
    }

    /// Whether `PublishedCoreMLModelStore` can download it. NanoJev's converted weights stay local.
    public var isDownloadable: Bool { self != .nanojev }

    /// Directory, relative to the repository root, holding the pinned `pyproject.toml` and `uv.lock`.
    var projectDirectory: String? {
        switch self {
        case .kev05, .lfm350, .jeff: ""
        case .kev06: "source"
        case .kai, .lex: "conversion"
        case .nanojev: nil
        }
    }

    /// Metadata file whose presence identifies a materialized repository root.
    var rootMarker: String { self == .nanojev ? "assets.lock.json" : "config.json" }

    func requiredPackages(precision: String) throws -> [String] {
        guard precisions.contains(precision) else { throw PublishedCoreMLError.invalidPrecision(precision) }
        switch self {
        case .kev05:
            return ["kev_0_5b_\(precision)_L128_options32.mlpackage"]
        case .kev06:
            return ["kev_0_6b_\(precision)_L128_options32.mlpackage"]
        case .kai, .lex:
            // Lex publishes no compressed Choice package, so W8 keeps its FP16 Choice path.
            let compressed = precision == "w8"
            return ["choice", "noul", "score"].map { kind in
                let useW8 = compressed && (self == .kai || kind != "choice")
                return "coreml/\(kind)\(useW8 ? "-embedding-w8" : "").mlpackage"
            }
        case .lfm350:
            return ["lfm350_rlcd_fp16_L256_B8_V16.mlpackage"]
        case .jeff:
            return [precision == "w8" ? "JeffDecision-L128-W8.mlpackage" : "JeffDecision-L128-FP16.mlpackage"]
        case .nanojev:
            return ["build/nanojev_encoder_fp16_L128_K4.mlpackage", "build/nanojev_heads_fp16_K4.mlpackage"]
        }
    }
}

public enum PublishedCoreMLError: Error, LocalizedError, Sendable, Equatable {
    case invalidPrecision(String)
    case missingAsset(String)
    case invalidRequest(String)
    case invalidResponse(String)
    case runtime(String)
    case timedOut(String)
    case closed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidPrecision(let value): "Unsupported published Core ML precision: \(value)"
        case .missingAsset(let value): "Missing published Core ML asset: \(value)"
        case .invalidRequest(let value): "Invalid published Core ML request: \(value)"
        case .invalidResponse(let value): "Invalid published Core ML response: \(value)"
        case .runtime(let value): "Published Core ML runtime failed: \(value)"
        case .timedOut(let value): "Published Core ML worker timed out: \(value)"
        case .closed(let value): "Published Core ML session is closed: \(value)"
        }
    }
}
