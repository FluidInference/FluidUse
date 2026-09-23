@preconcurrency import CoreML
import Foundation

/// Runs the released Verdict checkpoint with its trained abstention option and calibration.
public actor VerdictManager {
    public static let abstentionID = "__insufficient_evidence__"
    public static let maximumSubstantiveOptions = 24

    public struct Configuration: Sendable {
        public var lengths: [Int]
        public var computeUnits: MLComputeUnits

        public init(lengths: [Int] = [128, 512], computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
            self.lengths = lengths
            self.computeUnits = computeUnits
        }
    }

    private final class Bucket {
        let length: Int
        let model: MLModel
        let inputIDs: MLMultiArray
        let attentionMask: MLMultiArray
        let classMarkerMap: MLMultiArray
        let features: MLDictionaryFeatureProvider

        init(model: MLModel, length: Int) throws {
            self.model = model
            self.length = length
            inputIDs = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            attentionMask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            classMarkerMap = try MLMultiArray(shape: [1, 25, NSNumber(value: length)], dataType: .float32)
            features = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIDs),
                "attention_mask": MLFeatureValue(multiArray: attentionMask),
                "class_marker_map": MLFeatureValue(multiArray: classMarkerMap),
            ])
        }
    }

    private let buckets: [Bucket]
    private let tokenizer: GLiClassTokenizer
    private let defaultTemperature: Double
    private let temperatureByCount: [String: Double]

    /// Load already opened Core ML buckets and the released tokenizer/calibrator.
    public init(models: [MLModel], tokenizer: GLiClassTokenizer, calibratorData: Data) throws {
        guard !models.isEmpty else { throw VerdictError.invalidModel("At least one bucket is required") }
        guard let raw = try JSONSerialization.jsonObject(with: calibratorData) as? [String: Any],
            let temperature = raw["temperature"] as? Double, temperature.isFinite, temperature > 0,
            let byCount = raw["per_k"] as? [String: Double],
            byCount.values.allSatisfy({ $0.isFinite && $0 > 0 })
        else { throw VerdictError.invalidAsset("Invalid calibrator.json") }
        self.tokenizer = tokenizer
        defaultTemperature = temperature
        temperatureByCount = byCount
        buckets = try models.map { model in
            let description = model.modelDescription
            guard let shape = description.inputDescriptionsByName["input_ids"]?.multiArrayConstraint?.shape,
                shape.count == 2, shape[0].intValue == 1, [128, 512].contains(shape[1].intValue)
            else { throw VerdictError.invalidModel("input_ids must be [1, 128] or [1, 512]") }
            let length = shape[1].intValue
            let expected: [String: [Int]] = [
                "attention_mask": [1, length], "class_marker_map": [1, 25, length],
            ]
            for (name, dimensions) in expected {
                guard
                    description.inputDescriptionsByName[name]?.multiArrayConstraint?.shape.map(\.intValue) == dimensions
                else {
                    throw VerdictError.invalidModel("\(name) has the wrong shape")
                }
            }
            return try Bucket(model: model, length: length)
        }.sorted { $0.length < $1.length }
    }

    /// Download the pinned FP16 buckets and load them for on-device decisions.
    public static func load(
        cacheDirectory: URL? = nil, configuration: Configuration = Configuration(),
        progress: VerdictModelStore.Progress? = nil
    ) async throws -> VerdictManager {
        let directory = try await VerdictModelStore.ensure(
            lengths: configuration.lengths, cacheDirectory: cacheDirectory, progress: progress)
        return try await load(from: directory, configuration: configuration)
    }

    /// Load previously downloaded packages without network access.
    public static func load(
        from directory: URL, configuration: Configuration = Configuration()
    ) async throws -> VerdictManager {
        guard directory.isFileURL else { throw VerdictError.invalidAsset("A local directory is required") }
        let tokenizer = try GLiClassTokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let calibrator = try Data(contentsOf: directory.appendingPathComponent("calibrator.json"))
        var models: [MLModel] = []
        for length in configuration.lengths {
            let name = try VerdictModelStore.packageName(length: length)
            let package = directory.appendingPathComponent(name)
            let compiled = package.deletingPathExtension().appendingPathExtension("mlmodelc")
            let url: URL
            if FileManager.default.fileExists(atPath: compiled.path) {
                url = compiled
            } else if FileManager.default.fileExists(atPath: package.path) {
                url = try await compileAndCache(package, at: compiled)
            } else {
                throw VerdictError.invalidAsset("Missing \(name)")
            }
            let modelConfiguration = MLModelConfiguration()
            modelConfiguration.computeUnits = configuration.computeUnits
            models.append(try await MLModel.load(contentsOf: url, configuration: modelConfiguration))
        }
        return try VerdictManager(models: models, tokenizer: tokenizer, calibratorData: calibrator)
    }

    /// Compile once and keep the result beside the package; a read-only directory uses the temporary copy.
    private static func compileAndCache(_ package: URL, at destination: URL) async throws -> URL {
        let temporary = try await MLModel.compileModel(at: package)
        let staged = destination.appendingPathExtension("\(UUID().uuidString).partial")
        do {
            try FileManager.default.moveItem(at: temporary, to: staged)
        } catch {
            return temporary
        }
        guard rename(staged.path, destination.path) == 0 else {
            // Another load installed it first, or the directory is not writable.
            if FileManager.default.fileExists(atPath: destination.path) {
                try? FileManager.default.removeItem(at: staged)
                return destination
            }
            return staged
        }
        return destination
    }

    /// Answer one typed question; overlong inputs are rejected rather than truncated.
    public func answer(context: String, question: VerdictQuestion) throws -> VerdictAnswer {
        try Task.checkCancellation()
        let request = try Self.render(context: context, question: question)
        let ids = tokenizer.encode(request.text)
        guard let bucket = buckets.first(where: { ids.count <= $0.length }) else {
            throw VerdictError.invalidInput(
                "Prompt needs \(ids.count) tokens; largest loaded bucket is \(buckets.last?.length ?? 0)")
        }
        let markers = ids.indices.filter { ids[$0] == tokenizer.classTokenId }
        guard markers.count == request.ids.count else {
            throw VerdictError.invalidInput("Candidate markers do not match the supplied options")
        }
        let logits = try autoreleasepool { try predict(ids: ids, markers: markers, bucket: bucket) }
        let count = request.ids.count
        let scale = temperatureByCount[String(count)] ?? defaultTemperature
        let scaled = logits.prefix(count).map { Double($0) / scale }
        let peak = scaled.max() ?? 0
        let exponentials = scaled.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        guard total.isFinite, total > 0 else { throw VerdictError.invalidOutput("Invalid calibrated probabilities") }
        let probabilities = exponentials.map { $0 / total }
        let selected = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        let abstained = selected == count - 1
        let substantive = probabilities.dropLast().reduce(0, +)
        let score: Double?
        let probabilityTrue: Double?
        switch question {
        case .score:
            score =
                substantive > 0 && !abstained
                ? zip(request.values, probabilities).reduce(0) { $0 + $1.0 * $1.1 / substantive } : nil
            probabilityTrue = nil
        case .noul:
            score = nil
            probabilityTrue = substantive > 0 && !abstained ? probabilities[0] / substantive : nil
        case .choice:
            score = nil
            probabilityTrue = nil
        }
        return VerdictAnswer(
            candidateIDs: request.ids, probabilities: probabilities, logits: Array(logits.prefix(count)),
            tokenCount: ids.count, selectedID: request.ids[selected], isAbstention: abstained,
            score: score, probabilityTrue: probabilityTrue)
    }

    private func predict(ids: [Int], markers: [Int], bucket: Bucket) throws -> [Float] {
        let length = bucket.length
        let idPointer = bucket.inputIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        let attentionPointer = bucket.attentionMask.dataPointer.assumingMemoryBound(to: Int32.self)
        let markerPointer = bucket.classMarkerMap.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<length {
            idPointer[index] = Int32(index < ids.count ? ids[index] : tokenizer.padTokenId)
            attentionPointer[index] = index < ids.count ? 1 : 0
        }
        markerPointer.initialize(repeating: 0, count: 25 * length)
        for (row, position) in markers.enumerated() { markerPointer[row * length + position] = 1 }
        let output = try bucket.model.prediction(from: bucket.features)
        guard let array = output.featureValue(for: "logits")?.multiArrayValue,
            array.count == 25, array.dataType == .float32
        else { throw VerdictError.invalidOutput("logits must contain 25 float32 values") }
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        let values = (0..<25).map { pointer[$0] }
        guard values.allSatisfy(\.isFinite) else { throw VerdictError.invalidOutput("Non-finite logits") }
        return values
    }

    struct Rendered {
        let text: String
        let ids: [String]
        let values: [Double]
    }

    static func render(context: String, question: VerdictQuestion) throws -> Rendered {
        let labels: [String]
        let ids: [String]
        let values: [Double]
        let text: String
        switch question {
        case .choice(let prompt, let options):
            guard (1...maximumSubstantiveOptions).contains(options.count) else {
                throw VerdictError.invalidInput("Choice needs 1–24 options")
            }
            labels = options.map { "It is \($0.description)" }
            ids = options.map(\.id)
            values = []
            text = "Question: \(prompt)\n\nContext:\n\(context)"
        case .score(let prompt, let levels):
            guard (1...maximumSubstantiveOptions).contains(levels.count) else {
                throw VerdictError.invalidInput("Score needs 1–24 levels")
            }
            labels = levels.map { "\($0.description) (Value: \($0.value.rendered))" }
            ids = levels.map(\.id)
            values = levels.map(\.value.doubleValue)
            text = "Question: \(prompt)\n\nContext:\n\(context)"
        case .noul(let proposition):
            labels = ["true: \(proposition)", "false: not \(proposition)"]
            ids = ["true", "false"]
            values = []
            text = "Context:\n\(context)\n\nEvaluate proposition: \(proposition)"
        }
        guard ids.allSatisfy({ !$0.isEmpty }), !ids.contains(abstentionID),
            Set(ids).count == ids.count, values.allSatisfy(\.isFinite)
        else { throw VerdictError.invalidInput("Candidate IDs must be unique, nonempty, and finite") }
        let labelsWithAbstention = labels + ["insufficient evidence"]
        let rendered = labelsWithAbstention.map { "<<LABEL>>\($0)" }.joined() + "<<SEP>>" + text
        return Rendered(text: rendered, ids: ids + [abstentionID], values: values)
    }
}
