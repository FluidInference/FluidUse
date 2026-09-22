@preconcurrency import CoreML
import Foundation

/// On-device, dynamic-label classification with the 32.7M-parameter GLiClass Edge Apps v2 model.
public actor GLiClassManager {
    public static let maximumOptions = 25

    public struct Configuration: Sendable {
        public var lengths: [Int]
        public var computeUnits: [Int: MLComputeUnits]
        /// Weight representation: `fp16`, `fp16-mask`, `lut8`, `lut6`, or `lut4`.
        public var precision: String

        public init(
            lengths: [Int] = [128], computeUnits: [Int: MLComputeUnits] = [:], precision: String = "fp16"
        ) {
            self.lengths = lengths
            self.computeUnits = computeUnits
            self.precision = precision
        }

        func units(for length: Int) -> MLComputeUnits {
            computeUnits[length] ?? (length <= 128 ? .cpuAndNeuralEngine : .all)
        }
    }

    /// Reusable input storage for one fixed-shape model. `classify` is actor-isolated, so a
    /// prediction always finishes before the next call mutates these buffers.
    private final class Bucket {
        let length: Int
        let model: MLModel
        let usesAttentionBias: Bool
        let inputIds: MLMultiArray
        let classMarkerMap: MLMultiArray
        let attention: MLMultiArray
        let features: MLDictionaryFeatureProvider

        init(length: Int, model: MLModel, usesAttentionBias: Bool) throws {
            self.length = length
            self.model = model
            self.usesAttentionBias = usesAttentionBias
            inputIds = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            classMarkerMap = try MLMultiArray(
                shape: [1, NSNumber(value: GLiClassManager.maximumOptions), NSNumber(value: length)],
                dataType: .float32)
            if usesAttentionBias {
                attention = try MLMultiArray(
                    shape: [1, 1, 1, NSNumber(value: length)], dataType: .float32)
            } else {
                attention = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            }
            features = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "class_marker_map": MLFeatureValue(multiArray: classMarkerMap),
                usesAttentionBias ? "attention_bias" : "attention_mask": MLFeatureValue(multiArray: attention),
            ])
        }
    }

    private let buckets: [Bucket]
    private let tokenizer: GLiClassTokenizer
    private var sequenceCache: [String: [Int]] = [:]
    public nonisolated let lengths: [Int]

    public init(models: [MLModel], tokenizer: GLiClassTokenizer) throws {
        guard !models.isEmpty else { throw GLiClassError.invalidModel("At least one bucket model is required") }
        var buckets: [Bucket] = []
        for model in models {
            let length = try Self.modelLength(model.modelDescription)
            let usesAttentionBias = try Self.validate(model.modelDescription, length: length)
            buckets.append(try Bucket(length: length, model: model, usesAttentionBias: usesAttentionBias))
        }
        self.buckets = buckets.sorted { $0.length < $1.length }
        self.lengths = self.buckets.map(\.length)
        self.tokenizer = tokenizer
    }

    /// Load `.mlmodelc`/`.mlpackage` buckets and `tokenizer.json` from a local directory.
    public static func load(
        from directory: URL, configuration: Configuration = Configuration()
    ) async throws
        -> GLiClassManager
    {
        guard directory.isFileURL else { throw GLiClassError.invalidAsset("A local directory URL is required") }
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw GLiClassError.invalidAsset("Missing tokenizer.json in \(directory.path)")
        }
        let tokenizer = try GLiClassTokenizer(tokenizerJsonURL: tokenizerURL)
        var models: [MLModel] = []
        for length in configuration.lengths {
            let name = try modelName(length: length, precision: configuration.precision)
            let compiled = directory.appendingPathComponent(name).appendingPathExtension("mlmodelc")
            let package = directory.appendingPathComponent(name).appendingPathExtension("mlpackage")
            let modelURL: URL
            if FileManager.default.fileExists(atPath: compiled.path) {
                modelURL = compiled
            } else if FileManager.default.fileExists(atPath: package.path) {
                modelURL = try await MLModel.compileModel(at: package)
            } else {
                throw GLiClassError.invalidAsset("Missing \(compiled.lastPathComponent) in \(directory.path)")
            }
            let modelConfiguration = MLModelConfiguration()
            modelConfiguration.computeUnits = configuration.units(for: length)
            models.append(try await MLModel.load(contentsOf: modelURL, configuration: modelConfiguration))
        }
        return try GLiClassManager(models: models, tokenizer: tokenizer)
    }

    static func modelName(length: Int, precision: String) throws -> String {
        let representation: String
        switch precision {
        case "fp16": representation = "fp16"
        case "fp16-mask": representation = "float_mask_fp16"
        case "lut8", "lut6", "lut4": representation = "\(precision)_kmeans_per_tensor"
        default: throw GLiClassError.invalidAsset("Unknown GLiClass precision \(precision)")
        }
        return "gliclass_edge_apps_\(representation)_L\(length)_options\(maximumOptions)"
    }

    /// Score all labels in one encoder call. `prompt` describes the classification task.
    public func classify(text: String, labels: [String], prompt: String? = nil) throws -> GLiClassAnswer {
        try Task.checkCancellation()
        guard (2...Self.maximumOptions).contains(labels.count) else {
            throw GLiClassError.invalidOptionCount(labels.count)
        }
        for (index, label) in labels.enumerated() where label.isEmpty {
            throw GLiClassError.emptyOption(index)
        }
        let rendered = labels.map { "<<LABEL>>\($0)" }.joined() + "<<SEP>>" + (prompt ?? "") + text
        let untruncated: [Int]
        if let cached = sequenceCache[rendered] {
            untruncated = cached
        } else {
            untruncated = tokenizer.encodeClassification(text: text, labels: labels, prompt: prompt)
            if sequenceCache.count >= 4096 { sequenceCache.removeAll(keepingCapacity: true) }
            sequenceCache[rendered] = untruncated
        }
        guard let bucket = buckets.first(where: { untruncated.count <= $0.length }) ?? buckets.last else {
            throw GLiClassError.invalidModel("No bucket loaded")
        }
        let ids = Array(untruncated.prefix(bucket.length))
        let markers = ids.indices.filter { ids[$0] == tokenizer.classTokenId }
        guard markers.count == labels.count else {
            throw GLiClassError.promptTooLong(optionCount: labels.count, maximumLength: bucket.length)
        }
        let output = try autoreleasepool { try predict(ids: ids, markers: markers, bucket: bucket) }
        return GLiClassAnswer(
            labels: labels, probabilities: output.probabilities, logits: output.logits, tokenCount: ids.count,
            bucketLength: bucket.length, textWasTruncated: ids.count < untruncated.count)
    }

    /// Exact input ids and class-marker positions, exposed for reference-tokenizer parity tests.
    public nonisolated func tokenSequence(
        text: String, labels: [String], prompt: String? = nil
    ) -> (ids: [Int], markers: [Int]) {
        let ids = tokenizer.encodeClassification(text: text, labels: labels, prompt: prompt)
        return (ids, ids.indices.filter { ids[$0] == tokenizer.classTokenId })
    }

    private func predict(ids: [Int], markers: [Int], bucket: Bucket) throws -> (logits: [Float], probabilities: [Float])
    {
        let length = bucket.length
        let idPointer = bucket.inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<length {
            idPointer[index] = Int32(index < ids.count ? ids[index] : tokenizer.padTokenId)
        }
        let markerPointer = bucket.classMarkerMap.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<(Self.maximumOptions * length) { markerPointer[index] = 0 }
        for (row, position) in markers.enumerated() { markerPointer[row * length + position] = 1 }
        if bucket.usesAttentionBias {
            let pointer = bucket.attention.dataPointer.assumingMemoryBound(to: Float.self)
            for index in 0..<length { pointer[index] = index < ids.count ? 0 : -10_000 }
        } else {
            let pointer = bucket.attention.dataPointer.assumingMemoryBound(to: Int32.self)
            for index in 0..<length { pointer[index] = index < ids.count ? 1 : 0 }
        }
        let prediction = try bucket.model.prediction(from: bucket.features)
        let logits = try read("logits", output: prediction)
        let probabilities = try read("probabilities", output: prediction)
        guard logits.allSatisfy(\.isFinite), probabilities.allSatisfy(\.isFinite) else {
            throw GLiClassError.invalidOutput("Model returned non-finite values")
        }
        return (Array(logits.prefix(markers.count)), Array(probabilities.prefix(markers.count)))
    }

    private func read(_ name: String, output: MLFeatureProvider) throws -> [Float] {
        guard let array = output.featureValue(for: name)?.multiArrayValue,
            array.count == Self.maximumOptions, array.dataType == .float32
        else {
            throw GLiClassError.invalidOutput("\(name) must be float32 with \(Self.maximumOptions) values")
        }
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        return (0..<array.count).map { pointer[$0] }
    }

    private static func modelLength(_ description: MLModelDescription) throws -> Int {
        guard let shape = description.inputDescriptionsByName["input_ids"]?.multiArrayConstraint?.shape,
            shape.count == 2, shape[0].intValue == 1
        else { throw GLiClassError.invalidModel("input_ids must have shape [1, length]") }
        return shape[1].intValue
    }

    private static func validate(_ description: MLModelDescription, length: Int) throws -> Bool {
        let expected: [String: ([Int], MLMultiArrayDataType)] = [
            "input_ids": ([1, length], .int32),
            "class_marker_map": ([1, maximumOptions, length], .float32),
        ]
        for (name, requirement) in expected {
            guard let constraint = description.inputDescriptionsByName[name]?.multiArrayConstraint,
                constraint.shape.map(\.intValue) == requirement.0, constraint.dataType == requirement.1
            else { throw GLiClassError.invalidModel("\(name) has the wrong shape or type") }
        }
        if let constraint = description.inputDescriptionsByName["attention_bias"]?.multiArrayConstraint {
            guard constraint.shape.map(\.intValue) == [1, 1, 1, length], constraint.dataType == .float32 else {
                throw GLiClassError.invalidModel("attention_bias has the wrong shape or type")
            }
            return true
        }
        guard let constraint = description.inputDescriptionsByName["attention_mask"]?.multiArrayConstraint,
            constraint.shape.map(\.intValue) == [1, length], constraint.dataType == .int32
        else { throw GLiClassError.invalidModel("attention_mask has the wrong shape or type") }
        return false
    }
}
