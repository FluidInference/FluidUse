@preconcurrency import CoreML
import Foundation

/// On-device, dynamic-label classification with the 32.7M-parameter GLiClass Edge Apps v2 model.
public actor GLiClassManager {
    public static let maximumOptions = 25

    public struct Configuration: Sendable {
        public var lengths: [Int]
        public var computeUnits: [Int: MLComputeUnits]

        public init(lengths: [Int] = [128], computeUnits: [Int: MLComputeUnits] = [:]) {
            self.lengths = lengths
            self.computeUnits = computeUnits
        }

        func units(for length: Int) -> MLComputeUnits {
            computeUnits[length] ?? (length <= 128 ? .cpuAndNeuralEngine : .all)
        }
    }

    private struct Bucket: Sendable {
        let length: Int
        let model: MLModel
    }

    private let buckets: [Bucket]
    private let tokenizer: GLiClassTokenizer
    public nonisolated let lengths: [Int]

    public init(models: [MLModel], tokenizer: GLiClassTokenizer) throws {
        guard !models.isEmpty else { throw GLiClassError.invalidModel("At least one bucket model is required") }
        var buckets: [Bucket] = []
        for model in models {
            let length = try Self.modelLength(model.modelDescription)
            try Self.validate(model.modelDescription, length: length)
            buckets.append(Bucket(length: length, model: model))
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
            let name = "gliclass_edge_apps_fp16_L\(length)_options\(maximumOptions)"
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
        let untruncated = tokenizer.encode(rendered)
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
        let rendered = labels.map { "<<LABEL>>\($0)" }.joined() + "<<SEP>>" + (prompt ?? "") + text
        let ids = tokenizer.encode(rendered)
        return (ids, ids.indices.filter { ids[$0] == tokenizer.classTokenId })
    }

    private func predict(ids: [Int], markers: [Int], bucket: Bucket) throws -> (logits: [Float], probabilities: [Float])
    {
        let length = bucket.length
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let attention = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let markerMap = try MLMultiArray(
            shape: [1, NSNumber(value: Self.maximumOptions), NSNumber(value: length)], dataType: .float32)
        let idPointer = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPointer = attention.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<length {
            idPointer[index] = Int32(index < ids.count ? ids[index] : tokenizer.padTokenId)
            maskPointer[index] = index < ids.count ? 1 : 0
        }
        let markerPointer = markerMap.dataPointer.assumingMemoryBound(to: Float.self)
        markerPointer.initialize(repeating: 0, count: Self.maximumOptions * length)
        for (row, position) in markers.enumerated() { markerPointer[row * length + position] = 1 }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIds, "attention_mask": attention, "class_marker_map": markerMap,
        ])
        let prediction = try bucket.model.prediction(from: features)
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

    private static func validate(_ description: MLModelDescription, length: Int) throws {
        let expected: [String: ([Int], MLMultiArrayDataType)] = [
            "input_ids": ([1, length], .int32),
            "attention_mask": ([1, length], .int32),
            "class_marker_map": ([1, maximumOptions, length], .float32),
        ]
        for (name, requirement) in expected {
            guard let constraint = description.inputDescriptionsByName[name]?.multiArrayConstraint,
                constraint.shape.map(\.intValue) == requirement.0, constraint.dataType == requirement.1
            else { throw GLiClassError.invalidModel("\(name) has the wrong shape or type") }
        }
    }
}
