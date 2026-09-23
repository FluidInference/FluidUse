@preconcurrency import CoreML
import Foundation

/// On-device, dynamic-label classification with GLiNER 2.5 base or multilingual.
public actor GLiNER2Manager {
    public static let maximumLength = 128
    public static let maximumOptions = 8

    private let model: MLModel
    private let tokenizer: GLiNER2Tokenizer
    private let inputIds: MLMultiArray
    private let attentionMask: MLMultiArray
    private let markerIndices: MLMultiArray
    private let markerMask: MLMultiArray
    private let features: MLDictionaryFeatureProvider

    public init(model: MLModel, tokenizer: GLiNER2Tokenizer) throws {
        try Self.validate(model.modelDescription)
        self.model = model
        self.tokenizer = tokenizer
        inputIds = try MLMultiArray(shape: [1, 128], dataType: .int32)
        attentionMask = try MLMultiArray(shape: [1, 128], dataType: .int32)
        markerIndices = try MLMultiArray(shape: [1, 8], dataType: .int32)
        markerMask = try MLMultiArray(shape: [1, 8], dataType: .float32)
        features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "marker_indices": MLFeatureValue(multiArray: markerIndices),
            "marker_mask": MLFeatureValue(multiArray: markerMask),
        ])
    }

    /// Download and load the published W8 classification package.
    public static func load(
        variant: GLiNER2Variant, cacheDirectory: URL? = nil, computeUnits: MLComputeUnits = .all,
        progress: GLiNER2ModelStore.Progress? = nil
    ) async throws -> GLiNER2Manager {
        let directory = try await GLiNER2ModelStore.ensure(
            variant: variant, cacheDirectory: cacheDirectory, progress: progress)
        return try await load(from: directory, variant: variant, computeUnits: computeUnits)
    }

    /// Load an already downloaded tokenizer and Core ML package.
    public static func load(
        from directory: URL, variant: GLiNER2Variant, computeUnits: MLComputeUnits = .all
    ) async throws -> GLiNER2Manager {
        let tokenizerURL = directory.appendingPathComponent("tokenizer/tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw GLiNER2Error.invalidAsset("Missing tokenizer/tokenizer.json")
        }
        let tokenizer = try GLiNER2Tokenizer(tokenizerJsonURL: tokenizerURL)
        let package = directory.appendingPathComponent(variant.packageName)
        let compiled = directory.appendingPathComponent(
            variant.packageName.replacingOccurrences(of: ".mlpackage", with: ".mlmodelc"))
        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            modelURL = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            modelURL = try await MLModel.compileModel(at: package)
        } else {
            throw GLiNER2Error.invalidAsset("Missing \(variant.packageName)")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return try GLiNER2Manager(model: model, tokenizer: tokenizer)
    }

    /// Classify `text` against 1–8 labels using the checkpoint's native schema format.
    public func classify(text: String, task: String, labels: [String]) throws -> GLiNER2Answer {
        try Task.checkCancellation()
        guard (1...Self.maximumOptions).contains(labels.count) else {
            throw GLiNER2Error.invalidInput("Expected 1–8 labels; received \(labels.count)")
        }
        guard !task.isEmpty else { throw GLiNER2Error.invalidInput("Task must be nonempty") }
        guard !labels.contains(where: \.isEmpty) else {
            throw GLiNER2Error.invalidInput("Labels must be nonempty")
        }
        let sequence = try tokenizer.classificationSequence(text: text, task: task, labels: labels)
        guard sequence.ids.count <= Self.maximumLength else {
            throw GLiNER2Error.invalidInput("Schema and text require \(sequence.ids.count) tokens; maximum is 128")
        }
        guard sequence.markers.count == labels.count else {
            throw GLiNER2Error.invalidInput("A label marker was lost during tokenization")
        }
        let values = try autoreleasepool { try predict(ids: sequence.ids, markers: sequence.markers) }
        return GLiNER2Answer(
            labels: labels, probabilities: values.probabilities, logits: values.logits, tokenCount: sequence.ids.count)
    }

    /// Exposes the native schema sequence for reference parity checks.
    public nonisolated func tokenSequence(
        text: String, task: String, labels: [String]
    ) throws -> (ids: [Int], markers: [Int]) {
        try tokenizer.classificationSequence(text: text, task: task, labels: labels)
    }

    private func predict(ids: [Int], markers: [Int]) throws -> (logits: [Float], probabilities: [Float]) {
        let idPointer = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let attentionPointer = attentionMask.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<Self.maximumLength {
            idPointer[index] = Int32(index < ids.count ? ids[index] : tokenizer.padTokenId)
            attentionPointer[index] = index < ids.count ? 1 : 0
        }
        let markerPointer = markerIndices.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPointer = markerMask.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<Self.maximumOptions {
            markerPointer[index] = Int32(index < markers.count ? markers[index] : 0)
            maskPointer[index] = index < markers.count ? 1 : 0
        }
        let output = try model.prediction(from: features)
        let logits = try read("logits", from: output)
        let probabilities = try read("probabilities", from: output)
        guard logits.allSatisfy(\.isFinite), probabilities.allSatisfy(\.isFinite) else {
            throw GLiNER2Error.invalidOutput("Model returned non-finite scores")
        }
        return (Array(logits.prefix(markers.count)), Array(probabilities.prefix(markers.count)))
    }

    private func read(_ name: String, from output: MLFeatureProvider) throws -> [Float] {
        guard let array = output.featureValue(for: name)?.multiArrayValue,
            array.shape.map(\.intValue) == [1, Self.maximumOptions], array.dataType == .float32
        else { throw GLiNER2Error.invalidOutput("\(name) must be float32 [1, 8]") }
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        return (0..<Self.maximumOptions).map { pointer[$0] }
    }

    private static func validate(_ description: MLModelDescription) throws {
        let expected: [String: ([Int], MLMultiArrayDataType)] = [
            "input_ids": ([1, 128], .int32),
            "attention_mask": ([1, 128], .int32),
            "marker_indices": ([1, 8], .int32),
            "marker_mask": ([1, 8], .float32),
        ]
        for (name, requirement) in expected {
            guard let constraint = description.inputDescriptionsByName[name]?.multiArrayConstraint,
                constraint.shape.map(\.intValue) == requirement.0, constraint.dataType == requirement.1
            else { throw GLiNER2Error.invalidModel("\(name) has the wrong shape or type") }
        }
    }
}
