@preconcurrency import CoreML
import Foundation

/// On-device, dynamic-label classification with GLiNER 2.5 small, base, multilingual, or Decide.
public actor GLiNER2Manager {
    public nonisolated let maximumLength: Int
    public nonisolated let maximumOptions: Int

    private nonisolated let markerShape: [Int]
    private nonisolated let model: MLModel
    private nonisolated let tokenizer: GLiNER2Tokenizer
    private let inputIds: MLMultiArray
    private let attentionMask: MLMultiArray
    private let markerIndices: MLMultiArray
    private let markerMask: MLMultiArray
    private let features: MLDictionaryFeatureProvider

    public init(model: MLModel, tokenizer: GLiNER2Tokenizer, variant: GLiNER2Variant = .base) throws {
        maximumLength = variant.maximumLength
        maximumOptions = variant.maximumOptions
        markerShape = variant.markerShape
        try Self.validate(model.modelDescription, length: maximumLength, markerShape: markerShape)
        self.model = model
        self.tokenizer = tokenizer
        let length = NSNumber(value: maximumLength)
        let markers = markerShape.map { NSNumber(value: $0) }
        inputIds = try MLMultiArray(shape: [1, length], dataType: .int32)
        attentionMask = try MLMultiArray(shape: [1, length], dataType: .int32)
        // Zero-filled, so every head after the first stays masked.
        markerIndices = try MLMultiArray(shape: markers, dataType: .int32)
        markerMask = try MLMultiArray(shape: markers, dataType: .float32)
        markerIndices.dataPointer.initializeMemory(as: Int32.self, repeating: 0, count: markerIndices.count)
        markerMask.dataPointer.initializeMemory(as: Float.self, repeating: 0, count: markerMask.count)
        features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "marker_indices": MLFeatureValue(multiArray: markerIndices),
            "marker_mask": MLFeatureValue(multiArray: markerMask),
        ])
    }

    /// Download and load the variant's published classification package.
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
        let tokenizerURL = directory.appendingPathComponent(variant.tokenizerPath)
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw GLiNER2Error.invalidAsset("Missing \(variant.tokenizerPath)")
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
        return try GLiNER2Manager(model: model, tokenizer: tokenizer, variant: variant)
    }

    /// Classify `text` against 1…`maximumOptions` labels using the checkpoint's native schema format.
    public func classify(text: String, task: String, labels: [String]) throws -> GLiNER2Answer {
        try Task.checkCancellation()
        guard (1...maximumOptions).contains(labels.count) else {
            throw GLiNER2Error.invalidInput("Expected 1–\(maximumOptions) labels; received \(labels.count)")
        }
        guard !task.isEmpty else { throw GLiNER2Error.invalidInput("Task must be nonempty") }
        guard !labels.contains(where: \.isEmpty) else {
            throw GLiNER2Error.invalidInput("Labels must be nonempty")
        }
        let sequence = try tokenizer.classificationSequence(text: text, task: task, labels: labels)
        guard sequence.ids.count <= maximumLength else {
            throw GLiNER2Error.invalidInput(
                "Schema and text require \(sequence.ids.count) tokens; maximum is \(maximumLength)")
        }
        guard sequence.markers.count == labels.count else {
            throw GLiNER2Error.invalidInput("A label marker was lost during tokenization")
        }
        let values = try autoreleasepool { try predict(ids: sequence.ids, markers: sequence.markers) }
        return GLiNER2Answer(
            labels: labels, probabilities: values.probabilities, logits: values.logits, tokenCount: sequence.ids.count)
    }

    /// Same result as `classify`, but with per-call inputs and Core ML's async prediction, so several calls can be
    /// in flight on one model at once (the async API is thread-safe; the synchronous one is not).
    public nonisolated func classifyConcurrently(
        text: String, task: String, labels: [String]
    ) async throws -> GLiNER2Answer {
        try await classifyConcurrently(text: text, heads: [(task, labels)])[0]
    }

    /// Answers up to `maximumHeads` classification heads over `text` in one prediction, in the order given.
    public nonisolated func classifyConcurrently(
        text: String, heads: [(task: String, labels: [String])]
    ) async throws -> [GLiNER2Answer] {
        let headCount = markerShape.count == 3 ? markerShape[1] : 1
        guard (1...headCount).contains(heads.count) else {
            throw GLiNER2Error.invalidInput("Expected 1–\(headCount) heads; received \(heads.count)")
        }
        for head in heads {
            guard (1...maximumOptions).contains(head.labels.count), !head.task.isEmpty,
                !head.labels.contains(where: \.isEmpty)
            else {
                throw GLiNER2Error.invalidInput("Each head needs a task and 1–\(maximumOptions) nonempty labels")
            }
        }
        let sequence = try tokenizer.classificationSequence(text: text, heads: heads)
        guard sequence.ids.count <= maximumLength else {
            throw GLiNER2Error.invalidInput(
                "Schema and text require \(sequence.ids.count) tokens; maximum is \(maximumLength)")
        }
        guard zip(sequence.markers, heads).allSatisfy({ $0.count == $1.labels.count }) else {
            throw GLiNER2Error.invalidInput("A label marker was lost during tokenization")
        }
        let length = NSNumber(value: maximumLength)
        let shape = markerShape.map { NSNumber(value: $0) }
        let ids = try MLMultiArray(shape: [1, length], dataType: .int32)
        let attention = try MLMultiArray(shape: [1, length], dataType: .int32)
        let markers = try MLMultiArray(shape: shape, dataType: .int32)
        let mask = try MLMultiArray(shape: shape, dataType: .float32)
        let idPointer = ids.dataPointer.assumingMemoryBound(to: Int32.self)
        let attentionPointer = attention.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<maximumLength {
            idPointer[index] = Int32(index < sequence.ids.count ? sequence.ids[index] : tokenizer.padTokenId)
            attentionPointer[index] = index < sequence.ids.count ? 1 : 0
        }
        markers.dataPointer.initializeMemory(as: Int32.self, repeating: 0, count: markers.count)
        mask.dataPointer.initializeMemory(as: Float.self, repeating: 0, count: mask.count)
        let markerPointer = markers.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPointer = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for (head, positions) in sequence.markers.enumerated() {
            for (slot, position) in positions.enumerated() {
                markerPointer[head * maximumOptions + slot] = Int32(position)
                maskPointer[head * maximumOptions + slot] = 1
            }
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids), "attention_mask": MLFeatureValue(multiArray: attention),
            "marker_indices": MLFeatureValue(multiArray: markers), "marker_mask": MLFeatureValue(multiArray: mask),
        ])
        let output = try await model.prediction(from: features)
        let allLogits = try read("logits", from: output, count: headCount * maximumOptions)
        let allProbabilities = try read("probabilities", from: output, count: headCount * maximumOptions)
        return try heads.enumerated().map { head, request in
            let range = (head * maximumOptions)..<(head * maximumOptions + request.labels.count)
            let logits = Array(allLogits[range])
            let probabilities = Array(allProbabilities[range])
            guard logits.allSatisfy(\.isFinite), probabilities.allSatisfy(\.isFinite) else {
                throw GLiNER2Error.invalidOutput("Model returned non-finite scores")
            }
            return GLiNER2Answer(
                labels: request.labels, probabilities: probabilities, logits: logits,
                tokenCount: sequence.ids.count)
        }
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
        for index in 0..<maximumLength {
            idPointer[index] = Int32(index < ids.count ? ids[index] : tokenizer.padTokenId)
            attentionPointer[index] = index < ids.count ? 1 : 0
        }
        let markerPointer = markerIndices.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPointer = markerMask.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<maximumOptions {
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

    private nonisolated func read(
        _ name: String, from output: MLFeatureProvider, count: Int? = nil
    ) throws -> [Float] {
        guard let array = output.featureValue(for: name)?.multiArrayValue,
            array.shape.map(\.intValue) == markerShape, array.dataType == .float32
        else { throw GLiNER2Error.invalidOutput("\(name) must be float32 \(markerShape)") }
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        return (0..<(count ?? maximumOptions)).map { pointer[$0] }
    }

    private static func validate(_ description: MLModelDescription, length: Int, markerShape: [Int]) throws {
        let expected: [String: ([Int], MLMultiArrayDataType)] = [
            "input_ids": ([1, length], .int32),
            "attention_mask": ([1, length], .int32),
            "marker_indices": (markerShape, .int32),
            "marker_mask": (markerShape, .float32),
        ]
        for (name, requirement) in expected {
            guard let constraint = description.inputDescriptionsByName[name]?.multiArrayConstraint,
                constraint.shape.map(\.intValue) == requirement.0, constraint.dataType == requirement.1
            else { throw GLiNER2Error.invalidModel("\(name) has the wrong shape or type") }
        }
    }
}
