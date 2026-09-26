@preconcurrency import CoreML
import CoreGraphics
import Foundation

/// Zero-shot image classification with SigLIP 2's Core ML image and text encoders: embed labels once, then score
/// each image against them. Prediction uses Core ML's async API, so callers may keep several images in flight.
public final class SigLIP2Manager: Sendable {
    public let config: SigLIP2Config
    public let tokenizer: SigLIP2Tokenizer

    private let imageModel: MLModel
    private let textModel: MLModel

    public init(config: SigLIP2Config, tokenizer: SigLIP2Tokenizer, imageModel: MLModel, textModel: MLModel) {
        self.config = config
        self.tokenizer = tokenizer
        self.imageModel = imageModel
        self.textModel = textModel
    }

    /// Loads `config.json`, `tokenizer.json`, and the image and text packages (`.mlmodelc` preferred) from
    /// `directory`, as written by the mobius converter.
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws
        -> SigLIP2Manager
    {
        let configURL = directory.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw SigLIP2Error.invalidAsset("Missing config.json in \(directory.path)")
        }
        let config = try JSONDecoder().decode(SigLIP2Config.self, from: configData)
        let tokenizer = try SigLIP2Tokenizer(
            tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"), length: config.textLength)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        async let image = loadModel(named: "\(config.name)-image-\(config.precision)", in: directory, configuration)
        async let text = loadModel(named: "\(config.name)-text-\(config.precision)", in: directory, configuration)
        return try await SigLIP2Manager(config: config, tokenizer: tokenizer, imageModel: image, textModel: text)
    }

    private static func loadModel(
        named name: String, in directory: URL, _ configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiled = directory.appendingPathComponent("\(name).mlmodelc")
        let package = directory.appendingPathComponent("\(name).mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            url = try await MLModel.compileModel(at: package)
        } else {
            throw SigLIP2Error.invalidAsset("Missing \(name).mlmodelc or .mlpackage in \(directory.path)")
        }
        return try await MLModel.load(contentsOf: url, configuration: configuration)
    }

    /// L2-normalized text embedding per label. Compute once per label set and reuse.
    public func embed(labels: [String]) async throws -> [[Float]] {
        var embeddings: [[Float]] = []
        for label in labels {
            let ids = try tokenizer.encode(label)
            let input = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
            let pointer = input.dataPointer.assumingMemoryBound(to: Int32.self)
            for (index, id) in ids.enumerated() { pointer[index] = id }
            let output = try await textModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: input)]))
            embeddings.append(try Self.vector(output, name: "text_embeds"))
        }
        return embeddings
    }

    /// Image embedding plus when the Core ML call began and ended (`DispatchTime` uptime nanoseconds).
    public struct TimedEmbedding: Sendable {
        public let embedding: [Float]
        public let predictionStart: UInt64
        public let predictionEnd: UInt64
    }

    /// L2-normalized image embedding.
    public func embed(image: CGImage) async throws -> [Float] { try await embedTimed(image: image).embedding }

    /// L2-normalized image embedding, with the timing of the model call alone (no decoding or resizing).
    public func embedTimed(image: CGImage) async throws -> TimedEmbedding {
        let pixels = try SigLIP2ImagePreprocessor.pixels(from: image, config: config)
        let size = NSNumber(value: config.imageSize)
        let input = try MLMultiArray(shape: [1, 3, size, size], dataType: .float32)
        pixels.withUnsafeBufferPointer { source in
            input.dataPointer.assumingMemoryBound(to: Float.self).update(from: source.baseAddress!, count: pixels.count)
        }
        let features = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": MLFeatureValue(multiArray: input)])
        let start = DispatchTime.now().uptimeNanoseconds
        let output = try await imageModel.prediction(from: features)
        let end = DispatchTime.now().uptimeNanoseconds
        return TimedEmbedding(
            embedding: try Self.vector(output, name: "image_embeds"), predictionStart: start, predictionEnd: end)
    }

    /// Median wall time of the image encoder alone (no decoding or resizing), one call at a time after `warmup`
    /// calls. Measures the compute units the manager was loaded with.
    public func imageEncoderMilliseconds(iterations: Int = 30, warmup: Int = 5) async throws -> Double {
        let size = NSNumber(value: config.imageSize)
        let input = try MLMultiArray(shape: [1, 3, size, size], dataType: .float32)
        input.dataPointer.initializeMemory(as: Float.self, repeating: 0, count: input.count)
        let features = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": MLFeatureValue(multiArray: input)])
        var times: [Double] = []
        for index in 0..<(warmup + iterations) {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try await imageModel.prediction(from: features)
            if index >= warmup { times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6) }
        }
        return times.sorted()[times.count / 2]
    }

    /// Scores `image` against label embeddings from `embed(labels:)`.
    public func classify(image: CGImage, labels: [String], labelEmbeddings: [[Float]]) async throws -> SigLIP2Answer {
        guard labels.count == labelEmbeddings.count, !labels.isEmpty else {
            throw SigLIP2Error.invalidInput("Expected one embedding per label")
        }
        return score(imageEmbedding: try await embed(image: image), labels: labels, labelEmbeddings: labelEmbeddings)
    }

    public func score(imageEmbedding: [Float], labels: [String], labelEmbeddings: [[Float]]) -> SigLIP2Answer {
        let similarities = labelEmbeddings.map { label in zip(label, imageEmbedding).reduce(0) { $0 + $1.0 * $1.1 } }
        let probabilities = similarities.map { 1 / (1 + exp(-(config.logitScale * $0 + config.logitBias))) }
        return SigLIP2Answer(labels: labels, similarities: similarities, probabilities: probabilities)
    }

    private static func vector(_ output: MLFeatureProvider, name: String) throws -> [Float] {
        guard let array = output.featureValue(for: name)?.multiArrayValue else {
            throw SigLIP2Error.predictionFailed("Missing \(name)")
        }
        return (0..<array.count).map { array[$0].floatValue }
    }
}
