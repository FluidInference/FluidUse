import CoreGraphics
import FluidUse
import Foundation
import ImageIO

/// Sorts photos into breeds with SigLIP 2 on Core ML. Calls are not serialized: several `sort` calls may be in flight.
public final class ImageSorter: Sendable {
    public struct Result: Sendable {
        public let breed: String
        /// SigLIP's own sigmoid probability for the chosen label.
        public let probability: Float
        /// Softmax of the scaled similarities across all labels: the chosen label's share among the candidates.
        public let share: Float
        /// The five most likely breeds with their shares, best first.
        public let top: [(breed: String, share: Float)]
        public let milliseconds: Double
        /// When the Core ML call itself began and ended (`DispatchTime` uptime nanoseconds).
        public let predictionStart: UInt64
        public let predictionEnd: UInt64
    }

    public let breeds: [String]
    private let manager: SigLIP2Manager
    private let embeddings: [[Float]]

    private init(manager: SigLIP2Manager, breeds: [String], embeddings: [[Float]]) {
        self.manager = manager
        self.breeds = breeds
        self.embeddings = embeddings
    }

    /// Loads the encoders from `SIGLIP2_MODEL_DIR` and embeds the breed prompts once.
    public static func load(breeds: [String] = PetsSample.breeds) async throws -> ImageSorter {
        guard let path = ProcessInfo.processInfo.environment["SIGLIP2_MODEL_DIR"], !path.isEmpty else {
            throw SigLIP2Error.invalidAsset("Set SIGLIP2_MODEL_DIR to the converted siglip2-base-patch16-256 folder")
        }
        let manager = try await SigLIP2Manager.load(from: URL(fileURLWithPath: path))
        let embeddings = try await manager.embed(labels: breeds.map(PetsSample.prompt(for:)))
        return ImageSorter(manager: manager, breeds: breeds, embeddings: embeddings)
    }

    public var modelName: String { manager.config.name }

    /// Median time of the image encoder alone on the Neural Engine, one call at a time.
    public func encoderMilliseconds() async throws -> Double { try await manager.imageEncoderMilliseconds() }

    public func sort(_ item: PetItem) async throws -> Result {
        let image = try Self.decode(item.file)
        let start = DispatchTime.now().uptimeNanoseconds
        let timed = try await manager.embedTimed(image: image)
        let answer = manager.score(imageEmbedding: timed.embedding, labels: breeds, labelEmbeddings: embeddings)
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        let scale = manager.config.logitScale
        let best = answer.similarities[answer.selectedIndex]
        let weights = answer.similarities.map { exp(scale * ($0 - best)) }
        let total = weights.reduce(0, +)
        let ranked = weights.indices.sorted { weights[$0] > weights[$1] }.prefix(5).map {
            (breed: breeds[$0], share: weights[$0] / total)
        }
        return Result(
            breed: answer.selectedLabel, probability: answer.probabilities[answer.selectedIndex], share: 1 / total,
            top: ranked, milliseconds: milliseconds, predictionStart: timed.predictionStart,
            predictionEnd: timed.predictionEnd)
    }

    public static func decode(_ file: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw SigLIP2Error.invalidInput("Could not decode \(file.lastPathComponent)")
        }
        return image
    }
}
