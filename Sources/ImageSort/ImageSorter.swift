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
        public let milliseconds: Double
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

    public func sort(_ item: PetItem) async throws -> Result {
        let image = try Self.decode(item.file)
        let start = DispatchTime.now().uptimeNanoseconds
        let answer = try await manager.classify(image: image, labels: breeds, labelEmbeddings: embeddings)
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        return Result(
            breed: answer.selectedLabel, probability: answer.probabilities[answer.selectedIndex],
            milliseconds: milliseconds)
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
