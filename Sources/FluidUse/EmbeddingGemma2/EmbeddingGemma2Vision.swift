@preconcurrency import CoreML
import CoreGraphics
import Foundation
import ImageIO

/// EmbeddingGemma 2 image embeddings: the image is resized and cut into 16 px patches the way the Hugging Face
/// `Gemma4ImageProcessor` does, encoded by `EmbeddingGemma2Vision` (ViT + 3x3 pooling, on the GPU, where it beats the
/// Neural Engine at every size), and the pooled tokens are embedded by the text model, so images share the text space.
public final class EmbeddingGemma2Vision: Sendable {
    /// Soft tokens per image. More tokens keep more detail and cost more time (zero-shot Oxford Pets on 370 photos:
    /// 87.0% / 88.4% / 89.5%; M5 Pro GPU: about 15 / 34 / 64+ ms per image).
    public enum Budget: Int, CaseIterable, Sendable {
        case fast = 70
        case balanced = 140
        case detailed = 280

        var patches: Int { rawValue * 9 }
    }

    static let patchSize = 16
    static let pooling = 3
    static let hidden = 768
    static let tableRows = 1024
    public static let maxInFlight = 4

    public let text: EmbeddingGemma2Manager
    private let models: [Budget: MLModel]
    private let positionTable: Data

    init(text: EmbeddingGemma2Manager, models: [Budget: MLModel], positionTable: Data) {
        self.text = text
        self.models = models
        self.positionTable = positionTable
    }

    /// Downloads (once, checksum-verified) and loads the vision model next to an already loaded text model.
    public static func load(
        text: EmbeddingGemma2Manager, computeUnits: MLComputeUnits = .cpuAndGPU,
        progress: EmbeddingGemma2ModelStore.Progress? = nil
    ) async throws -> EmbeddingGemma2Vision {
        guard #available(macOS 15, iOS 18, *) else {
            throw EmbeddingGemma2Error.unsupported("multifunction Core ML models need macOS 15 / iOS 18")
        }
        let directory: URL
        if let path = ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_MODEL_DIR"], !path.isEmpty {
            directory = URL(fileURLWithPath: path)
        } else {
            directory = try await EmbeddingGemma2ModelStore.ensureVision(progress: progress)
        }
        let compiled = directory.appendingPathComponent("EmbeddingGemma2Vision.mlmodelc")
        let package = directory.appendingPathComponent("EmbeddingGemma2Vision.mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            url = try await MLModel.compileModel(at: package)
        } else {
            throw EmbeddingGemma2Error.invalidAsset("Missing EmbeddingGemma2Vision in \(directory.path)")
        }
        guard
            let table = try? Data(
                contentsOf: directory.appendingPathComponent("position_embeddings.f16"), options: .alwaysMapped),
            table.count == 2 * tableRows * hidden * 2
        else { throw EmbeddingGemma2Error.invalidAsset("Missing or malformed position_embeddings.f16") }
        var models: [Budget: MLModel] = [:]
        for budget in Budget.allCases {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            configuration.functionName = "vision_\(budget.rawValue)"
            models[budget] = try await MLModel.load(contentsOf: url, configuration: configuration)
        }
        return EmbeddingGemma2Vision(text: text, models: models, positionTable: table)
    }

    /// Decodes an image file (anything ImageIO reads).
    public static func image(contentsOf url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw EmbeddingGemma2Error.invalidAsset("Cannot decode \(url.lastPathComponent)") }
        return image
    }

    /// Resized size in pixels: the largest aspect-preserving size whose sides are multiples of 48 px and which has at
    /// most `budget.patches` 16 px patches (HF `get_aspect_ratio_preserving_size`).
    static func targetSize(width: Int, height: Int, budget: Budget) -> (width: Int, height: Int) {
        let side = patchSize * pooling
        let factor = (Double(budget.patches * patchSize * patchSize) / Double(width * height)).squareRoot()
        var targetWidth = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        var targetHeight = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        let maxSide = (budget.patches / (pooling * pooling)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min((width / height) * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min((height / width) * side, maxSide)
        }
        return (targetWidth, targetHeight)
    }

    /// Embeddings of many images, in order; up to `maxInFlight` run at once.
    public func embed(images: [CGImage], budget: Budget = .fast) async throws -> [[Float]] {
        try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            var result = [[Float]](repeating: [], count: images.count)
            var next = 0
            func addImage() {
                guard next < images.count else { return }
                let index = next
                let image = images[index]
                next += 1
                group.addTask { (index, try await self.embed(image: image, budget: budget)) }
            }
            for _ in 0..<min(Self.maxInFlight, images.count) { addImage() }
            for try await (index, vector) in group {
                result[index] = vector
                addImage()
            }
            return result
        }
    }

    /// L2-normalized embedding of one image.
    public func embed(image: CGImage, budget: Budget = .fast) async throws -> [Float] {
        guard let model = models[budget] else { throw EmbeddingGemma2Error.predictionFailed("no vision model") }
        let size = Self.targetSize(width: image.width, height: image.height, budget: budget)
        let pixels = try Self.rgba(image, width: size.width, height: size.height)
        let columns = size.width / Self.patchSize
        let rows = size.height / Self.patchSize
        let count = columns * rows
        let total = budget.patches
        let tokens = budget.rawValue
        guard count <= total, columns <= Self.tableRows, rows <= Self.tableRows else {
            throw EmbeddingGemma2Error.predictionFailed("image needs \(count) patches; the budget has \(total)")
        }
        let patches = try MLMultiArray(shape: [1, NSNumber(value: total), 768], dataType: .float16)
        let positions = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .float16)
        let positionEmbeddings = try MLMultiArray(shape: [1, NSNumber(value: total), 768], dataType: .float16)
        let valid = try MLMultiArray(shape: [1, NSNumber(value: total)], dataType: .float16)
        let pool = try MLMultiArray(shape: [NSNumber(value: tokens), NSNumber(value: total)], dataType: .float16)
        let patchPointer = patches.dataPointer.assumingMemoryBound(to: Float16.self)
        let positionPointer = positions.dataPointer.assumingMemoryBound(to: Float16.self)
        let embeddingPointer = positionEmbeddings.dataPointer.assumingMemoryBound(to: Float16.self)
        let validPointer = valid.dataPointer.assumingMemoryBound(to: Float16.self)
        let poolPointer = pool.dataPointer.assumingMemoryBound(to: Float16.self)
        patchPointer.update(repeating: 0, count: total * 768)
        embeddingPointer.update(repeating: 0, count: total * 768)
        positionPointer.update(repeating: -1, count: total * 2)
        validPointer.update(repeating: 0, count: total)
        poolPointer.update(repeating: 0, count: tokens * total)
        positionTable.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self)
            for row in 0..<rows {
                for column in 0..<columns {
                    let patch = row * columns + column
                    // Patch layout: [16 rows][16 columns][RGB], values in [0, 1] (the model maps them to [-1, 1]).
                    let base = patch * 768
                    for y in 0..<Self.patchSize {
                        for x in 0..<Self.patchSize {
                            let pixel = ((row * Self.patchSize + y) * size.width + column * Self.patchSize + x) * 4
                            let offset = base + (y * Self.patchSize + x) * 3
                            for channel in 0..<3 {
                                patchPointer[offset + channel] = Float16(Float(pixels[pixel + channel]) / 255)
                            }
                        }
                    }
                    positionPointer[patch * 2] = Float16(column)
                    positionPointer[patch * 2 + 1] = Float16(row)
                    validPointer[patch] = 1
                    let xRow = column * Self.hidden
                    let yRow = (Self.tableRows + row) * Self.hidden
                    for channel in 0..<Self.hidden {
                        embeddingPointer[base + channel] = Float16(
                            Float(table[xRow + channel]) + Float(table[yRow + channel]))
                    }
                    // 3x3 average pooling over the patch grid, groups numbered row-major.
                    let group = (column / Self.pooling) + (columns / Self.pooling) * (row / Self.pooling)
                    poolPointer[group * total + patch] = Float16(1.0 / 9.0)
                }
            }
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "patches": MLFeatureValue(multiArray: patches), "positions": MLFeatureValue(multiArray: positions),
            "position_embeddings": MLFeatureValue(multiArray: positionEmbeddings),
            "valid": MLFeatureValue(multiArray: valid), "pool": MLFeatureValue(multiArray: pool),
        ])
        let output = try await model.prediction(from: input)
        guard let imageTokens = output.featureValue(for: "image_tokens")?.multiArrayValue else {
            throw EmbeddingGemma2Error.predictionFailed("missing image_tokens output")
        }
        return try await text.embed(imageTokens: imageTokens, count: count / (Self.pooling * Self.pooling))
    }

    /// RGBA8 pixels of `image` drawn at `width` x `height` with high-quality interpolation.
    static func rgba(_ image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw EmbeddingGemma2Error.predictionFailed("cannot draw the image") }
        return pixels
    }
}
