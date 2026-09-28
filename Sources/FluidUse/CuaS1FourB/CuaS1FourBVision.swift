import Accelerate
@preconcurrency import CoreML
import CoreGraphics
import Foundation

/// Host side of the Qwen3.5 vision tower: the `Qwen2VLImageProcessor` preprocessing and the
/// grid-dependent inputs the static Core ML graph takes (see mobius `qwen35_vision.py`).
///
/// 1. `smart_resize` to multiples of 32 px (pixel budget capped at the model's patch budget);
/// 2. bicubic resampling with antialiasing on 8-bit RGB (PIL/torchvision uint8 semantics);
/// 3. patches in spatial-merge-window order, normalized to [-1, 1];
/// 4. learned position table resampled bilinearly (align_corners) to the patch grid;
/// 5. axial 2D rotary tables; padded patches are masked out of attention.
final class CuaS1FourBVision {
    struct ImageFeatures {
        let array: MLMultiArray
        let tokens: Int
        let gridRows: Int
        let gridCols: Int
        let padTokenId: Int
        let hiddenSize: Int

        func copyRow(_ row: Int, to destination: UnsafeMutableRawPointer) {
            let bytes = hiddenSize * 2
            destination.copyMemory(from: array.dataPointer + row * bytes, byteCount: bytes)
        }
    }

    let model: MLModel
    let maxPatches: Int
    let positionTable: [Float]  // [side * side, dim]
    let side: Int
    let dim: Int
    let heads: Int
    let patchSize: Int
    let merge: Int
    let ropeTheta: Double
    let outHidden: Int
    let padTokenId: Int
    let minPixels = 65_536
    let maxPixels = 16_777_216

    private init(
        model: MLModel, maxPatches: Int, positionTable: [Float], side: Int, dim: Int, heads: Int, patchSize: Int,
        merge: Int, ropeTheta: Double, outHidden: Int, padTokenId: Int
    ) {
        self.model = model
        self.maxPatches = maxPatches
        self.positionTable = positionTable
        self.side = side
        self.dim = dim
        self.heads = heads
        self.patchSize = patchSize
        self.merge = merge
        self.ropeTheta = ropeTheta
        self.outHidden = outHidden
        self.padTokenId = padTokenId
    }

    static func load(
        from directory: URL, tokenizer: QwenTokenizer, computeUnits: MLComputeUnits
    ) async throws
        -> CuaS1FourBVision
    {
        let manager = FileManager.default
        let names = (try? manager.contentsOfDirectory(atPath: directory.path)) ?? []
        guard let bundle = names.filter({ $0.hasPrefix("CuaS1Vision_P") }).sorted().first else {
            throw CuaS1FourBError.invalidAsset("no CuaS1Vision_P*.mlmodelc in \(directory.path)")
        }
        let base = directory.appendingPathComponent((bundle as NSString).deletingPathExtension)
        let url = try await CuaS1FourBManager.compiledModel(base, manager: manager)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try await MLModel.load(contentsOf: url, configuration: configuration)
        guard let shape = model.modelDescription.inputDescriptionsByName["patches"]?.multiArrayConstraint?.shape,
            let maxPatches = shape.first?.intValue
        else {
            throw CuaS1FourBError.invalidModel("vision model has no patches input")
        }

        let configData = try Data(contentsOf: directory.appendingPathComponent("vision_config.json"))
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
            let dim = config["hidden_size"] as? Int, let heads = config["num_heads"] as? Int,
            let patch = config["patch_size"] as? Int, let merge = config["spatial_merge_size"] as? Int,
            let positions = config["num_position_embeddings"] as? Int, let out = config["out_hidden_size"] as? Int
        else {
            throw CuaS1FourBError.invalidAsset("bad vision_config.json")
        }
        let theta = ((config["rope_parameters"] as? [String: Any])?["rope_theta"] as? Double) ?? 10_000
        let side = Int(Double(positions).squareRoot())
        let tableData = try Data(contentsOf: directory.appendingPathComponent("pos_embed_table.f16"))
        guard tableData.count == positions * dim * 2 else {
            throw CuaS1FourBError.invalidAsset("pos_embed_table.f16 is \(tableData.count) bytes")
        }
        var table = [Float](repeating: 0, count: positions * dim)
        tableData.withUnsafeBytes { src in
            table.withUnsafeMutableBytes { dst in
                var input = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src.baseAddress), height: 1,
                    width: vImagePixelCount(positions * dim), rowBytes: positions * dim * 2)
                var output = vImage_Buffer(
                    data: dst.baseAddress, height: 1, width: vImagePixelCount(positions * dim),
                    rowBytes: positions * dim * 4)
                vImageConvert_Planar16FtoPlanarF(&input, &output, 0)
            }
        }
        guard let pad = tokenizer.tokenId("<|image_pad|>") else {
            throw CuaS1FourBError.invalidAsset("tokenizer has no <|image_pad|>")
        }
        return CuaS1FourBVision(
            model: model, maxPatches: maxPatches, positionTable: table, side: side, dim: dim, heads: heads,
            patchSize: patch, merge: merge, ropeTheta: theta, outHidden: out, padTokenId: pad)
    }

    /// One prediction on an all-padding input to trigger GPU specialization.
    func prewarm() throws {
        let patchDim = 3 * patchSize * patchSize
        let headDim = dim / heads
        let zeros = { (count: Int) in [Float](repeating: 0, count: count) }
        let inputs: [String: Any] = [
            "patches": try CuaS1FourBManager.half(zeros(maxPatches * patchDim), shape: [maxPatches, patchDim]),
            "pos_embed": try CuaS1FourBManager.half(zeros(maxPatches * dim), shape: [maxPatches, dim]),
            "cos": try CuaS1FourBManager.half(zeros(maxPatches * headDim), shape: [maxPatches, headDim]),
            "sin": try CuaS1FourBManager.half(zeros(maxPatches * headDim), shape: [maxPatches, headDim]),
            "key_mask": try CuaS1FourBManager.half(zeros(maxPatches), shape: [1, maxPatches]),
        ]
        _ = try autoreleasepool { try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs)) }
    }

    func expandImagePads(_ ids: [Int], count: Int) throws -> [Int] {
        guard let index = ids.firstIndex(of: padTokenId), ids.filter({ $0 == padTokenId }).count == 1 else {
            throw CuaS1FourBError.invalidInput("prompt must contain exactly one <|image_pad|>")
        }
        return Array(ids[..<index]) + Array(repeating: padTokenId, count: count) + Array(ids[(index + 1)...])
    }

    /// `smart_resize` with factor = patch * merge; the pixel budget is also capped by the patch budget,
    /// so screenshots larger than the budget are downscaled where the reference would not be.
    func targetSize(height: Int, width: Int) -> (height: Int, width: Int) {
        Self.smartResize(
            height: height, width: width, factor: patchSize * merge, minPixels: minPixels,
            maxPixels: min(maxPixels, maxPatches * patchSize * patchSize))
    }

    /// `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize`.
    static func smartResize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int
    )
        -> (height: Int, width: Int)
    {
        let f = Double(factor)
        let h = Double(height)
        let w = Double(width)
        var hBar = (h / f).rounded(.toNearestOrEven) * f
        var wBar = (w / f).rounded(.toNearestOrEven) * f
        if hBar * wBar > Double(maxPixels) {
            let beta = (h * w / Double(maxPixels)).squareRoot()
            hBar = max(f, (h / beta / f).rounded(.down) * f)
            wBar = max(f, (w / beta / f).rounded(.down) * f)
        } else if hBar * wBar < Double(minPixels) {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = (h * beta / f).rounded(.up) * f
            wBar = (w * beta / f).rounded(.up) * f
        }
        return (Int(hBar), Int(wBar))
    }

    func features(for image: CGImage) throws -> ImageFeatures {
        let rgb = try Self.rgbBytes(image)
        let (height, width) = targetSize(height: image.height, width: image.width)
        let resized = Self.resizeBicubicAA(
            rgb, width: image.width, height: image.height, toWidth: width, toHeight: height)
        let gh = height / patchSize
        let gw = width / patchSize
        let count = gh * gw
        guard count <= maxPatches else {
            throw CuaS1FourBError.invalidInput("\(count) patches exceed the vision budget \(maxPatches)")
        }

        let patchDim = 3 * patchSize * patchSize
        var patches = [Float](repeating: 0, count: maxPatches * patchDim)
        var rows = [Int](repeating: 0, count: count)
        var cols = [Int](repeating: 0, count: count)
        var n = 0
        for bh in 0..<(gh / merge) {
            for bw in 0..<(gw / merge) {
                for mh in 0..<merge {
                    for mw in 0..<merge {
                        let row = bh * merge + mh
                        let col = bw * merge + mw
                        rows[n] = row
                        cols[n] = col
                        let base = n * patchDim
                        for c in 0..<3 {
                            for py in 0..<patchSize {
                                let y = row * patchSize + py
                                for px in 0..<patchSize {
                                    let x = col * patchSize + px
                                    let value = Float(resized[(y * width + x) * 3 + c])
                                    patches[base + (c * patchSize + py) * patchSize + px] = value / 127.5 - 1
                                }
                            }
                        }
                        n += 1
                    }
                }
            }
        }

        var pos = [Float](repeating: 0, count: maxPatches * dim)
        let hTaps = (0..<gh).map { axisTaps(index: $0, size: gh) }
        let wTaps = (0..<gw).map { axisTaps(index: $0, size: gw) }
        for i in 0..<count {
            let (h0, h1, hw) = hTaps[rows[i]]
            let (w0, w1, ww) = wTaps[cols[i]]
            let corners = [
                (h0 * side + w0, (1 - hw) * (1 - ww)), (h0 * side + w1, (1 - hw) * ww),
                (h1 * side + w0, hw * (1 - ww)), (h1 * side + w1, hw * ww),
            ]
            for (index, weight) in corners where weight != 0 {
                let src = index * dim
                let dst = i * dim
                for d in 0..<dim { pos[dst + d] += weight * positionTable[src + d] }
            }
        }

        let headDim = dim / heads
        let quarter = headDim / 4
        let invFreq = (0..<quarter).map { 1.0 / pow(ropeTheta, Double(2 * $0) / Double(headDim / 2)) }
        var cosValues = [Float](repeating: 1, count: maxPatches * headDim)
        var sinValues = [Float](repeating: 0, count: maxPatches * headDim)
        for i in 0..<count {
            for (axis, coordinate) in [rows[i], cols[i]].enumerated() {
                for f in 0..<quarter {
                    let angle = Double(coordinate) * invFreq[f]
                    for copy in 0..<2 {
                        let slot = i * headDim + copy * (headDim / 2) + axis * quarter + f
                        cosValues[slot] = Float(Foundation.cos(angle))
                        sinValues[slot] = Float(Foundation.sin(angle))
                    }
                }
            }
        }
        var mask = [Float](repeating: -1e4, count: maxPatches)
        for i in 0..<count { mask[i] = 0 }

        let inputs: [String: Any] = [
            "patches": try CuaS1FourBManager.half(patches, shape: [maxPatches, patchDim]),
            "pos_embed": try CuaS1FourBManager.half(pos, shape: [maxPatches, dim]),
            "cos": try CuaS1FourBManager.half(cosValues, shape: [maxPatches, headDim]),
            "sin": try CuaS1FourBManager.half(sinValues, shape: [maxPatches, headDim]),
            "key_mask": try CuaS1FourBManager.half(mask, shape: [1, maxPatches]),
        ]
        let output = try autoreleasepool { try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs)) }
        guard var embeds = output.featureValue(for: "image_embeds")?.multiArrayValue else {
            throw CuaS1FourBError.invalidModel("vision model returned no image_embeds")
        }
        if embeds.dataType != .float16 {
            embeds = try CuaS1FourBManager.half(CuaS1FourBManager.floats(embeds), shape: embeds.shape.map(\.intValue))
        }
        return ImageFeatures(
            array: embeds, tokens: count / (merge * merge), gridRows: gh / merge, gridCols: gw / merge,
            padTokenId: padTokenId, hiddenSize: outHidden)
    }

    /// Bilinear, align_corners=True, border padding (HF `_interpolation_axis_taps_weights`).
    private func axisTaps(index: Int, size: Int) -> (Int, Int, Float) {
        let src = Float(index) * Float(side - 1) / Float(max(size - 1, 1))
        let lower = Int(src.rounded(.down))
        return (min(lower, side - 1), min(lower + 1, side - 1), src - Float(lower))
    }

    /// 8-bit RGB, row-major, exactly the stored pixel values (no color management).
    static func rgbBytes(_ image: CGImage) throws -> [UInt8] {
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let space =
            image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpaceCreateDeviceRGB()
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw CuaS1FourBError.invalidInput("could not read screenshot pixels") }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return rgb
    }

    /// PIL-style separable bicubic (a = -0.5) with antialiasing on downscale, horizontal pass then
    /// vertical, rounding to 8 bits between passes -- torchvision's uint8 `resize(antialias=True)`.
    static func resizeBicubicAA(_ src: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        var current = src
        var w = width
        if toWidth != width {
            let coeffs = coefficients(inSize: width, outSize: toWidth)
            var out = [UInt8](repeating: 0, count: toWidth * height * 3)
            for y in 0..<height {
                for x in 0..<toWidth {
                    let (start, weights) = coeffs[x]
                    for c in 0..<3 {
                        var sum: Double = 0
                        for (k, weight) in weights.enumerated() {
                            sum += weight * Double(current[(y * w + start + k) * 3 + c])
                        }
                        out[(y * toWidth + x) * 3 + c] = clamp8(sum)
                    }
                }
            }
            current = out
            w = toWidth
        }
        if toHeight != height {
            let coeffs = coefficients(inSize: height, outSize: toHeight)
            var out = [UInt8](repeating: 0, count: w * toHeight * 3)
            for y in 0..<toHeight {
                let (start, weights) = coeffs[y]
                for x in 0..<w {
                    for c in 0..<3 {
                        var sum: Double = 0
                        for (k, weight) in weights.enumerated() {
                            sum += weight * Double(current[((start + k) * w + x) * 3 + c])
                        }
                        out[(y * w + x) * 3 + c] = clamp8(sum)
                    }
                }
            }
            current = out
        }
        return current
    }

    private static func clamp8(_ value: Double) -> UInt8 {
        UInt8(max(0, min(255, value.rounded(.toNearestOrAwayFromZero))))
    }

    private static func coefficients(inSize: Int, outSize: Int) -> [(Int, [Double])] {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        return (0..<outSize).map { i in
            let center = (Double(i) + 0.5) * scale
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize)
            var weights = (xmin..<xmax).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            return (xmin, weights)
        }
    }

    private static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }
}
