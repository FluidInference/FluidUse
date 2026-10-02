import Accelerate
import CoreGraphics
import Foundation

/// Qwen2-VL image preprocessing as transformers' torchvision-backed `Qwen2VLImageProcessor` does it: smart-resize
/// to multiples of patch×merge within the pixel budget, antialiased bicubic resize, rescale and normalize, then
/// patchify into `[n, channel · temporal · patch · patch]` rows in spatial-merge-block order with the single frame
/// repeated along the temporal patch axis.
enum ClefImagePreprocessor {
    struct Patches: Sendable {
        let rows: [Float]  // n × patchDim
        let gridH: Int
        let gridW: Int
        var count: Int { gridH * gridW }
        var mergedTokens: Int { count / 4 }
    }

    /// `smart_resize`: both sides multiples of `factor`, area within [minPixels, maxPixels], aspect ratio kept.
    static func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int) -> (Int, Int) {
        func roundTo(_ v: Double) -> Int { Int((v / Double(factor)).rounded(.toNearestOrEven)) * factor }
        var h = roundTo(Double(height))
        var w = roundTo(Double(width))
        if h * w > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (h, w)
    }

    static func patches(from image: CGImage, config: ClefVisionConfig) throws -> Patches {
        let shape = config.vision
        let factor = shape.patchSize * shape.mergeSize
        let (targetH, targetW) = smartResize(
            height: image.height, width: image.width, factor: factor, minPixels: config.image.minPixels,
            maxPixels: config.image.maxPixels)
        let planar = try resizedPlanar(image, height: targetH, width: targetW)  // [3][targetH][targetW] in 0...255
        let gridH = targetH / shape.patchSize, gridW = targetW / shape.patchSize
        let merge = shape.mergeSize, p = shape.patchSize, t = shape.temporalPatch
        let patchDim = 3 * t * p * p
        var rows = [Float](repeating: 0, count: gridH * gridW * patchDim)
        var scale = [Float](repeating: 0, count: 3), offset = [Float](repeating: 0, count: 3)
        for c in 0..<3 {
            scale[c] = config.image.rescale / config.image.std[c]
            offset[c] = config.image.mean[c] / config.image.std[c]
        }
        // Row order: (block_row, block_col, in_row, in_col); within a row: (channel, temporal, py, px).
        var row = 0
        for blockRow in 0..<(gridH / merge) {
            for blockCol in 0..<(gridW / merge) {
                for inRow in 0..<merge {
                    for inCol in 0..<merge {
                        let gy = blockRow * merge + inRow, gx = blockCol * merge + inCol
                        let base = row * patchDim
                        for c in 0..<3 {
                            for frame in 0..<t {
                                for py in 0..<p {
                                    let y = gy * p + py
                                    let src = c * targetH * targetW + y * targetW + gx * p
                                    let dst = base + ((c * t + frame) * p + py) * p
                                    for px in 0..<p {
                                        rows[dst + px] = planar[src + px] * scale[c] - offset[c]
                                    }
                                }
                            }
                        }
                        row += 1
                    }
                }
            }
        }
        return Patches(rows: rows, gridH: gridH, gridW: gridW)
    }

    /// Decode to RGB and resize with torchvision's antialiased bicubic (Keys a = -0.5, support scaled by the
    /// shrink factor, border clamp, weights normalised), rounding to 8 bits after each separable pass like
    /// torch's uint8 path. Returns planar `[3][h][w]` values in 0...255.
    ///
    /// The image is drawn into a *device* RGB context so CoreGraphics does not colour-manage an embedded ICC
    /// profile: PIL / torchvision read the raw channel values, and a managed decode shifted them by ~5/255.
    static func resizedPlanar(_ image: CGImage, height: Int, width: Int) throws -> [Float] {
        let srcW = image.width, srcH = image.height
        let rgba = try rawRGBA(image)  // non-premultiplied, no colour management: PIL's convert("RGB") drops alpha as is
        var planar = [Float](repeating: 0, count: 3 * srcH * srcW)
        for y in 0..<srcH {
            for x in 0..<srcW {
                for c in 0..<3 { planar[c * srcH * srcW + y * srcW + x] = Float(rgba[(y * srcW + x) * 4 + c]) }
            }
        }
        if srcH == height && srcW == width { return planar }
        func toUInt8(_ v: Float) -> Float { min(255, max(0, (v + 0.5).rounded(.down))) }  // torch: round half up on uint8
        let horizontal = resample(planar, channels: 3, lines: srcH, inLength: srcW, outLength: width, alongWidth: true).map(toUInt8)
        let vertical = resample(horizontal, channels: 3, lines: width, inLength: srcH, outLength: height, alongWidth: false)
        return vertical.map(toUInt8)
    }

    /// Interleaved RGBA8, straight (non-premultiplied) alpha, device RGB (no colour matching of embedded profiles).
    static func rawRGBA(_ image: CGImage) throws -> [UInt8] {
        var format = vImage_CGImageFormat(
            bitsPerComponent: 8, bitsPerPixel: 32, colorSpace: Unmanaged.passRetained(CGColorSpaceCreateDeviceRGB()),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), version: 0, decode: nil,
            renderingIntent: .defaultIntent)
        defer { format.colorSpace.release() }
        var buffer = vImage_Buffer()
        let source = image.copy(colorSpace: CGColorSpaceCreateDeviceRGB()) ?? image  // reinterpret, do not convert
        let status = vImageBuffer_InitWithCGImage(&buffer, &format, nil, source, vImage_Flags(kvImageNoFlags))
        guard status == kvImageNoError else {
            throw ClefVisionError.invalidInput("could not decode a \(image.width)×\(image.height) image (vImage \(status))")
        }
        defer { free(buffer.data) }
        var rgba = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let rowBytes = image.width * 4
        for y in 0..<image.height {
            let src = buffer.data.advanced(by: y * buffer.rowBytes).assumingMemoryBound(to: UInt8.self)
            rgba.withUnsafeMutableBufferPointer { ($0.baseAddress! + y * rowBytes).update(from: src, count: rowBytes) }
        }
        return rgba
    }

    /// One separable bicubic pass. `alongWidth`: input is [c][lines][inLength] → output [c][lines][outLength];
    /// otherwise input is [c][inLength][lines] (rows) → output [c][outLength][lines].
    static func resample(_ input: [Float], channels: Int, lines: Int, inLength: Int, outLength: Int, alongWidth: Bool) -> [Float] {
        let scale = Double(inLength) / Double(outLength)
        let support = 2.0 * max(scale, 1)  // bicubic support 2, widened when shrinking (antialias)
        let invScale = 1 / max(scale, 1)
        func cubic(_ x: Double) -> Double {
            let a = -0.5
            let x = abs(x)
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }
        var taps: [[(Int, Float)]] = []
        taps.reserveCapacity(outLength)
        for out in 0..<outLength {
            let center = (Double(out) + 0.5) * scale
            let low = max(Int((center - support + 0.5).rounded(.down)), 0)
            let high = min(Int((center + support + 0.5).rounded(.down)), inLength)
            var weights: [Double] = []
            for index in low..<high { weights.append(cubic((Double(index) - center + 0.5) * invScale)) }
            let total = weights.reduce(0, +)
            taps.append((low..<high).enumerated().map { ($0.element, Float(total != 0 ? weights[$0.offset] / total : 0)) })
        }
        var output = [Float](repeating: 0, count: channels * lines * outLength)
        for c in 0..<channels {
            for line in 0..<lines {
                for out in 0..<outLength {
                    var acc: Float = 0
                    if alongWidth {
                        let base = c * lines * inLength + line * inLength
                        for (index, weight) in taps[out] { acc += input[base + index] * weight }
                        output[c * lines * outLength + line * outLength + out] = acc
                    } else {
                        let base = c * inLength * lines
                        for (index, weight) in taps[out] { acc += input[base + index * lines + line] * weight }
                        output[c * outLength * lines + out * lines + line] = acc
                    }
                }
            }
        }
        return output
    }
}
