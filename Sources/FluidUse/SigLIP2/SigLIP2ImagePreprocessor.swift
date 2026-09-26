import CoreGraphics
import Foundation

/// Matches the Hugging Face SigLIP image processor: PIL bilinear resize (antialiased when shrinking) to a square,
/// rescale to [0, 1], normalize per channel. Output is planar float32 `[3, size, size]`.
public enum SigLIP2ImagePreprocessor {
    public static func pixels(from image: CGImage, config: SigLIP2Config) throws -> [Float] {
        let width = image.width
        let height = image.height
        let size = config.imageSize
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard
            let space = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: &rgba, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw SigLIP2Error.invalidInput("Could not decode a \(width)×\(height) image")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        // Horizontal pass, then vertical, each rounded to 8 bits like PIL. Layout stays 4 bytes per pixel.
        let horizontal = resample(
            rgba, lines: height, inLength: width, outLength: size, sampleStride: 4, lineStride: width * 4,
            outSampleStride: 4, outLineStride: size * 4, outCount: height * size * 4)
        let resized = resample(
            horizontal, lines: size, inLength: height, outLength: size, sampleStride: size * 4, lineStride: 4,
            outSampleStride: size * 4, outLineStride: 4, outCount: size * size * 4)
        var planar = [Float](repeating: 0, count: 3 * size * size)
        for channel in 0..<3 {
            let scale = 1 / (255 * config.imageStd[channel])
            let offset = config.imageMean[channel] / config.imageStd[channel]
            let base = channel * size * size
            for pixel in 0..<(size * size) {
                planar[base + pixel] = Float(resized[pixel * 4 + channel]) * scale - offset
            }
        }
        return planar
    }

    /// One separable pass of PIL's `ImagingResample` with the triangle filter, rounding to 8 bits like PIL.
    static func resample(
        _ input: [UInt8], lines: Int, inLength: Int, outLength: Int, sampleStride: Int, lineStride: Int,
        outSampleStride: Int, outLineStride: Int, outCount: Int
    ) -> [UInt8] {
        let scale = Double(inLength) / Double(outLength)
        let filterScale = max(scale, 1)
        var starts = [Int](repeating: 0, count: outLength)
        var counts = [Int](repeating: 0, count: outLength)
        let taps = Int((filterScale * 2).rounded(.up)) + 2
        var weights = [Float](repeating: 0, count: outLength * taps)
        for out in 0..<outLength {
            let center = (Double(out) + 0.5) * scale
            let low = max(Int((center - filterScale + 0.5).rounded(.down)), 0)
            let high = min(Int((center + filterScale + 0.5).rounded(.down)), inLength)
            var row = [Double](repeating: 0, count: high - low)
            for index in low..<high { row[index - low] = max(0, 1 - abs((Double(index) - center + 0.5) / filterScale)) }
            let total = row.reduce(0, +)
            for (offset, weight) in row.enumerated() {
                weights[out * taps + offset] = Float(total > 0 ? weight / total : 0)
            }
            starts[out] = low
            counts[out] = min(high - low, taps)
        }
        var output = [UInt8](repeating: 0, count: outCount)
        input.withUnsafeBufferPointer { source in
            output.withUnsafeMutableBufferPointer { destination in
                weights.withUnsafeBufferPointer { weight in
                    for line in 0..<lines {
                        let lineBase = line * lineStride
                        let outBase = line * outLineStride
                        for out in 0..<outLength {
                            var r: Float = 0
                            var g: Float = 0
                            var b: Float = 0
                            var index = lineBase + starts[out] * sampleStride
                            let weightBase = out * taps
                            for tap in 0..<counts[out] {
                                let w = weight[weightBase + tap]
                                r += w * Float(source[index])
                                g += w * Float(source[index + 1])
                                b += w * Float(source[index + 2])
                                index += sampleStride
                            }
                            let target = outBase + out * outSampleStride
                            destination[target] = UInt8(max(0, min(255, r.rounded())))
                            destination[target + 1] = UInt8(max(0, min(255, g.rounded())))
                            destination[target + 2] = UInt8(max(0, min(255, b.rounded())))
                        }
                    }
                }
            }
        }
        return output
    }
}
