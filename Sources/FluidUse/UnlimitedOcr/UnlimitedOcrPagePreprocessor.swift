import CoreGraphics
import Foundation
import PDFKit

/// One preprocessed page: planar RGB in [-1, 1], 3 x 1024 x 1024.
public struct UnlimitedOcrPage: Sendable {
    public let pixels: [Float]

    public init(pixels: [Float]) {
        self.pixels = pixels
    }
}

/// Upstream's base-mode page view: aspect-fit into a 1024 x 1024 square padded with mid-gray (`ImageOps.pad`, color
/// 127), RGB normalized to [-1, 1], planar [3, 1024, 1024]. PDF pages are rasterized straight to the target
/// size (vector rendering instead of upstream's 300 dpi PNG + bicubic downscale).
public enum UnlimitedOcrPagePreprocessor {
    public static let size = 1024

    public static func page(from image: CGImage) throws -> UnlimitedOcrPage {
        let fit = fitRect(width: Double(image.width), height: Double(image.height))
        return try render { context in
            context.interpolationQuality = .high
            context.draw(image, in: fit)
        }
    }

    public static func page(from page: PDFPage) throws -> UnlimitedOcrPage {
        let box = page.bounds(for: .mediaBox)
        let rotation = ((page.rotation % 360) + 360) % 360
        let quarterTurn = rotation == 90 || rotation == 270
        let fit = fitRect(
            width: quarterTurn ? box.height : box.width, height: quarterTurn ? box.width : box.height)
        return try render { context in
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(fit)
            context.translateBy(x: fit.midX, y: fit.midY)
            context.rotate(by: -CGFloat(rotation) * .pi / 180)
            let scale = quarterTurn ? fit.width / box.height : fit.width / box.width
            context.scaleBy(x: scale, y: scale)
            context.translateBy(x: -box.midX, y: -box.midY)
            if let cgPage = page.pageRef { context.drawPDFPage(cgPage) }
        }
    }

    /// Centered aspect-fit rectangle inside the square, in pixels.
    static func fitRect(width: Double, height: Double) -> CGRect {
        let scale = min(Double(size) / width, Double(size) / height)
        let w = (width * scale).rounded()
        let h = (height * scale).rounded()
        return CGRect(
            x: ((Double(size) - w) / 2).rounded(.down), y: ((Double(size) - h) / 2).rounded(.down), width: w, height: h)
    }

    private static func render(_ draw: (CGContext) -> Void) throws -> UnlimitedOcrPage {
        let n = size
        var rgba = [UInt8](repeating: 127, count: n * n * 4)
        try rgba.withUnsafeMutableBytes { buffer in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: n, height: n, bitsPerComponent: 8, bytesPerRow: n * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { throw UnlimitedOcrError.invalidInput("could not create a \(n)x\(n) bitmap") }
            context.setFillColor(CGColor(gray: 127.0 / 255.0, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: n, height: n))
            draw(context)
        }
        let plane = n * n
        var out = [Float](repeating: 0, count: 3 * plane)
        for i in 0..<plane {
            out[i] = Float(rgba[i * 4]) / 127.5 - 1
            out[plane + i] = Float(rgba[i * 4 + 1]) / 127.5 - 1
            out[2 * plane + i] = Float(rgba[i * 4 + 2]) / 127.5 - 1
        }
        return UnlimitedOcrPage(pixels: out)
    }
}
