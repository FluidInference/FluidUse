import CoreGraphics
import Foundation

/// A bar chart made of photos: one row per breed, each sorted photo drawn as a tile at the end of its row.
/// Tiles go into one bitmap, so thousands of them cost one image on screen. `levels` tiles stack in each row;
/// the column count follows so every row holds `capacity` photos.
@MainActor
final class PhotoChart {
    static let tile = 16
    static let rowGap = 6
    static let capacity = 260

    let rows: Int
    let levels: Int
    let columns: Int
    let width: Int
    let height: Int
    var rowHeight: Int { Self.tile * levels + Self.rowGap }
    private let context: CGContext

    init(rows: Int, levels: Int, tint: (Int) -> CGColor) {
        self.rows = rows
        self.levels = levels
        columns = (Self.capacity + levels - 1) / levels
        width = columns * Self.tile
        height = rows * (Self.tile * levels + Self.rowGap)
        context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        for row in 0..<rows {
            context.setFillColor(tint(row))
            context.fill(rect(top: row * rowHeight, x: 0, width: width, height: Self.tile * levels))
        }
    }

    /// Stack depth whose bitmap fills a `size` area best with `rows` rows.
    static func bestLevels(rows: Int, size: CGSize) -> Int {
        guard size.width > 0, size.height > 0 else { return 2 }
        return (1...10).max { scale(rows: rows, levels: $0, size: size) < scale(rows: rows, levels: $1, size: size) }
            ?? 2
    }

    private static func scale(rows: Int, levels: Int, size: CGSize) -> CGFloat {
        let width = CGFloat((capacity + levels - 1) / levels * tile)
        let height = CGFloat(rows * (tile * levels + rowGap))
        return min(size.width / width, size.height / height)
    }

    /// Tile rectangle for the `index`-th photo of `row`, in top-left pixel coordinates (nil past capacity).
    func slot(row: Int, index: Int) -> CGRect? {
        guard index < Self.capacity else { return nil }
        let column = index / levels
        let level = index % levels
        return CGRect(
            x: column * Self.tile, y: row * rowHeight + level * Self.tile, width: Self.tile, height: Self.tile)
    }

    /// Row a slot belongs to.
    func row(of slot: CGRect) -> Int { Int(slot.minY) / rowHeight }

    func draw(_ image: CGImage?, row: Int, index: Int, wrong: Bool) {
        guard let slot = slot(row: row, index: index) else { return }
        let target = rect(top: Int(slot.minY), x: Int(slot.minX), width: Self.tile, height: Self.tile)
        if let image {
            context.draw(Self.squareCrop(image), in: target.insetBy(dx: 0.5, dy: 0.5))
        }
        if wrong {
            context.setStrokeColor(CGColor(red: 1, green: 0.2, blue: 0.2, alpha: 1))
            context.setLineWidth(2.5)
            context.stroke(target.insetBy(dx: 1.25, dy: 1.25))
        }
    }

    func snapshot() -> CGImage? { context.makeImage() }

    /// Converts a top-left rectangle to the context's bottom-left coordinates.
    private func rect(top: Int, x: Int, width: Int, height: Int) -> CGRect {
        CGRect(x: x, y: self.height - top - height, width: width, height: height)
    }

    private static func squareCrop(_ image: CGImage) -> CGImage {
        let side = min(image.width, image.height)
        let crop = CGRect(x: (image.width - side) / 2, y: (image.height - side) / 2, width: side, height: side)
        return image.cropping(to: crop) ?? image
    }
}
