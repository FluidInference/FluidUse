import CoreGraphics
import Foundation

/// A bar chart made of photos: one row per breed, each sorted photo drawn as a tile at the end of its row.
/// Tiles go into one bitmap, so thousands of them cost one image on screen.
@MainActor
final class PhotoChart {
    static let tile = 16
    static let tilesPerColumn = 2
    static let columns = 130
    static let rowGap = 6
    static var rowHeight: Int { tile * tilesPerColumn + rowGap }
    static var capacity: Int { tilesPerColumn * columns }

    let rows: Int
    let width: Int
    let height: Int
    private let context: CGContext

    init(rows: Int, tint: (Int) -> CGColor) {
        self.rows = rows
        width = Self.columns * Self.tile
        height = rows * Self.rowHeight
        context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        for row in 0..<rows {
            context.setFillColor(tint(row))
            context.fill(rect(row: row, x: 0, width: width, height: Self.tile * Self.tilesPerColumn))
        }
    }

    /// Tile rectangle for the `index`-th photo of `row`, in top-left pixel coordinates (nil past capacity).
    static func slot(row: Int, index: Int) -> CGRect? {
        guard index < capacity else { return nil }
        let column = index / tilesPerColumn
        let level = index % tilesPerColumn
        return CGRect(x: column * tile, y: row * rowHeight + level * tile, width: tile, height: tile)
    }

    func draw(_ image: CGImage?, row: Int, index: Int, wrong: Bool) {
        guard let slot = Self.slot(row: row, index: index) else { return }
        let target = rect(
            row: row, x: Int(slot.minX), width: Self.tile, height: Self.tile, level: index % Self.tilesPerColumn)
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
    private func rect(row: Int, x: Int, width: Int, height: Int, level: Int = 0) -> CGRect {
        let top = row * Self.rowHeight + level * Self.tile
        return CGRect(x: x, y: self.height - top - height, width: width, height: height)
    }

    private static func squareCrop(_ image: CGImage) -> CGImage {
        let side = min(image.width, image.height)
        let crop = CGRect(x: (image.width - side) / 2, y: (image.height - side) / 2, width: side, height: side)
        return image.cropping(to: crop) ?? image
    }
}
