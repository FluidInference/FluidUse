import CoreGraphics
import Foundation

/// Parsed model output: `<PAGE>` sections of `<|det|>label [x1, y1, x2, y2]<|/det|>content` blocks. Boxes are in
/// upstream's 0...999 page coordinates (top-left origin). Parsing is tolerant of a partially streamed tail.
public struct UnlimitedOcrDocument: Sendable {
    public struct Block: Sendable, Identifiable {
        public let id: Int
        public let label: String
        /// Normalized 0...1 rectangles (top-left origin); usually one.
        public let boxes: [CGRect]
        public let content: String
    }

    public struct Page: Sendable, Identifiable {
        public let id: Int
        public let blocks: [Block]
    }

    public let pages: [Page]

    public init(parsing text: String) {
        var pages: [Page] = []
        var sections = text.components(separatedBy: "<PAGE>")
        if sections.first?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true { sections.removeFirst() }
        var blockID = 0
        for (pageIndex, section) in sections.enumerated() {
            var blocks: [Block] = []
            let parts = section.components(separatedBy: "<|det|>")
            for part in parts.dropFirst() {
                let pieces = part.components(separatedBy: "<|/det|>")
                let header = pieces[0]
                let content = pieces.count > 1 ? pieces[1...].joined(separator: "") : ""
                let label =
                    header.split(separator: "[", maxSplits: 1).first.map {
                        $0.trimmingCharacters(in: .whitespaces)
                    } ?? ""
                blocks.append(
                    Block(
                        id: blockID, label: label, boxes: Self.boxes(in: header),
                        content: content.trimmingCharacters(in: .whitespacesAndNewlines)))
                blockID += 1
            }
            if blocks.isEmpty, !section.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                blocks.append(
                    Block(
                        id: blockID, label: "text", boxes: [],
                        content: section.trimmingCharacters(in: .whitespacesAndNewlines)))
                blockID += 1
            }
            pages.append(Page(id: pageIndex, blocks: blocks))
        }
        self.pages = pages
    }

    /// Every complete `[x1, y1, x2, y2]` group in a det header, normalized.
    static func boxes(in header: String) -> [CGRect] {
        var rects: [CGRect] = []
        var numbers: [Double] = []
        var current = ""
        var inside = false
        for character in header {
            switch character {
            case "[":
                inside = true
                numbers = []
                current = ""
            case "]":
                if let value = Double(current.trimmingCharacters(in: .whitespaces)) { numbers.append(value) }
                if inside, numbers.count == 4 {
                    rects.append(
                        CGRect(
                            x: numbers[0] / 999, y: numbers[1] / 999, width: (numbers[2] - numbers[0]) / 999,
                            height: (numbers[3] - numbers[1]) / 999))
                }
                inside = false
                current = ""
            case ",":
                if let value = Double(current.trimmingCharacters(in: .whitespaces)) { numbers.append(value) }
                current = ""
            default:
                if inside { current.append(character) }
            }
        }
        return rects
    }

    /// Plain markdown: layout markup removed, pages separated by a rule.
    public var markdown: String {
        pages.map { page in page.blocks.map(\.content).filter { !$0.isEmpty }.joined(separator: "\n\n") }
            .joined(separator: "\n\n---\n\n")
    }
}
