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
        var sections = text.components(separatedBy: "<PAGE>")
        if sections.first?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true { sections.removeFirst() }
        var blockID = 0
        pages = sections.enumerated().map { Self.page($1, index: $0, blockID: &blockID) }
    }

    init(pages: [Page]) {
        self.pages = pages
    }

    /// One `<PAGE>` section: `<|det|>label [boxes]<|/det|>content` blocks (plain text if it has no markup yet).
    static func page(_ section: String, index: Int, blockID: inout Int) -> Page {
        var blocks: [Block] = []
        for part in section.components(separatedBy: "<|det|>").dropFirst() {
            let pieces = part.components(separatedBy: "<|/det|>")
            let header = pieces[0]
            let content = pieces.count > 1 ? pieces[1...].joined(separator: "") : ""
            let label =
                header.split(separator: "[", maxSplits: 1).first.map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
            blocks.append(
                Block(
                    id: blockID, label: label, boxes: boxes(in: header),
                    content: content.trimmingCharacters(in: .whitespacesAndNewlines)))
            blockID += 1
        }
        let plain = section.trimmingCharacters(in: .whitespacesAndNewlines)
        if blocks.isEmpty, !plain.isEmpty {
            blocks.append(Block(id: blockID, label: "text", boxes: [], content: plain))
            blockID += 1
        }
        return Page(id: index, blocks: blocks)
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

/// Incremental `UnlimitedOcrDocument` for streamed output: finished pages are parsed once at their `<PAGE>` boundary,
/// so each update re-parses only the page being written (O(page) instead of O(document)).
public struct UnlimitedOcrDocumentBuilder: Sendable {
    private var finished: [UnlimitedOcrDocument.Page] = []
    private var nextBlockID = 0
    private var current = ""
    private var sawMarker = false

    public init() {}

    public mutating func append(_ delta: String) {
        current += delta
        guard delta.contains(">") else { return }  // "<PAGE>" streams as several tokens; it ends with ">"
        while let marker = current.range(of: "<PAGE>") {
            let section = String(current[..<marker.lowerBound])
            current = String(current[marker.upperBound...])
            sawMarker = true
            if section.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, finished.isEmpty { continue }
            finished.append(UnlimitedOcrDocument.page(section, index: finished.count, blockID: &nextBlockID))
        }
    }

    public var document: UnlimitedOcrDocument {
        // Same as a full parse: only a leading empty section is dropped; a page just opened by "<PAGE>" exists (empty).
        if finished.isEmpty, !sawMarker, current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return UnlimitedOcrDocument(pages: [])
        }
        var blockID = nextBlockID
        return UnlimitedOcrDocument(
            pages: finished + [UnlimitedOcrDocument.page(current, index: finished.count, blockID: &blockID)])
    }
}
