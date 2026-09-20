import Foundation
import PDFKit

/// Port of upstream `cua_s1.pdf`: `Label: value` lines become entities.
enum DocumentEntities {
    static let maxValueLength = 120
    // Same expression as upstream `LINE`, applied per line.
    private static let line = try? NSRegularExpression(
        pattern: #"^\s*([A-Za-z][A-Za-z0-9 .'/#&()-]{1,40}?)\s*[:\u2013-]\s+(.+?)\s*$"#)

    enum Failure: Error, LocalizedError {
        case unreadable(URL)
        case unsupported(String)

        var errorDescription: String? {
            switch self {
            case .unreadable(let url): return "Could not read \(url.lastPathComponent)"
            case .unsupported(let ext): return "Unsupported document type .\(ext); use PDF or plain text"
            }
        }
    }

    static func extract(from url: URL) throws -> [Entity] {
        let text: String
        switch url.pathExtension.lowercased() {
        case "pdf":
            guard let document = PDFDocument(url: url) else { throw Failure.unreadable(url) }
            text = (0..<document.pageCount).compactMap { document.page(at: $0)?.string }.joined(separator: "\n")
        case "txt", "md", "text":
            text = try String(contentsOf: url, encoding: .utf8)
        default:
            throw Failure.unsupported(url.pathExtension)
        }
        return FormSchema.deriveEntities(parse(text))
    }

    static func parse(_ text: String) -> [Entity] {
        guard let line else { return [] }
        var seen = Set<String>()
        var entities: [Entity] = []
        for rawLine in text.components(separatedBy: .newlines) {
            let range = NSRange(rawLine.startIndex..., in: rawLine)
            guard let match = line.firstMatch(in: rawLine, range: range),
                let labelRange = Range(match.range(at: 1), in: rawLine),
                let valueRange = Range(match.range(at: 2), in: rawLine)
            else { continue }
            let label = rawLine[labelRange].trimmingCharacters(in: .whitespaces)
            let value = rawLine[valueRange].trimmingCharacters(in: .whitespaces)
            let key = label + "\u{0}" + value
            guard value.count <= maxValueLength, !seen.contains(key) else { continue }
            seen.insert(key)
            entities.append(Entity(label: label, value: value))
        }
        return entities
    }
}
