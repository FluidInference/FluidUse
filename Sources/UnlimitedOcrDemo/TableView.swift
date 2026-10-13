import SwiftUI

/// The model's HTML tables (`<table><tr><td colspan=…>…</td></tr></table>`) as a SwiftUI grid. Only complete rows are
/// shown while a table is still streaming; cells may hold inline LaTeX.
struct TableView: View {
    struct Cell {
        let text: String
        let span: Int
    }

    let content: String

    var body: some View {
        let rows = Self.parse(content)
        if rows.isEmpty {
            Text(content.contains("<table") ? "…" : content)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.secondary)
        } else {
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                                MathText(content: cell.text, size: 13)
                                    .font(.system(size: 13, weight: index == 0 ? .semibold : .regular))
                                    .gridCellColumns(cell.span)
                            }
                        }
                        if index == 0 { Divider().gridCellUnsizedAxes(.horizontal) }
                    }
                }
                .padding(10)
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
        }
    }

    /// Complete `<tr>` rows; each cell's text with tags stripped and entities decoded. `rowspan` cells reserve their
    /// column in the rows below (an empty placeholder), so later cells stay in their columns.
    static func parse(_ html: String) -> [[Cell]] {
        let rowPattern = try! NSRegularExpression(
            pattern: #"<tr[^>]*>(.*?)</tr>"#, options: [.dotMatchesLineSeparators])
        let cellPattern = try! NSRegularExpression(
            pattern: #"<t[dh]([^>]*)>(.*?)</t[dh]>"#, options: [.dotMatchesLineSeparators])
        let ns = html as NSString
        var rows: [[Cell]] = []
        var reserved: [Int: (rows: Int, span: Int)] = [:]  // column -> rows still covered by a rowspan above
        for row in rowPattern.matches(in: html, range: NSRange(location: 0, length: ns.length)) {
            let rowHTML = ns.substring(with: row.range(at: 1))
            let rowNS = rowHTML as NSString
            var cells: [Cell] = []
            var column = 0
            func fillReserved() {
                while let hold = reserved[column], hold.rows > 0 {
                    cells.append(Cell(text: "", span: hold.span))
                    reserved[column] = hold.rows > 1 ? (hold.rows - 1, hold.span) : nil
                    column += hold.span
                }
            }
            for cell in cellPattern.matches(in: rowHTML, range: NSRange(location: 0, length: rowNS.length)) {
                fillReserved()
                let attributes = rowNS.substring(with: cell.range(at: 1))
                let span = max(1, attribute("colspan", in: attributes) ?? 1)
                let rowSpan = attribute("rowspan", in: attributes) ?? 1
                if rowSpan > 1 { reserved[column] = (rowSpan - 1, span) }
                cells.append(Cell(text: decode(rowNS.substring(with: cell.range(at: 2))), span: span))
                column += span
            }
            fillReserved()
            if !cells.isEmpty { rows.append(cells) }
        }
        return rows
    }

    private static func attribute(_ name: String, in attributes: String) -> Int? {
        guard let match = attributes.range(of: name + #"\s*=\s*"?\d+"#, options: .regularExpression) else { return nil }
        return Int(attributes[match].filter(\.isNumber))
    }

    private static func decode(_ text: String) -> String {
        text.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
            .replacingOccurrences(of: "&lt;", with: "<").replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&amp;", with: "&").replacingOccurrences(of: "&nbsp;", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
