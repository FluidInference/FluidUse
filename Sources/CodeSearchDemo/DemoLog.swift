import Foundation

/// Colour-coded console lines for the side terminal (`demo.sh` tails stdout): model load, indexing, searches (code search).
enum DemoLog {
    /// 256-colour codes matching the collection colours.
    private static let topicColors = [33, 208, 34, 205, 135, 37, 196, 61, 130, 121]

    static func line(_ text: String, color: Int? = nil, bold: Bool = false) {
        let time = timeFormatter.string(from: Date())
        var styled = text
        if let color { styled = "\u{1B}[38;5;\(color)m\(styled)\u{1B}[0m" }
        if bold { styled = "\u{1B}[1m\(styled)\u{1B}[0m" }
        print("\u{1B}[2m\(time)\u{1B}[0m \(styled)")
    }

    static func colored(_ text: String, slot: Int) -> String {
        "\u{1B}[38;5;\(topicColors[slot % topicColors.count])m\(text)\u{1B}[0m"
    }

    static func event(_ text: String) { line(text, color: 220, bold: true) }

    static func model(_ text: String) { line("[model] " + text, color: 81) }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()
}
