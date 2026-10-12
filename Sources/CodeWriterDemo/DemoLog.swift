import Foundation

/// Colour-coded console lines for the side terminal (`demo.sh` tails stdout): model load, each task, test results.
enum DemoLog {
    static func line(_ text: String, color: Int? = nil, bold: Bool = false) {
        let time = timeFormatter.string(from: Date())
        var styled = text
        if let color { styled = "\u{1B}[38;5;\(color)m\(styled)\u{1B}[0m" }
        if bold { styled = "\u{1B}[1m\(styled)\u{1B}[0m" }
        print("\u{1B}[2m\(time)\u{1B}[0m \(styled)")
    }

    static func event(_ text: String) { line(text, color: 220, bold: true) }

    static func model(_ text: String) { line("[model] " + text, color: 81) }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()
}
