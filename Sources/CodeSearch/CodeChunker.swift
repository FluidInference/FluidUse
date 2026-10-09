import Foundation

/// One searchable piece of code: a function, initializer, or type declaration with its doc comment.
public struct CodeChunk: Sendable, Hashable {
    /// Path relative to the repository root.
    public let path: String
    /// 1-based line of the declaration.
    public let line: Int
    /// `Type.member`, or the type's own name.
    public let name: String
    public let kind: String
    /// Doc comment and code, up to `CodeChunker.maxLines` lines.
    public let code: String

    /// The document text EmbeddingGemma 2 was trained on for code retrieval: `title: <file> <symbol> | text: <code>`.
    public var document: String { "title: \((path as NSString).lastPathComponent) \(name) | text: \(code)" }
}

/// Splits Swift sources into declaration chunks with a line scanner (no compiler): each `func`, `init`,
/// `subscript`, or type declaration starts a chunk that runs, doc comment first, to the next declaration.
public enum CodeChunker {
    public static let maxLines = 40

    private static let declaration = try! NSRegularExpression(
        pattern:
            #"^(\s*)(?:@[A-Za-z_]+(?:\([^)]*\))?\s+)*(?:(?:public|private|internal|fileprivate|open|static|final|override|mutating|nonmutating|nonisolated|convenience|required|indirect|package|class|distributed)\s+|(?:private|fileprivate|internal|public)\(set\)\s+)*(func|init|deinit|subscript|struct|class|enum|actor|protocol|extension)\b\s*([A-Za-z_][A-Za-z0-9_.]*)?"#
    )
    private static let typeKinds: Set<String> = ["struct", "class", "enum", "actor", "protocol", "extension"]
    private static let skippedDirectories: Set<String> = [".build", ".git", ".swiftpm", "Packages", "DerivedData"]

    /// Chunks of every `.swift` file under `repository`, in path order.
    public static func chunks(repository: URL) -> [CodeChunk] {
        let root = repository.standardizedFileURL
        guard
            let walker = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        else { return [] }
        var files: [URL] = []
        for case let url as URL in walker {
            if skippedDirectories.contains(url.lastPathComponent) {
                walker.skipDescendants()
                continue
            }
            if url.pathExtension == "swift" { files.append(url) }
        }
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return files.sorted { $0.path < $1.path }.flatMap { file -> [CodeChunk] in
            guard let source = try? String(contentsOf: file, encoding: .utf8) else { return [] }
            let path = file.standardizedFileURL.path.replacingOccurrences(of: prefix, with: "")
            return chunks(source: source, path: path)
        }
    }

    /// Chunks of one file's source.
    public static func chunks(source: String, path: String) -> [CodeChunk] {
        let lines = source.components(separatedBy: "\n")
        // (line index, indent, kind, name)
        var declarations: [(index: Int, indent: Int, kind: String, name: String)] = []
        for (index, line) in lines.enumerated() {
            let range = NSRange(line.startIndex..., in: line)
            guard let match = declaration.firstMatch(in: line, range: range),
                let kindRange = Range(match.range(at: 2), in: line)
            else { continue }
            let kind = String(line[kindRange])
            // `class func` / `class var` are members, not types.
            if kind == "class",
                line.range(of: #"\bclass\s+(func|var|let|subscript)\b"#, options: .regularExpression) != nil
            {
                continue
            }
            let indent = Range(match.range(at: 1), in: line).map { line[$0].count } ?? 0
            var name = Range(match.range(at: 3), in: line).map { String(line[$0]) } ?? kind
            if kind == "init" || kind == "deinit" || kind == "subscript" { name = kind }
            declarations.append((index, indent, kind, name))
        }
        var result: [CodeChunk] = []
        var types: [(indent: Int, name: String)] = []
        for (position, item) in declarations.enumerated() {
            while let last = types.last, last.indent >= item.indent { types.removeLast() }
            let owner = types.last?.name
            if typeKinds.contains(item.kind) { types.append((item.indent, item.name)) }
            // Doc comment and attributes directly above the declaration.
            var first = item.index
            while first > 0 {
                let above = lines[first - 1].trimmingCharacters(in: .whitespaces)
                guard above.hasPrefix("///") || above.hasPrefix("@") || above.hasPrefix("*") || above.hasPrefix("/**")
                else { break }
                first -= 1
            }
            let next = position + 1 < declarations.count ? declarations[position + 1].index : lines.count
            var last = max(item.index, next - 1)
            // Leave the next declaration's doc comment to it.
            while last > item.index, lines[last].trimmingCharacters(in: .whitespaces).hasPrefix("///") { last -= 1 }
            let body = lines[first...min(last, first + maxLines - 1)]
            let code = body.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                .joined(separator: "\n")
            let name = typeKinds.contains(item.kind) || owner == nil ? item.name : "\(owner!).\(item.name)"
            result.append(CodeChunk(path: path, line: item.index + 1, name: name, kind: item.kind, code: code))
        }
        return result
    }
}
