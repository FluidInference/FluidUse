import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: CodeSearchModel

    var body: some View {
        VStack(spacing: 0) {
            Header()
            Divider()
            SearchBar()
            Divider()
            HSplitView {
                Results().frame(minWidth: 420)
                CodePreview().frame(minWidth: 420)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct Header: View {
    @EnvironmentObject private var model: CodeSearchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text("Code search").font(.title2.bold())
                Spacer()
                Text(status).font(.callout).foregroundStyle(.secondary)
                Button {
                    model.play()
                } label: {
                    Label(model.paused ? "Resume" : "Play", systemImage: "play.fill")
                }
                .keyboardShortcut(.return, modifiers: [])
                .disabled(!model.canPlay)
                Button {
                    model.pause()
                } label: {
                    Label("Pause", systemImage: "pause.fill")
                }
                .keyboardShortcut(.space, modifiers: [])
                .disabled(!model.isRunning || model.paused)
                Button {
                    model.replay()
                } label: {
                    Label("Replay", systemImage: "arrow.counterclockwise")
                }
                .keyboardShortcut("r", modifiers: [.command])
                .disabled(!model.canReplay)
            }
            GeometryReader { proxy in
                Capsule().fill(Color.primary.opacity(0.08))
                    .overlay(alignment: .leading) {
                        Capsule().fill(Color.accentColor)
                            .frame(
                                width: proxy.size.width * Double(model.chunksDone) / Double(max(model.chunksTotal, 1)))
                    }
            }
            .frame(height: 6)
            HStack(spacing: 10) {
                Tile(value: "\(model.fileCount)", caption: "Swift files")
                Tile(value: model.lineCount.formatted(), caption: "lines of code")
                Tile(value: model.chunksDone.formatted(), caption: "functions & types indexed", tint: .cyan)
                Tile(value: String(format: "%.1f s", model.indexSeconds), caption: "indexing time", tint: .green)
            }
            HStack(spacing: 10) {
                Tile(value: model.burstQueries == 0 ? "–" : model.burstQueries.formatted(), caption: "searches run")
                Tile(
                    value: model.burstPerSecond == 0 ? "–" : String(format: "%.0f", model.burstPerSecond),
                    caption: "searches / second", tint: .orange)
                Tile(
                    value: model.burstMilliseconds == 0 ? "–" : String(format: "%.2f ms", model.burstMilliseconds),
                    caption: "per search", tint: .pink)
                Tile(
                    value: model.burstPerSecond == 0
                        ? "–" : String(format: "%.1f M", model.burstPerSecond * Double(model.chunksDone) / 1e6),
                    caption: "functions ranked / s")
            }
            .opacity(model.step == .speed || model.step == .finished ? 1 : 0.55)
            HStack {
                Text("\(model.repositoryName) · plain-English questions, no keywords, no grep")
                    .font(.caption.bold()).foregroundStyle(.secondary)
                Spacer()
                Text("EmbeddingGemma 2 on the Neural Engine · nothing leaves this Mac")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(16)
    }

    private var status: String {
        let pause = model.paused ? " · paused" : ""
        switch model.step {
        case .loading(let message): return message
        case .idle: return "Ready · press Play"
        case .indexing: return "📥 Indexing \(model.chunksDone)/\(model.chunksTotal)" + pause
        case .examples:
            return "🔎 Example question \(model.exampleNumber) of \(CodeSearchModel.examples.count)" + pause
        case .speed: return "⚡ Searching as fast as it can · \(model.stepRemaining) s left" + pause
        case .finished: return "Done · ask your own question, or Replay"
        case .failed(let message): return message
        }
    }
}

struct SearchBar: View {
    @EnvironmentObject private var model: CodeSearchModel

    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("Ask in plain English: “where do we …”", text: $model.query)
                .textFieldStyle(.plain).font(.title3)
                .onSubmit { model.search() }
                .onChange(of: model.query) { model.search() }
                .disabled(model.isRunning)
            if let grep = model.grepMatches {
                Text("exact-phrase grep: \(grep) file\(grep == 1 ? "" : "s")")
                    .font(.caption.bold()).foregroundStyle(grep == 0 ? .orange : .secondary)
            }
            if model.queryMilliseconds > 0, !model.results.isEmpty {
                Text(String(format: "%.0f ms", model.queryMilliseconds)).font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
    }
}

struct Results: View {
    @EnvironmentObject private var model: CodeSearchModel

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                if model.step == .indexing || model.step == .speed {
                    // Live list: functions as they are indexed, or questions as they are answered.
                    ForEach(model.feed) { item in
                        FeedRow(item: item)
                            .transition(.move(edge: .top).combined(with: .opacity))
                        Divider().opacity(0.4)
                    }
                } else {
                    ForEach(Array(model.results.prefix(12).enumerated()), id: \.element.id) { rank, result in
                        ResultRow(rank: rank + 1, result: result, selected: model.selected?.id == result.id)
                            .onTapGesture { model.select(result) }
                        Divider().opacity(0.4)
                    }
                }
            }
            .animation(.easeOut(duration: 0.15), value: model.feed.map(\.id))
        }
        .background(Color(red: 0.11, green: 0.12, blue: 0.14))
        .overlay {
            if model.results.isEmpty, model.feed.isEmpty {
                Text(model.step == .finished ? "Ask a question" : "Press Play").foregroundStyle(.secondary)
            }
        }
    }
}

/// `Type.member` with the type in the type colour and the member bold in the function colour.
struct SymbolName: View {
    let name: String
    var size: CGFloat = 13

    var body: some View {
        let parts = name.split(separator: ".", maxSplits: 1).map(String.init)
        let owner = parts.count == 2 ? Text(parts[0] + ".").foregroundColor(SwiftHighlighter.type) : Text("")
        let member = Text(parts.last ?? name).bold()
            .foregroundColor(parts.count == 2 ? SwiftHighlighter.call : SwiftHighlighter.type)
        return (owner + member).font(.system(size: size, design: .monospaced)).lineLimit(1)
    }
}

struct AreaTag: View {
    let path: String

    var body: some View {
        let area = CodeSearchModel.area(path)
        let color = SwiftHighlighter.areaColor(area)
        Text(area).font(.system(size: 10, weight: .bold, design: .rounded)).foregroundStyle(color)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.18)))
    }
}

struct FeedRow: View {
    let item: CodeSearchModel.FeedItem

    var body: some View {
        HStack(spacing: 10) {
            AreaTag(path: item.chunk.path).frame(width: 64, alignment: .leading)
            if let question = item.question {
                Text(question).font(.system(size: 12)).foregroundStyle(.primary.opacity(0.85)).lineLimit(1)
                Image(systemName: "arrow.right").font(.system(size: 10, weight: .bold)).foregroundStyle(.secondary)
            }
            SymbolName(name: item.chunk.name, size: 12)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14).padding(.vertical, 5)
    }
}

struct ResultRow: View {
    let rank: Int
    let result: CodeSearchModel.Result
    let selected: Bool

    var body: some View {
        let color = SwiftHighlighter.areaColor(CodeSearchModel.area(result.chunk.path))
        HStack(spacing: 12) {
            Text("\(rank)").font(.system(.callout, design: .rounded).bold().monospacedDigit())
                .foregroundStyle(rank <= 3 ? color : .secondary).frame(width: 22)
            VStack(alignment: .leading, spacing: 4) {
                SymbolName(name: result.chunk.name)
                HStack(spacing: 6) {
                    AreaTag(path: result.chunk.path)
                    Text("\(result.chunk.path):\(result.chunk.line)").font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.head)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 3) {
                Text(String(format: "%.3f", result.score)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                GeometryReader { proxy in
                    Capsule().fill(Color.primary.opacity(0.08))
                        .overlay(alignment: .leading) {
                            // Scores here sit around 0.6–0.85; stretch that range across the bar.
                            Capsule().fill(
                                LinearGradient(
                                    colors: [color.opacity(0.6), color], startPoint: .leading, endPoint: .trailing)
                            )
                            .frame(width: proxy.size.width * CGFloat(min(max((result.score - 0.6) / 0.25, 0.05), 1)))
                        }
                }
                .frame(width: 70, height: 5)
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
        .background(selected ? color.opacity(0.16) : .clear)
        .overlay(alignment: .leading) { if selected { Rectangle().fill(color).frame(width: 3) } }
    }
}

struct CodePreview: View {
    @EnvironmentObject private var model: CodeSearchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let result = model.selected {
                HStack(spacing: 8) {
                    AreaTag(path: result.chunk.path)
                    SymbolName(name: result.chunk.name, size: 14)
                    Spacer()
                    Text("\(result.chunk.path):\(result.chunk.line)").font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.head)
                }
                .padding(12)
                Divider()
                ScrollView {
                    Text(SwiftHighlighter.highlight(model.snippet(result.chunk)))
                        .font(.system(size: 12, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                        .textSelection(.enabled)
                }
            } else {
                Spacer()
                Text("The matching code shows here").foregroundStyle(.secondary).frame(maxWidth: .infinity)
                Spacer()
            }
        }
        .background(Color(red: 0.12, green: 0.12, blue: 0.14))
        .environment(\.colorScheme, .dark)
    }
}

struct Tile: View {
    let value: String
    let caption: String
    var tint: Color = .primary

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.system(size: 24, weight: .semibold, design: .rounded)).monospacedDigit()
                .foregroundStyle(tint)
                .contentTransition(.numericText())
            Text(caption).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }
}
