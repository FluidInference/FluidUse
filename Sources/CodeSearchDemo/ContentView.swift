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
                Tile(value: model.chunksDone.formatted(), caption: "functions & types indexed")
                Tile(value: String(format: "%.1f s", model.indexSeconds), caption: "indexing time")
            }
            HStack(spacing: 10) {
                Tile(value: model.burstQueries == 0 ? "–" : model.burstQueries.formatted(), caption: "searches run")
                Tile(
                    value: model.burstPerSecond == 0 ? "–" : String(format: "%.0f", model.burstPerSecond),
                    caption: "searches / second")
                Tile(
                    value: model.burstMilliseconds == 0 ? "–" : String(format: "%.2f ms", model.burstMilliseconds),
                    caption: "per search")
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
                ForEach(Array(model.results.prefix(12).enumerated()), id: \.element.id) { rank, result in
                    ResultRow(rank: rank + 1, result: result, selected: model.selected?.id == result.id)
                        .onTapGesture { model.select(result) }
                    Divider()
                }
            }
        }
        .overlay {
            if model.results.isEmpty {
                Text(model.step == .finished ? "Ask a question" : "Press Play").foregroundStyle(.secondary)
            }
        }
    }
}

struct ResultRow: View {
    let rank: Int
    let result: CodeSearchModel.Result
    let selected: Bool

    var body: some View {
        HStack(spacing: 12) {
            Text("\(rank)").font(.callout.monospacedDigit()).foregroundStyle(.secondary).frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                Text(result.chunk.name).font(.system(.callout, design: .monospaced).bold()).lineLimit(1)
                Text("\(result.chunk.path):\(result.chunk.line)").font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.head)
            }
            Spacer()
            Text(String(format: "%.3f", result.score)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
        .background(selected ? Color.accentColor.opacity(0.14) : .clear)
    }
}

struct CodePreview: View {
    @EnvironmentObject private var model: CodeSearchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let result = model.selected {
                HStack {
                    Text(result.chunk.name).font(.system(.headline, design: .monospaced))
                    Spacer()
                    Text("\(result.chunk.path):\(result.chunk.line)").font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.head)
                }
                .padding(12)
                Divider()
                ScrollView {
                    Text(model.snippet(result.chunk)).font(.system(size: 12, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                        .textSelection(.enabled)
                }
            } else {
                Spacer()
                Text("The matching code shows here").foregroundStyle(.secondary).frame(maxWidth: .infinity)
                Spacer()
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
    }
}

struct Tile: View {
    let value: String
    let caption: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.system(size: 24, weight: .semibold, design: .rounded)).monospacedDigit()
                .contentTransition(.numericText())
            Text(caption).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }
}
