import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: AudioSearchModel

    var body: some View {
        VStack(spacing: 0) {
            Header()
            Divider()
            SearchBar()
            Divider()
            Results()
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct Header: View {
    @EnvironmentObject private var model: AudioSearchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 10) {
                Text("Search audio").font(.title2.bold())
                Spacer()
                Text(status).font(.callout).foregroundStyle(.secondary)
                Button {
                    model.start()
                } label: {
                    Label("Index", systemImage: "play.fill")
                }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(model.phase != .ready && !(model.phase == .done && !model.isIndexing))
                Button {
                    model.toggleAutoPlay()
                } label: {
                    Label(model.autoPlay ? "Stop" : "Auto", systemImage: model.autoPlay ? "pause.fill" : "sparkles")
                }
                .keyboardShortcut(.space, modifiers: [])
                .disabled(model.indexedWindows == 0)
                Button {
                    model.reset()
                } label: {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }
                .keyboardShortcut("r", modifiers: [.command])
                .disabled(model.indexedWindows == 0 && !model.isIndexing)
            }
            GeometryReader { proxy in
                Capsule().fill(Color.primary.opacity(0.08))
                    .overlay(alignment: .leading) {
                        Capsule().fill(Color.accentColor)
                            .frame(
                                width: proxy.size.width * Double(model.windowsDone)
                                    / Double(max(model.windowsTotal, 1)))
                    }
            }
            .frame(height: 6)
            HStack(spacing: 10) {
                Tile(
                    value: AudioSearchModel.clock(model.audioSeconds),
                    caption: model.pass > 1 ? "audio read, pass \(model.pass)" : "audio read")
                Tile(value: String(format: "%.1f s", model.indexSeconds), caption: "indexing time")
                Tile(
                    value: model.realTimeFactor == 0 ? "–" : String(format: "%.0f×", model.realTimeFactor),
                    caption: "faster than real time")
                Tile(value: "\(model.indexedWindows)", caption: "windows searchable")
            }
            HStack(spacing: 8) {
                ForEach(model.collections) { collection in
                    HStack(spacing: 5) {
                        Circle().fill(collection.color).frame(width: 8, height: 8)
                        Text("\(collection.name) · \(collection.files.count)")
                    }
                    .font(.caption)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill(collection.color.opacity(0.12)))
                }
                Spacer()
                Text("EmbeddingGemma 2 · audio on the GPU, text on the Neural Engine · nothing leaves this Mac")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(16)
    }

    private var status: String {
        switch (model.segment, model.phase) {
        case (.speed, _):
            model.pass > 1 && model.indexedWindows > 0
                ? "⚡ Reading everything again, as fast as it can · \(model.segmentRemaining) s"
                : "⚡ Reading everything for the first time · \(model.segmentRemaining) s"
        case (.search, _): "🔎 Searching · \(model.segmentRemaining) s left"
        case (nil, .loading(let message)): message
        case (nil, .ready): "Ready · press Index (⌘↩)"
        case (nil, .indexing): "Indexing \(model.windowsDone)/\(model.windowsTotal) windows…"
        case (nil, .done): "Indexed · search below"
        case (nil, .failed(let message)): message
        }
    }
}

struct SearchBar: View {
    @EnvironmentObject private var model: AudioSearchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Describe what you're looking for: a phrase, a topic, a sound…", text: $model.query)
                    .textFieldStyle(.plain).font(.title3)
                    .onSubmit { model.search() }
                    .onChange(of: model.query) { model.search() }
                if model.queryMilliseconds > 0, !model.results.isEmpty {
                    Text(String(format: "%.0f ms", model.queryMilliseconds)).font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(model.allSuggestions, id: \.self) { suggestion in
                        Button(suggestion) {
                            model.query = suggestion
                            model.search()
                        }
                        .buttonStyle(.bordered).controlSize(.small)
                    }
                }
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
    }
}

struct Results: View {
    @EnvironmentObject private var model: AudioSearchModel

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(Array(model.results.enumerated()), id: \.element.id) { rank, result in
                    ResultRow(rank: rank + 1, result: result)
                    Divider()
                }
            }
        }
        .overlay {
            if model.results.isEmpty {
                Text(model.phase == .done ? "Type a query or pick a suggestion" : "Index the audio, then search it")
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct ResultRow: View {
    @EnvironmentObject private var model: AudioSearchModel
    let rank: Int
    let result: AudioSearchModel.Result

    var body: some View {
        let collection = model.collections[result.entry.collection]
        HStack(spacing: 12) {
            Text("\(rank)").font(.callout.monospacedDigit()).foregroundStyle(.secondary).frame(width: 24)
            Button {
                model.play(result)
            } label: {
                Image(systemName: model.playing == result.id ? "stop.circle.fill" : "play.circle.fill")
                    .font(.system(size: 26)).foregroundStyle(collection.color)
            }
            .buttonStyle(.plain)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(collection.name).font(.caption.bold()).foregroundStyle(collection.color)
                    Text(result.entry.file.lastPathComponent).font(.callout).lineLimit(1)
                }
                Text(
                    "\(AudioSearchModel.clock(result.entry.start)) – "
                        + AudioSearchModel.clock(result.entry.start + result.entry.duration)
                )
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Spacer()
            ScoreBar(score: result.score, color: collection.color)
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
        .background(model.playing == result.id ? collection.color.opacity(0.08) : .clear)
    }
}

struct ScoreBar: View {
    let score: Float
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            GeometryReader { proxy in
                Capsule().fill(Color.primary.opacity(0.08))
                    .overlay(alignment: .leading) {
                        // Cosine scores here sit around 0.5–0.8; stretch that range across the bar.
                        Capsule().fill(color)
                            .frame(width: proxy.size.width * CGFloat(min(max((score - 0.5) / 0.35, 0.03), 1)))
                    }
            }
            .frame(width: 120, height: 6)
            Text(String(format: "%.3f", score)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
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
