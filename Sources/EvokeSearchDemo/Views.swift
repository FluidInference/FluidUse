import SwiftUI

extension Color {
    static let feedBackground = Color(red: 0, green: 0, blue: 0)
    static let feedDivider = Color(white: 0.18)
    static let feedSecondary = Color(white: 0.45)
    static let feedAccent = Color(red: 0.11, green: 0.61, blue: 0.94)
    static let evoked = Color(red: 0.98, green: 0.69, blue: 0.2)
}

struct RootView: View {
    @State private var model = DemoModel()

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            FeedColumn(model: model)
                .frame(width: 600)
            Divider().overlay(Color.feedDivider)
            InsightColumn(model: model)
                .frame(width: 320)
        }
        .background(Color.feedBackground)
        .foregroundStyle(.white)
        .preferredColorScheme(.dark)
        .task { await model.start() }
    }
}

struct FeedColumn: View {
    @Bindable var model: DemoModel

    var body: some View {
        VStack(spacing: 0) {
            header
            if !model.caption.isEmpty {
                Text(model.caption)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(model.mode == .evoke ? Color.evoked : Color.feedAccent)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background((model.mode == .evoke ? Color.evoked : Color.feedAccent).opacity(0.1))
            }
            Divider().overlay(Color.feedDivider)
            switch model.phase {
            case .loading(let message):
                ProgressView(message).padding(40)
                Spacer()
            case .failed(let message):
                Text(message).foregroundStyle(.red).padding(24)
                Spacer()
            case .ready:
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(model.hits) { hit in
                            TweetRow(hit: hit, showTerms: !model.query.isEmpty)
                            Divider().overlay(Color.feedDivider)
                        }
                        if model.hits.isEmpty {
                            Text("No results for “\(model.query)”")
                                .foregroundStyle(Color.feedSecondary)
                                .padding(40)
                        }
                    }
                }
            }
        }
    }

    private var header: some View {
        VStack(spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").foregroundStyle(Color.feedSecondary)
                TextField("Search", text: $model.query)
                    .textFieldStyle(.plain)
                    .font(.system(size: 16))
                    .onChange(of: model.query) { model.search() }
                    .onKeyPress { _ in
                        if model.isAutoplaying { model.stopAutoplay() }
                        return .ignored
                    }
                if !model.query.isEmpty {
                    Button {
                        model.query = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(Color.feedAccent)
                    }
                    .buttonStyle(.plain)
                }
                Button {
                    model.restart()
                } label: {
                    Label("Restart", systemImage: "arrow.clockwise")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Color.feedAccent)
                }
                .buttonStyle(.plain)
                .help("Re-index every post, then run the demo again")
                Button {
                    model.isAutoplaying ? model.stopAutoplay(resume: false) : model.startAutoplay()
                } label: {
                    Label(
                        model.isAutoplaying ? "Pause demo" : "Auto demo",
                        systemImage: model.isAutoplaying ? "pause.fill" : "play.fill"
                    )
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Color.feedAccent)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Capsule().fill(Color(white: 0.13)))

            HStack(spacing: 0) {
                ForEach(SearchMode.allCases, id: \.self) { mode in
                    Button {
                        model.stopAutoplay()
                        model.mode = mode
                        model.search()
                    } label: {
                        VStack(spacing: 8) {
                            Text(mode == .evoke ? "Evoke (on-device)" : "Keyword")
                                .font(.system(size: 15, weight: model.mode == mode ? .bold : .regular))
                                .foregroundStyle(model.mode == mode ? .white : Color.feedSecondary)
                            Capsule()
                                .fill(model.mode == mode ? Color.feedAccent : .clear)
                                .frame(width: 70, height: 4)
                        }
                        .frame(maxWidth: .infinity)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(DemoModel.suggestions, id: \.self) { s in
                        Button(s) {
                            model.stopAutoplay()
                            model.query = s
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 13))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .overlay(Capsule().stroke(Color.feedDivider))
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 14)
        .padding(.bottom, 4)
    }
}

struct TweetRow: View {
    let hit: SearchHit
    let showTerms: Bool

    var body: some View {
        let tweet = hit.tweet
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(Color(hue: tweet.hue, saturation: 0.55, brightness: 0.75))
                .frame(width: 42, height: 42)
                .overlay(
                    Text(String(tweet.name.prefix(1)))
                        .font(.system(size: 18, weight: .bold))
                )
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 4) {
                    Text(tweet.name).fontWeight(.bold)
                    Text("@\(tweet.handle) · \(tweet.age)").foregroundStyle(Color.feedSecondary)
                }
                .font(.system(size: 15))
                Text(tweet.text)
                    .font(.system(size: 15))
                    .fixedSize(horizontal: false, vertical: true)
                if showTerms && !hit.terms.isEmpty {
                    TermChips(terms: hit.terms)
                }
                HStack {
                    stat("bubble.right", tweet.replies)
                    stat("arrow.2.squarepath", tweet.reposts)
                    stat("heart", tweet.likes)
                    stat("chart.bar", tweet.likes * 13)
                }
                .padding(.top, 4)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func stat(_ icon: String, _ n: Int) -> some View {
        Label(n >= 1000 ? String(format: "%.1fK", Double(n) / 1000) : "\(n)", systemImage: icon)
            .font(.system(size: 13))
            .foregroundStyle(Color.feedSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct TermChips: View {
    let terms: [MatchTerm]

    var body: some View {
        HStack(spacing: 6) {
            Text("matched").font(.system(size: 12)).foregroundStyle(Color.feedSecondary)
            ForEach(terms, id: \.self) { term in
                HStack(spacing: 3) {
                    if term.evoked { Image(systemName: "sparkles").font(.system(size: 10)) }
                    Text(term.word)
                }
                .lineLimit(1)
                .fixedSize()
                .font(.system(size: 12, weight: .medium))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .foregroundStyle(term.evoked ? Color.evoked : Color.feedAccent)
                .background(
                    Capsule().fill((term.evoked ? Color.evoked : Color.feedAccent).opacity(0.14))
                )
            }
        }
    }
}

struct InsightColumn: View {
    let model: DemoModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Query expansion").font(.system(size: 18, weight: .heavy))
                if model.queryTerms.isEmpty {
                    Text("Type a query. Evoke turns it into weighted words, including related words you never typed.")
                        .font(.system(size: 14))
                        .foregroundStyle(Color.feedSecondary)
                } else {
                    let top = model.queryTerms.map(\.weight).max() ?? 1
                    ForEach(model.queryTerms, id: \.word) { term in
                        HStack(spacing: 8) {
                            Text(term.word)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(term.evoked ? Color.evoked : .white)
                                .frame(width: 110, alignment: .leading)
                            GeometryReader { geo in
                                Capsule()
                                    .fill(term.evoked ? Color.evoked : Color.feedAccent)
                                    .frame(width: max(4, geo.size.width * CGFloat(term.weight / top)))
                            }
                            .frame(height: 6)
                        }
                    }
                    HStack(spacing: 6) {
                        Image(systemName: "sparkles").foregroundStyle(Color.evoked)
                        Text("= not in your query").foregroundStyle(Color.feedSecondary)
                    }
                    .font(.system(size: 12))
                }
            }
            .padding(16)
            .background(RoundedRectangle(cornerRadius: 16).fill(Color(white: 0.09)))

            VStack(alignment: .leading, spacing: 8) {
                Text("On this Mac").font(.system(size: 18, weight: .heavy))
                if let ms = model.queryLatencyMs {
                    Text(String(format: "Query encoded in %.2f ms · Neural Engine", ms))
                }
                if !model.indexSummary.isEmpty { Text(model.indexSummary) }
                Text("Granite-Embedding-30M-Sparse · Core ML fp16 · on-device")
            }
            .font(.system(size: 13))
            .foregroundStyle(Color.feedSecondary)
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 16).fill(Color(white: 0.09)))
            Spacer()
        }
        .padding(16)
    }
}
