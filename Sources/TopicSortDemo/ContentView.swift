import AppKit
import BookmarkSort
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: TopicSortModel
    @State private var renaming: TopicNode?
    @State private var newName = ""

    var body: some View {
        NavigationSplitView {
            Sidebar(renaming: $renaming, newName: $newName).navigationSplitViewColumnWidth(min: 280, ideal: 340)
        } detail: {
            VStack(spacing: 0) {
                Header()
                Divider()
                Feed()
            }
        }
        .alert("Rename topic", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $newName)
            Button("Rename") { if let renaming { model.rename(renaming.id, to: newName) } }
            Button("Cancel", role: .cancel) {}
        }
    }
}

struct Sidebar: View {
    @EnvironmentObject private var model: TopicSortModel
    @Binding var renaming: TopicNode?
    @Binding var newName: String

    var body: some View {
        List(selection: $model.selection) {
            TopicRow(name: "All posts", count: model.posts.count, color: .secondary).tag(TopicSortModel.allID)
            Section(model.topics.isEmpty ? "Topics appear after 40 posts" : "Broad topics") {
                ForEach(model.topics) { topic in
                    TopicRow(name: topic.name, count: topic.members.count, color: model.color(topic)) {
                        if topic.children.isEmpty {
                            Button("Split") { Task { await model.split(topic.id) } }
                                .disabled(!model.canSplit)
                        } else {
                            Button("Merge") { model.merge(topic.id) }
                        }
                    }
                    .tag(topic.id)
                    .contextMenu { renameButton(topic) }
                    ForEach(topic.children) { child in
                        TopicRow(name: child.name, count: child.members.count, color: model.color(child))
                            .padding(.leading, 18)
                            .tag(child.id)
                            .contextMenu { renameButton(child) }
                            .transition(.move(edge: .leading).combined(with: .opacity))
                    }
                }
            }
        }
        .listStyle(.sidebar)
    }

    private func renameButton(_ node: TopicNode) -> some View {
        Button("Rename…") {
            newName = node.name
            renaming = node
        }
    }
}

struct TopicRow<Accessory: View>: View {
    let name: String
    let count: Int
    let color: Color
    @ViewBuilder var accessory: () -> Accessory

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(color).frame(width: 9, height: 9)
            Text(name).lineLimit(2)
            Spacer()
            Text("\(count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .contentTransition(.numericText())
            accessory().controlSize(.small).buttonStyle(.bordered)
        }
    }
}

extension TopicRow where Accessory == EmptyView {
    init(name: String, count: Int, color: Color) {
        self.init(name: name, count: count, color: color) { EmptyView() }
    }
}

struct Feed: View {
    @EnvironmentObject private var model: TopicSortModel

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0, pinnedViews: .sectionHeaders) {
                ForEach(model.feedSections, id: \.id) { section in
                    Section {
                        ForEach(section.items.prefix(300), id: \.self) { index in
                            let path = model.path(index)
                            TweetCard(
                                post: model.posts[index], path: path, color: path.first.map { model.color($0) })
                            Divider()
                        }
                    } header: {
                        if let title = section.title, let color = section.color {
                            HStack {
                                Circle().fill(color).frame(width: 9, height: 9)
                                Text(title).font(.headline)
                                Text("\(section.items.count) posts").font(.caption).foregroundStyle(.secondary)
                                Spacer()
                            }
                            .padding(.horizontal, 16).padding(.vertical, 8)
                            .background(.bar)
                        }
                    }
                }
            }
            .frame(maxWidth: 640)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .textBackgroundColor))
    }
}

struct Header: View {
    @EnvironmentObject private var model: TopicSortModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 10) {
                Text("Sort by topic").font(.title2.bold())
                Spacer()
                Text(status).font(.callout).foregroundStyle(.secondary)
                controls
            }
            GeometryReader { proxy in
                Capsule().fill(Color.primary.opacity(0.08))
                    .overlay(alignment: .leading) {
                        Capsule().fill(Color.accentColor)
                            .frame(
                                width: proxy.size.width * Double(model.posts.count) / Double(max(model.totalCount, 1)))
                    }
            }
            .frame(height: 6)
            HStack(spacing: 10) {
                Tile(value: "\(model.posts.count)", caption: "posts sorted")
                Tile(
                    value: model.postsPerSecond == 0 ? "–" : String(format: "%.0f", model.postsPerSecond),
                    caption: "posts / second")
                Tile(value: model.topics.isEmpty ? "–" : "\(model.topics.count)", caption: "broad topics")
                Tile(value: model.subtopicCount == 0 ? "–" : "\(model.subtopicCount)", caption: "subtopics")
            }
            Text(
                model.lastEvent.isEmpty
                    ? "EmbeddingGemma 2 on the Neural Engine · topics found from the posts alone · nothing leaves this Mac"
                    : model.lastEvent
            )
            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(16)
    }

    @ViewBuilder private var controls: some View {
        switch model.phase {
        case .streaming, .resorting:
            Button {
                model.pause()
            } label: {
                Label("Pause", systemImage: "pause.fill")
            }
            .keyboardShortcut(.space, modifiers: [])
        case .paused:
            Button {
                model.start()
            } label: {
                Label("Resume", systemImage: "play.fill")
            }
            .keyboardShortcut(.space, modifiers: [])
        default:
            Button {
                model.start()
            } label: {
                Label("Start", systemImage: "play.fill")
            }
            .keyboardShortcut(.space, modifiers: [])
            .disabled(model.phase != .ready)
        }
        Button {
            model.reset()
        } label: {
            Label("Reset", systemImage: "arrow.counterclockwise")
        }
        .keyboardShortcut("r", modifiers: [.command])
        .disabled(model.posts.isEmpty)
    }

    private var status: String {
        switch model.phase {
        case .loading(let message): message
        case .ready: "Ready · \(model.totalCount) posts"
        case .streaming: "Live · \(model.posts.count)/\(model.totalCount)"
        case .paused: "Paused · \(model.posts.count)/\(model.totalCount)"
        case .resorting: "Re-sorting broad topics…"
        case .splitting(let name): "Splitting “\(name)”…"
        case .done: "Done · pick a topic, press Split"
        case .failed(let message): message
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

/// `--snapshot=path.png`: renders the current state (topic list beside the selected feed) to an image, for
/// checking the layout without a screen.
@MainActor
enum SnapshotRenderer {
    static func write(model: TopicSortModel, to path: String) throws {
        let sections = model.feedSections
        let view = HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 9) {
                Text("Broad topics").font(.headline)
                ForEach(model.topics) { topic in
                    TopicRow(name: topic.name, count: topic.members.count, color: model.color(topic)) {
                        Text(topic.children.isEmpty ? "Split" : "Merge").font(.caption)
                            .padding(.horizontal, 7).padding(.vertical, 2)
                            .background(RoundedRectangle(cornerRadius: 5).fill(Color.primary.opacity(0.08)))
                    }
                    ForEach(topic.children) { child in
                        TopicRow(name: child.name, count: child.members.count, color: model.color(child))
                            .padding(.leading, 18)
                    }
                }
                Spacer()
            }
            .frame(width: 340).padding(14).background(Color(white: 0.95))
            VStack(alignment: .leading, spacing: 0) {
                Header().environmentObject(model)
                Divider()
                ForEach(sections.prefix(3), id: \.id) { section in
                    if let title = section.title, let color = section.color {
                        HStack {
                            Circle().fill(color).frame(width: 9, height: 9)
                            Text(title).font(.headline)
                            Text("\(section.items.count) posts").font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 16).padding(.vertical, 8)
                    }
                    ForEach(section.items.prefix(sections.count > 1 ? 1 : 4), id: \.self) { index in
                        let path = model.path(index)
                        TweetCard(post: model.posts[index], path: path, color: path.first.map { model.color($0) })
                        Divider()
                    }
                }
                Spacer()
            }
            .frame(width: 640).background(Color.white)
        }
        .frame(height: 1500)
        .environment(\.colorScheme, .light)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2
        guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
            let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:])
        else { throw BookmarkSortError.unavailable("Could not render the snapshot") }
        try png.write(to: URL(fileURLWithPath: path))
        print("snapshot → \(path)")
    }
}
