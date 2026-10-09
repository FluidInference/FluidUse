import BookmarkSort
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: BookmarkSortModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            switch model.phase {
            case .loading(let message), .waitingForPage(let message):
                Spacer()
                ProgressView(message).controlSize(.large)
                Spacer()
            case .failed(let message):
                Spacer()
                Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.red).padding()
                Spacer()
            case .watching:
                feed
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("Sort bookmarks").font(.title2.bold())
                Spacer()
                Text(model.pageTitle).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                Toggle("Auto-scroll", isOn: $model.autoScroll).toggleStyle(.switch).controlSize(.small)
            }
            HStack(spacing: 10) {
                tile("\(model.newCount)", "new posts sorted")
                tile("\(model.filedCount)", "filed on its own", tint: .green)
                tile("\(model.pendingCount)", "waiting for you", tint: .orange)
                tile(String(format: "%.0f ms", model.medianMilliseconds), "per post")
            }
            Text(model.modeDescription)
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(16)
    }

    private func tile(_ value: String, _ caption: String, tint: Color = .primary) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.system(size: 26, weight: .semibold, design: .rounded)).foregroundStyle(tint)
                .monospacedDigit()
            Text(caption).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }

    private var feed: some View {
        ScrollView {
            LazyVStack(spacing: 10) {
                ForEach(model.items) { item in
                    PostCard(item: item, visible: model.onPage.contains(item.id))
                }
            }
            .padding(16)
            .animation(.easeOut(duration: 0.25), value: model.items.map(\.id))
        }
    }
}

private struct PostCard: View {
    @EnvironmentObject private var model: BookmarkSortModel
    let item: BookmarkSortModel.Sorted
    let visible: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle().fill(visible ? Color.accentColor : .clear).frame(width: 7, height: 7)
                Text("@\(item.post.author)").font(.callout.bold())
                Spacer()
                Text(String(format: "%.1f ms", item.result.milliseconds)).font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(item.post.text.replacingOccurrences(of: "\n", with: " "))
                .font(.callout).lineLimit(3).foregroundStyle(.primary.opacity(0.85))
            decision
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.primary.opacity(0.04)))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(border, lineWidth: 1))
    }

    private var border: Color {
        switch item.status {
        case .filed: .green.opacity(0.5)
        case .suggested: .orange.opacity(0.5)
        case .confirmed: .accentColor.opacity(0.6)
        case .known: .clear
        }
    }

    @ViewBuilder private var decision: some View {
        let result = item.result
        switch item.status {
        case .filed:
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                destination(result.folder, result.category)
                Spacer()
                confidence(result.categoryConfidence ?? result.folderConfidence)
            }
        case .confirmed(let category):
            HStack(spacing: 6) {
                Image(systemName: "person.crop.circle.badge.checkmark").foregroundStyle(Color.accentColor)
                Text(BookmarkSortModel.short(category)).font(.callout.bold())
                Text("· filed by you").font(.caption).foregroundStyle(.secondary)
            }
        case .known(let category):
            HStack(spacing: 6) {
                Image(systemName: "folder").foregroundStyle(.secondary)
                Text("Already filed: \(BookmarkSortModel.short(category))").font(.caption).foregroundStyle(.secondary)
            }
        case .suggested:
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "questionmark.circle.fill").foregroundStyle(.orange)
                    Text(result.folder).font(.callout.bold())
                    Text("· pick one").font(.caption).foregroundStyle(.secondary)
                }
                let options =
                    result.categories.isEmpty
                    ? [(name: result.folder, probability: result.folderConfidence)]
                    : Array(result.categories.prefix(3))
                HStack(spacing: 6) {
                    ForEach(options, id: \.name) { option in
                        Button {
                            model.confirm(item.id, category: option.name)
                        } label: {
                            HStack(spacing: 4) {
                                Text(BookmarkSortModel.short(option.name)).lineLimit(1)
                                Text(String(format: "%.0f%%", option.probability * 100)).foregroundStyle(.secondary)
                                    .monospacedDigit()
                            }
                            .font(.caption)
                        }
                        .buttonStyle(.bordered)
                    }
                    Menu("Other") {
                        ForEach(model.folders, id: \.folder) { group in
                            Section(group.folder) {
                                ForEach(group.categories, id: \.self) { category in
                                    Button(BookmarkSortModel.short(category)) {
                                        model.confirm(item.id, category: category)
                                    }
                                }
                            }
                        }
                    }
                    .font(.caption).fixedSize()
                }
            }
        }
    }

    private func destination(_ folder: String, _ category: String?) -> some View {
        HStack(spacing: 4) {
            Text(folder).font(.callout).foregroundStyle(.secondary)
            if let category {
                Text("›").foregroundStyle(.secondary)
                Text(BookmarkSortModel.short(category)).font(.callout.bold())
                if let english = BookmarkSortModel.english(category) {
                    Text(english).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
            }
        }
    }

    private func confidence(_ value: Float) -> some View {
        Text(String(format: "%.0f%%", value * 100)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
    }
}
