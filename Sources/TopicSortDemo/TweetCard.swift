import BookmarkSort
import SwiftUI

/// A post drawn like a timeline entry: avatar, name and handle, text, quoted post, media, action row, and the
/// topic it was sorted into.
struct TweetCard: View {
    let post: Bookmark
    let path: [TopicNode]
    let color: Color?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Avatar(name: post.author)
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 4) {
                    Text(Self.displayName(post.author)).font(.system(size: 14, weight: .bold)).lineLimit(1)
                    Text("@\(post.author) · \(Self.shortDate(post.createdAt))")
                        .font(.system(size: 14)).foregroundStyle(.secondary).lineLimit(1)
                    Spacer(minLength: 0)
                    Image(systemName: "ellipsis").foregroundStyle(.secondary)
                }
                Text(post.text).font(.system(size: 14)).lineLimit(9).fixedSize(horizontal: false, vertical: true)
                if let quoted = post.quotedText, !quoted.isEmpty {
                    Text(quoted).font(.system(size: 13)).foregroundStyle(.primary.opacity(0.8)).lineLimit(4)
                        .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.secondary.opacity(0.3)))
                }
                if !post.media.isEmpty {
                    RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.12)).frame(height: 70)
                        .overlay(
                            Label(
                                post.media.contains("video") ? "Video" : "Photo",
                                systemImage: post.media.contains("video") ? "play.rectangle" : "photo"
                            )
                            .font(.caption).foregroundStyle(.secondary))
                }
                actions
                if let color, !path.isEmpty { TopicChip(path: path, color: color) }
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
    }

    private var actions: some View {
        HStack {
            ForEach(["bubble.left", "arrow.2.squarepath", "heart", "chart.bar.xaxis", "bookmark"], id: \.self) {
                Image(systemName: $0)
                Spacer()
            }
            Image(systemName: "square.and.arrow.up")
        }
        .font(.system(size: 13)).foregroundStyle(.secondary).padding(.top, 2).padding(.trailing, 20)
    }

    /// `some_handle` → `Some Handle`; the export only has handles.
    static func displayName(_ handle: String) -> String {
        handle.split(separator: "_").map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined(separator: " ")
    }

    /// `Tue Sep 08 07:35:48 +0000 2026` → `Sep 8`; `2 hours ago` → `2h`.
    static func shortDate(_ value: String) -> String {
        let parts = value.split(separator: " ")
        if parts.count == 6, let day = Int(parts[2]) { return "\(parts[1]) \(day)" }
        if parts.count >= 3, parts.last == "ago", let amount = Int(parts[0]) {
            return "\(amount)\(parts[1].first.map(String.init) ?? "")"
        }
        return value
    }
}

struct TopicChip: View {
    let path: [TopicNode]
    let color: Color

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: "folder.fill")
            Text(path.map(\.name).joined(separator: "  ›  ")).lineLimit(1)
        }
        .font(.system(size: 11, weight: .semibold)).foregroundStyle(color)
        .padding(.horizontal, 9).padding(.vertical, 4)
        .background(Capsule().fill(color.opacity(0.13)))
    }
}

struct Avatar: View {
    let name: String

    var body: some View {
        let hue = Double(name.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0xFFFF } % 360) / 360
        Circle().fill(Color(hue: hue, saturation: 0.45, brightness: 0.8)).frame(width: 40, height: 40)
            .overlay(Text(name.prefix(1).uppercased()).font(.system(size: 17, weight: .bold)).foregroundStyle(.white))
    }
}
