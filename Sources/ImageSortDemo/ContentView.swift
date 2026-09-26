import ImageSort
import SwiftUI

private struct FramesKey: PreferenceKey {
    static let defaultValue: [String: CGRect] = [:]
    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue()) { $1 }
    }
}

extension View {
    fileprivate func reportFrame(_ key: String) -> some View {
        background(
            GeometryReader { proxy in
                Color.clear.preference(key: FramesKey.self, value: [key: proxy.frame(in: .named("board"))])
            })
    }
}

private let incomingKey = "__incoming"

struct ContentView: View {
    @EnvironmentObject private var model: ImageSortModel
    @State private var frames: [String: CGRect] = [:]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            switch model.phase {
            case .loading(let message):
                status(message, spinning: true)
            case .failed(let message):
                status("Failed: \(message)", spinning: false)
            default:
                board
            }
            Divider()
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 18) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Sort photos").font(.system(size: 26, weight: .bold))
                Text("SigLIP 2 · Core ML on the Neural Engine · 37 breeds, zero-shot, nothing trained on these photos")
                    .font(.callout).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 12)
            HStack(spacing: 16) {
                stat("Sorted", "\(model.sorted) / \(model.total)")
                stat("Photos / s", model.sorted > 0 ? String(format: "%.0f", model.photosPerSecond) : "–")
                stat("Elapsed", String(format: "%.1f s", model.elapsed))
                stat("ms / photo", millisecondsPerPhoto)
                stat("Correct breed", model.accuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "–")
            }
            controls
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    /// Show: one model call, pre- and post-processing included. Turbo: wall time per photo with calls overlapping.
    private var millisecondsPerPhoto: String {
        guard model.sorted > 0 else { return "–" }
        if model.mode == .turbo { return String(format: "%.1f", 1000 / model.photosPerSecond) }
        return model.medianMilliseconds.map { String(format: "%.1f", $0) } ?? "–"
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 20, weight: .semibold, design: .rounded)).monospacedDigit()
                .contentTransition(.numericText()).fixedSize()
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button(model.phase == .running ? "Pause" : "Start") { model.toggleRun() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(!(model.phase == .ready || model.phase == .running || model.phase == .paused))
                    .buttonStyle(.borderedProminent)
                Button("Reset") { model.reset() }.disabled(model.phase == .running)
            }
            Picker("Mode", selection: $model.mode) {
                ForEach(ImageSortModel.Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().frame(width: 150)
            if model.mode == .show {
                HStack(spacing: 6) {
                    Text("Pace").font(.caption).fixedSize()
                    Slider(value: $model.pace, in: 2...30).frame(width: 90)
                    Text(String(format: "%.0f/s", model.pace)).font(.caption).monospacedDigit().fixedSize()
                }
            } else {
                Text("\(ImageSortModel.turboInFlight) calls in flight").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var board: some View {
        GeometryReader { outer in
            let incomingWidth = min(260, max(190, outer.size.width * 0.17))
            HStack(alignment: .top, spacing: 14) {
                incoming.frame(width: incomingWidth)
                GeometryReader { proxy in
                    let columns = max(4, min(8, Int(proxy.size.width / 150)))
                    let rows = (model.breeds.count + columns - 1) / columns
                    let height = max(78, (proxy.size.height - CGFloat(rows - 1) * 8) / CGFloat(rows))
                    ScrollView {
                        LazyVGrid(
                            columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: columns), spacing: 8
                        ) {
                            ForEach(model.breeds, id: \.self) { breed in
                                BucketView(
                                    name: breed, placed: model.buckets[breed] ?? [], count: model.counts[breed] ?? 0,
                                    color: color(for: breed), height: height
                                )
                                .reportFrame(breed)
                            }
                        }
                    }
                }
            }
        }
        .padding(14)
        .coordinateSpace(name: "board")
        .onPreferenceChange(FramesKey.self) { frames = $0 }
        .overlay(alignment: .topLeading) { flying }
    }

    private var incoming: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("INCOMING · \(model.queue.count) left").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ZStack {
                RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.06))
                if model.incomingImage != nil { PhotoView(image: model.incomingImage).padding(8) }
            }
            .aspectRatio(1, contentMode: .fit)
            .reportFrame(incomingKey)
            Text("Labels are just the 37 breed names, typed once:\n\"a photo of a {breed}, a type of pet.\"")
                .font(.caption).foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
    }

    private var flying: some View {
        ZStack(alignment: .topLeading) {
            ForEach(model.flights) { flight in
                if let from = frames[incomingKey], let to = frames[flight.placed.result.breed] {
                    let target = flight.arrived ? to : from
                    VStack(spacing: 4) {
                        PhotoView(image: flight.placed.thumbnail)
                        Text("→ \(flight.placed.result.breed)").font(.headline)
                            .foregroundStyle(flight.placed.matchesGold ? Color.green : Color.red)
                    }
                    .padding(6)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
                    .frame(width: model.mode == .turbo ? 130 : from.width - 16)
                    .scaleEffect(flight.arrived ? 0.3 : 1)
                    .opacity(flight.arrived ? 0.2 : 1)
                    .position(x: target.midX, y: target.midY)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func status(_ message: String, spinning: Bool) -> some View {
        VStack(spacing: 12) {
            if spinning { ProgressView() }
            Text(message).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var footer: some View {
        HStack {
            Text(PetsSample.attribution)
            Spacer()
            Text("Model: google/siglip2-base-patch16-256 (Apache-2.0), converted to Core ML by FluidInference")
        }
        .font(.caption).foregroundStyle(.secondary)
        .padding(.horizontal, 20).padding(.vertical, 8)
    }

    private func color(for breed: String) -> Color { ImageSortModel.cats.contains(breed) ? .orange : .blue }
}

private struct PhotoView: View {
    let image: CGImage?

    var body: some View {
        Color.secondary.opacity(0.1)
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let image { Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fill) }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct BucketView: View {
    let name: String
    let placed: [ImageSortModel.Placed]
    let count: Int
    let color: Color
    let height: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(name).font(.callout.weight(.semibold)).lineLimit(1).minimumScaleFactor(0.7)
                Spacer(minLength: 2)
                Text("\(count)").font(.headline).monospacedDigit().contentTransition(.numericText())
            }
            GeometryReader { proxy in
                let side = max(20, min(proxy.size.height, (proxy.size.width - 8) / 3))
                HStack(spacing: 4) {
                    ForEach(placed.prefix(3)) { entry in
                        Image(decorative: entry.thumbnail ?? blank, scale: 1).resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: side, height: side).clipped()
                            .clipShape(RoundedRectangle(cornerRadius: 5))
                            .overlay(
                                RoundedRectangle(cornerRadius: 5)
                                    .stroke(entry.matchesGold ? Color.green : Color.red, lineWidth: 2)
                            )
                            .help("Gold breed: \(entry.item.breed)")
                    }
                }
            }
        }
        .padding(8)
        .frame(height: height)
        .background(RoundedRectangle(cornerRadius: 10).fill(color.opacity(0.10)))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(color.opacity(0.35), lineWidth: 1))
    }

    private var blank: CGImage {
        CGContext(
            data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
    }
}
