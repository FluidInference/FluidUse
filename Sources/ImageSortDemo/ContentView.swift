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
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 18) {
                titleBlock
                Spacer(minLength: 12)
                stats
                controls
            }
            VStack(alignment: .leading, spacing: 10) {
                titleBlock
                HStack(alignment: .center, spacing: 14) {
                    stats
                    Spacer(minLength: 8)
                    controls
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Sort photos").font(.system(size: 26, weight: .bold))
            Text("SigLIP 2 · Core ML on the Neural Engine · 37 breeds, zero-shot")
                .font(.callout).foregroundStyle(.secondary).lineLimit(1).fixedSize()
        }
    }

    private var stats: some View {
        HStack(spacing: 16) {
            stat("Sorted", "\(model.sorted) / \(model.total)")
            stat("Photos / s", model.sorted > 0 ? String(format: "%.0f", model.photosPerSecond) : "–")
            stat("Elapsed", String(format: "%.1f s", model.elapsed))
            stat("ms / photo", millisecondsPerPhoto)
            stat("Correct breed", model.accuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "–")
        }
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
                .lineLimit(1).fixedSize()
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
        HStack(alignment: .top, spacing: 18) {
            nowSorting.frame(width: 250)
            chart
        }
        .padding(14)
        .coordinateSpace(name: "board")
        .onPreferenceChange(FramesKey.self) { frames = $0 }
        .overlay(alignment: .topLeading) { flying }
    }

    /// Photos travelling from the Now Sorting panel to their tile; they shrink to tile size on arrival.
    private var flying: some View {
        ZStack(alignment: .topLeading) {
            if let from = frames["photo"], let chart = frames["chart"] {
                let scale = chart.width / CGFloat(PhotoChart.columns * PhotoChart.tile)
                ForEach(model.flights) { flight in
                    let target = CGRect(
                        x: chart.minX + flight.slot.minX * scale, y: chart.minY + flight.slot.minY * scale,
                        width: flight.slot.width * scale, height: flight.slot.height * scale)
                    let side = flight.arrived ? max(target.width, 4) : from.width * 0.55
                    Color.clear
                        .frame(width: side, height: side)
                        .overlay {
                            if let image = flight.image {
                                Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fill)
                            }
                        }
                        .clipShape(RoundedRectangle(cornerRadius: flight.arrived ? 1 : 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: flight.arrived ? 1 : 8)
                                .stroke(flight.wrong ? Color.red : Color.green, lineWidth: flight.arrived ? 1 : 3)
                        )
                        .shadow(color: .black.opacity(0.4), radius: flight.arrived ? 0 : 6)
                        .position(
                            x: flight.arrived ? target.midX : from.midX, y: flight.arrived ? target.midY : from.midY)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private var nowSorting: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("NOW SORTING · \(model.remaining) left").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Color.secondary.opacity(0.08)
                .aspectRatio(1, contentMode: .fit)
                .overlay {
                    if let image = model.currentImage {
                        Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fill)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .reportFrame("photo")
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(model.current.map { $0.matchesGold ? Color.green : Color.red } ?? .clear, lineWidth: 3))
            if let current = model.current {
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(Array(current.result.top.enumerated()), id: \.offset) { rank, entry in
                        topRow(entry.breed, share: entry.share, first: rank == 0, gold: current.item.breed)
                    }
                    if !current.matchesGold {
                        Text("label: \(current.item.breed)").font(.caption.weight(.semibold)).foregroundStyle(.red)
                    }
                }
            }
            Spacer(minLength: 0)
            Text("Labels are the 37 breed names, typed once: \"a photo of a {breed}, a type of pet.\"")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func topRow(_ breed: String, share: Float, first: Bool, gold: String) -> some View {
        HStack(spacing: 6) {
            Text(breed).font(first ? .callout.weight(.bold) : .caption).lineLimit(1)
                .frame(width: 128, alignment: .leading)
            GeometryReader { proxy in
                Capsule().fill(breed == gold ? Color.green : color(for: breed))
                    .frame(width: max(2, proxy.size.width * CGFloat(share)))
            }
            .frame(height: first ? 10 : 6)
            Text("\(Int(share * 100))%").font(.caption).monospacedDigit().frame(width: 34, alignment: .trailing)
        }
    }

    private var chart: some View {
        GeometryReader { proxy in
            let labelWidth: CGFloat = 190
            let width = CGFloat(PhotoChart.columns * PhotoChart.tile)
            let height = CGFloat(model.breeds.count * PhotoChart.rowHeight)
            let scale = min((proxy.size.width - labelWidth) / width, proxy.size.height / height)
            let rowHeight = CGFloat(PhotoChart.rowHeight) * scale
            HStack(alignment: .top, spacing: 0) {
                VStack(alignment: .trailing, spacing: 0) {
                    ForEach(model.breeds, id: \.self) { breed in
                        HStack(spacing: 6) {
                            Text(breed).lineLimit(1).foregroundStyle(color(for: breed))
                            Text("\(model.counts[breed] ?? 0)").monospacedDigit().foregroundStyle(.secondary)
                                .frame(width: 34, alignment: .trailing)
                        }
                        .font(.system(size: max(9, min(13, rowHeight * 0.55)), weight: .semibold))
                        .frame(width: labelWidth - 8, height: rowHeight, alignment: .trailing)
                        .padding(.trailing, 8)
                    }
                }
                ZStack(alignment: .topLeading) {
                    if let image = model.chartImage {
                        Image(decorative: image, scale: 1).resizable().interpolation(.medium)
                            .frame(width: width * scale, height: height * scale)
                    }
                    if let slot = model.lastSlot {
                        RoundedRectangle(cornerRadius: 3).stroke(Color.yellow, lineWidth: 2)
                            .frame(width: slot.width * scale + 6, height: slot.height * scale + 6)
                            .offset(x: slot.minX * scale - 3, y: slot.minY * scale - 3)
                    }
                }
                .frame(width: width * scale, height: height * scale, alignment: .topLeading)
                .reportFrame("chart")
            }
        }
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
