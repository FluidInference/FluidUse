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
        GeometryReader { proxy in
            // Narrow windows merge the header and the current photo into one band so the chart gets the rest.
            let narrow = proxy.size.width < 980
            VStack(spacing: 0) {
                if narrow { compactTop } else { header }
                Divider()
                switch model.phase {
                case .loading(let message):
                    status(message, spinning: true)
                case .failed(let message):
                    status("Failed: \(message)", spinning: false)
                default:
                    board(narrow: narrow)
                }
                Divider()
                footer
            }
            .coordinateSpace(name: "board")
            .onPreferenceChange(FramesKey.self) { frames = $0 }
            .overlay(alignment: .topLeading) { flying }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var compactTop: some View {
        HStack(alignment: .top, spacing: 16) {
            photo.frame(width: 176, height: 176)
            VStack(alignment: .leading, spacing: 10) {
                titleBlock
                stats(size: 17)
                topFive.frame(maxWidth: 380, alignment: .leading)
            }
            Spacer(minLength: 8)
            controls
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var header: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 18) {
                titleBlock
                Spacer(minLength: 12)
                stats()
                controls
            }
            VStack(alignment: .leading, spacing: 10) {
                titleBlock
                HStack(alignment: .center, spacing: 14) {
                    stats()
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

    private func stats(size: CGFloat = 20) -> some View {
        HStack(spacing: size < 20 ? 12 : 16) {
            stat("Sorted", "\(model.sorted) / \(model.total)", size: size)
            stat("Elapsed", String(format: "%.1f s", model.elapsed), size: size)
            stat("Photos / s", model.sorted > 0 ? String(format: "%.0f", model.photosPerSecond) : "–", size: size)
            stat("ms per photo", millisecondsPerPhoto, size: size)
            stat("Correct breed", model.accuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "–", size: size)
        }
    }

    /// Wall time per photo with several model calls overlapping.
    private var millisecondsPerPhoto: String {
        model.sorted > 0 ? String(format: "%.1f", 1000 / model.photosPerSecond) : "–"
    }

    private func stat(_ title: String, _ value: String, size: CGFloat) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                .lineLimit(1).fixedSize()
            Text(value).font(.system(size: size, weight: .semibold, design: .rounded)).monospacedDigit()
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
            Text("\(ImageSortModel.turboInFlight) photos processed in parallel").font(.caption).foregroundStyle(
                .secondary)
        }
    }

    private func board(narrow: Bool) -> some View {
        Group {
            if narrow {
                chart(labelWidth: 150)
            } else {
                HStack(alignment: .top, spacing: 18) {
                    nowSorting.frame(width: 250)
                    chart(labelWidth: 190)
                }
            }
        }
        .padding(14)
    }

    /// Photos travelling from the Now Sorting panel to their tile; they shrink to tile size on arrival.
    private var flying: some View {
        ZStack(alignment: .topLeading) {
            if let from = frames["photo"], let chart = frames["chart"] {
                let scale = chart.width / CGFloat(model.chartGeometry.width)
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

    private var photo: some View {
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
    }

    @ViewBuilder
    private var topFive: some View {
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
    }

    private var nowSorting: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("NOW SORTING · \(model.remaining) left").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            photo
            topFive
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

    private func chart(labelWidth: CGFloat) -> some View {
        GeometryReader { proxy in
            let geometry = model.chartGeometry
            let width = CGFloat(geometry.width)
            let height = CGFloat(geometry.height)
            let available = CGSize(width: proxy.size.width - labelWidth, height: proxy.size.height)
            let scale = min(available.width / width, available.height / height)
            let rowHeight = CGFloat(geometry.rowHeight) * scale
            HStack(alignment: .top, spacing: 0) {
                VStack(alignment: .trailing, spacing: 0) {
                    ForEach(model.breeds, id: \.self) { breed in
                        HStack(spacing: 6) {
                            Text(breed).lineLimit(1).minimumScaleFactor(0.7).foregroundStyle(color(for: breed))
                            Text("\(model.counts[breed] ?? 0)").monospacedDigit().foregroundStyle(.secondary)
                                .frame(width: 30, alignment: .trailing)
                        }
                        .font(.system(size: max(8, min(13, rowHeight * 0.6)), weight: .semibold))
                        .frame(width: labelWidth - 8, height: rowHeight, alignment: .trailing)
                        .padding(.trailing, 8)
                    }
                }
                ZStack(alignment: .topLeading) {
                    if let image = model.chartImage {
                        Image(decorative: image, scale: 1).resizable().interpolation(.medium)
                            .frame(width: width * scale, height: height * scale)
                    }
                }
                .frame(width: width * scale, height: height * scale, alignment: .topLeading)
                .reportFrame("chart")
            }
            .onAppear { model.fitChart(to: available) }
            .onChange(of: available) { _, size in model.fitChart(to: size) }
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
