import ImageSort
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: VideoSortModel

    private var scene: VideoSortModel.Scene { model.scene }

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
                if scene.isGrid {
                    HStack(alignment: .top, spacing: 16) {
                        video
                        gridLegend.frame(width: 240)
                    }
                    .padding(14)
                } else {
                    VStack(spacing: 12) {
                        HStack(alignment: .top, spacing: 16) {
                            video
                            topFive.frame(width: 290)
                        }
                        spotted
                    }
                    .padding(14)
                }
            }
            Divider()
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: Header

    private var header: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 18) {
                titleBlock
                Spacer(minLength: 12)
                stats
                button
            }
            VStack(alignment: .leading, spacing: 10) {
                titleBlock
                HStack(spacing: 14) {
                    stats
                    Spacer(minLength: 8)
                    button
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(scene.title).font(.system(size: 26, weight: .bold))
            Text(subtitle).font(.callout).foregroundStyle(.secondary).lineLimit(1).fixedSize()
        }
    }

    private var subtitle: String {
        scene.isGrid
            ? "SigLIP 2 · Core ML on the Neural Engine · \(scene.columns * scene.rows) cells per frame, zero-shot"
            : "SigLIP 2 · Core ML on the Neural Engine · \(scene.labels.count) animals, zero-shot, every frame"
    }

    private var stats: some View {
        HStack(spacing: 16) {
            stat(
                scene.isGrid ? "Frames / s" : "Frames / s",
                model.phase == .running ? String(format: "%.0f", model.framesPerSecond) : "–")
            stat(
                scene.isGrid ? "ms / frame" : "ms / frame",
                model.grid.map { String(format: "%.1f", $0.milliseconds) } ?? "–")
            stat("Frames labeled", "\(model.framesLabeled)")
            if !scene.isGrid {
                stat("Correct frames", model.accuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "–")
                stat("Species", "\(model.speciesSpotted) / \(scene.labels.count)")
            }
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                .lineLimit(1).fixedSize()
            Text(value).font(.system(size: 20, weight: .semibold, design: .rounded)).monospacedDigit()
                .contentTransition(.numericText()).fixedSize()
        }
    }

    private var button: some View {
        Button(model.phase == .running ? "Stop" : "Start") { model.toggle() }
            .keyboardShortcut(.space, modifiers: [])
            .buttonStyle(.borderedProminent)
            .disabled(!(model.phase == .ready || model.phase == .running))
    }

    // MARK: Video

    private var video: some View {
        Color.black
            .aspectRatio(16 / 9, contentMode: .fit)
            .overlay {
                if let frame = model.frame {
                    Image(decorative: frame, scale: 1).resizable().aspectRatio(contentMode: .fit)
                }
            }
            .overlay { if model.phase == .running { overlay } }
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    @ViewBuilder
    private var overlay: some View {
        if let grid = model.grid {
            if scene.isGrid {
                GeometryReader { proxy in
                    let width = proxy.size.width / CGFloat(scene.columns)
                    let height = proxy.size.height / CGFloat(scene.rows)
                    ForEach(grid.cells.indices, id: \.self) { index in
                        let cell = grid.cells[index]
                        let label = scene.labels[cell.label]
                        ZStack(alignment: .topLeading) {
                            Rectangle().fill(label.color.opacity(0.12))
                            Rectangle().stroke(label.color, lineWidth: 2)
                            Text("\(label.name) \(Int(cell.share * 100))%")
                                .font(.system(size: max(10, min(15, width / 13)), weight: .bold))
                                .foregroundStyle(label.color).lineLimit(1)
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .background(Color.black.opacity(0.6))
                        }
                        .frame(width: width, height: height)
                        .offset(x: CGFloat(index % scene.columns) * width, y: CGFloat(index / scene.columns) * height)
                    }
                }
            } else if let cell = grid.cells.first {
                let label = scene.labels[cell.label]
                let correct = model.truth.map { $0 == cell.label }
                VStack {
                    Spacer()
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(label.name).font(.system(size: 44, weight: .heavy, design: .rounded))
                        Text("\(Int(cell.share * 100))%").font(.system(size: 28, weight: .bold, design: .rounded))
                            .monospacedDigit()
                        if correct == false {
                            Image(systemName: "xmark.circle.fill").foregroundStyle(.red).font(.system(size: 26))
                        }
                    }
                    .foregroundStyle(label.color)
                    .padding(.horizontal, 18).padding(.vertical, 8)
                    .background(Capsule().fill(.black.opacity(0.65)))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(18)
                }
            }
        }
    }

    // MARK: Side panels

    private var topFive: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("TOP 5 · THIS FRAME").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            if let cell = model.grid?.cells.first, model.phase == .running {
                ForEach(Array(cell.top.enumerated()), id: \.offset) { rank, entry in
                    let label = scene.labels[entry.label]
                    HStack(spacing: 8) {
                        Text(label.name).font(rank == 0 ? .title3.weight(.bold) : .callout)
                            .foregroundStyle(rank == 0 ? label.color : .primary).lineLimit(1)
                            .frame(width: 120, alignment: .leading)
                        GeometryReader { proxy in
                            Capsule().fill(label.color.opacity(rank == 0 ? 1 : 0.6))
                                .frame(width: max(2, proxy.size.width * CGFloat(entry.share)))
                        }
                        .frame(height: rank == 0 ? 12 : 7)
                        Text("\(Int(entry.share * 100))%").font(.callout).monospacedDigit()
                            .frame(width: 42, alignment: .trailing)
                    }
                }
            }
            Spacer(minLength: 0)
            Text(
                "Labels are \(scene.labels.count) animal names, typed once: \"a photo of a {animal}.\" Nothing is trained on this video."
            )
            .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var spotted: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("SPOTTED · \(model.spots.count)").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ScrollViewReader { reader in
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(model.spots) { spot in
                            let label = scene.labels[spot.label]
                            VStack(spacing: 3) {
                                Color.secondary.opacity(0.1)
                                    .frame(width: 92, height: 92)
                                    .overlay {
                                        if let image = spot.thumbnail {
                                            Image(decorative: image, scale: 1).resizable().aspectRatio(
                                                contentMode: .fill)
                                        }
                                    }
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(
                                                spot.correct.map { $0 ? Color.green : Color.red } ?? label.color,
                                                lineWidth: 2.5))
                                Text(label.name).font(.caption.weight(.semibold)).foregroundStyle(label.color)
                                    .lineLimit(1)
                                Text(String(format: "%.1f s · %d%%", spot.seconds, Int(spot.share * 100)))
                                    .font(.caption2).foregroundStyle(.secondary).monospacedDigit()
                            }
                            .frame(width: 96)
                            .id(spot.id)
                            .transition(.scale.combined(with: .opacity))
                        }
                    }
                    .animation(.spring(duration: 0.35), value: model.spots.count)
                }
                .onChange(of: model.spots.count) { _, _ in
                    if let last = model.spots.last { withAnimation { reader.scrollTo(last.id, anchor: .trailing) } }
                }
            }
            .frame(height: 132)
        }
    }

    private var gridLegend: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("CELLS IN THIS FRAME").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            let counts = model.counts
            let total = max(1, counts.reduce(0, +))
            ForEach(Array(scene.labels.enumerated()), id: \.offset) { index, label in
                HStack(spacing: 6) {
                    Text(label.name).font(.callout.weight(counts[index] > 0 ? .semibold : .regular))
                        .foregroundStyle(counts[index] > 0 ? label.color : .secondary).lineLimit(1)
                        .frame(width: 130, alignment: .leading)
                    GeometryReader { proxy in
                        Capsule().fill(label.color)
                            .frame(width: max(2, proxy.size.width * CGFloat(counts[index]) / CGFloat(total)))
                    }
                    .frame(height: 8)
                    Text("\(counts[index])").font(.callout).monospacedDigit().frame(width: 24, alignment: .trailing)
                }
            }
            Spacer(minLength: 0)
            Text("Labels are plain phrases, typed once: \"a photo of {label}.\" Nothing is trained on this video.")
                .font(.caption).foregroundStyle(.secondary)
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
            Text(scene.credit)
            Spacer()
            Text("Model: google/siglip2-base-patch16-256 (Apache-2.0), converted to Core ML by FluidInference")
        }
        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
        .padding(.horizontal, 20).padding(.vertical, 8)
    }
}
