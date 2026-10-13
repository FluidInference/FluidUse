import AppKit
import FluidUse
import SwiftUI
import UniformTypeIdentifiers

@available(macOS 15.0, *)
struct ContentView: View {
    @EnvironmentObject var model: ReaderModel
    @State private var dropTargeted = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                pagesColumn
                    .frame(width: 420)
                    .background(Color(nsColor: .underPageBackgroundColor))
                Divider()
                outputColumn
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url, url.pathExtension.lowercased() == "pdf" else { return }
                Task { @MainActor in model.open(url) }
            }
            return true
        }
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 12).stroke(Color.accentColor, lineWidth: 4).padding(6)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text("A whole PDF, read in one pass — on this Mac")
                    .font(.system(size: 22, weight: .semibold))
                Text("Baidu Unlimited-OCR · 3B MoE · MLX + Core ML\(model.turbo ? " (turbo)" : "") · offline")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            statusView
            HStack(spacing: 8) {
                Toggle(
                    "Turbo",
                    isOn: Binding(get: { model.turbo }, set: { on in Task { await model.setTurbo(on) } })
                )
                .toggleStyle(.switch)
                .help("Routed experts in 4-bit: ~15% faster, small differences in formulas and tables")
                .disabled(model.canPause || model.phase == .loading)
                Button("Open PDF…") { openPanel() }
                    .disabled(model.phase == .loading || model.canPause)
                Button(model.hasStarted ? "Restart" : "Start") { model.start() }
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(!model.canStart)
                Button(model.phase == .paused ? "Resume" : "Pause") { model.togglePause() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(!model.canPause)
                    .frame(minWidth: 76)
            }
            .controlSize(.large)
            .focusable(false)
            .fixedSize()  // controls keep their width; the title wraps instead
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var statusView: some View {
        HStack(spacing: 18) {
            metric(value: "\(model.pageImages.count)", label: "pages")
            metric(value: "\(model.outputTokens)", label: "tokens")
            metric(value: String(format: "%.0f", model.tokensPerSecond), label: "tokens / s")
            metric(value: String(format: "%.1f s", model.elapsed), label: phaseLabel)
        }
    }

    private var phaseLabel: String {
        switch model.phase {
        case .loading: "loading model"
        case .ready: "drop a PDF"
        case .loaded: "press Start"
        case .encoding: "looking at pages"
        case .prefilling: "reading pages in"
        case .reading: "writing"
        case .paused: "paused"
        case .done: "done"
        case .failed: "error"
        }
    }

    private func metric(value: String, label: String) -> some View {
        VStack(alignment: .trailing, spacing: 0) {
            Text(value).font(.system(size: 20, weight: .semibold, design: .rounded)).monospacedDigit()
            Text(label).font(.system(size: 11)).foregroundStyle(.secondary)
        }
        .frame(minWidth: 70, alignment: .trailing)
    }

    // MARK: Pages with layout boxes

    private var currentPage: Int { max(0, model.document.pages.count - 1) }

    private var pagesColumn: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 14) {
                    ForEach(Array(model.pageImages.enumerated()), id: \.offset) { index, image in
                        pageView(index: index, image: image).id(index)
                    }
                }
                .padding(14)
            }
            .onChange(of: model.document.pages.count) { _, count in
                withAnimation(.easeInOut(duration: 0.4)) { proxy.scrollTo(max(0, count - 1), anchor: .top) }
            }
        }
    }

    private func pageView(index: Int, image: NSImage) -> some View {
        let blocks = index < model.document.pages.count ? model.document.pages[index].blocks : []
        let reading = (model.phase == .reading || model.phase == .paused) && index == currentPage
        return Image(nsImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .overlay {
                GeometryReader { geo in
                    ForEach(Array(blocks.enumerated()), id: \.element.id) { i, block in
                        let isCurrent = reading && i == blocks.count - 1
                        ForEach(Array(block.boxes.enumerated()), id: \.offset) { _, box in
                            Rectangle()
                                .fill(isCurrent ? Color.accentColor.opacity(0.18) : Color.green.opacity(0.06))
                                .overlay(
                                    Rectangle().stroke(
                                        isCurrent ? Color.accentColor : Color.green.opacity(0.5),
                                        lineWidth: isCurrent ? 2.5 : 1)
                                )
                                .frame(width: box.width * geo.size.width, height: box.height * geo.size.height)
                                .position(x: box.midX * geo.size.width, y: box.midY * geo.size.height)
                        }
                    }
                }
            }
            .shadow(radius: 2)
            .opacity(index <= currentPage || !model.canPause ? 1 : 0.55)
    }

    // MARK: Markdown output

    private var outputColumn: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if model.document.pages.isEmpty {
                        placeholder
                    }
                    ForEach(model.document.pages) { page in
                        HStack {
                            Text("Page \(page.id + 1)").font(.system(size: 12, weight: .semibold)).foregroundStyle(
                                .secondary)
                            VStack { Divider() }
                        }
                        .padding(.top, page.id == 0 ? 0 : 12)
                        ForEach(page.blocks) { block in
                            blockView(block)
                        }
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
            }
            .onChange(of: model.outputTokens) { _, _ in
                if model.phase == .reading { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch model.phase {
            case .loading:
                ProgressView().controlSize(.small)
                Text(model.status).foregroundStyle(.secondary)
            case .failed(let message):
                Text(message).foregroundStyle(.red)
            case .loaded:
                Text(
                    "Press Start to read \(model.pageImages.count) page\(model.pageImages.count == 1 ? "" : "s") in one pass."
                )
                .foregroundStyle(.secondary)
            case .encoding, .prefilling:
                ProgressView().controlSize(.small)
                Text("Taking in all \(model.pageImages.count) pages…").foregroundStyle(.secondary)
            default:
                Text("Drop a PDF here (up to \(model.maxPages) pages per read).").foregroundStyle(.secondary)
            }
        }
        .font(.system(size: 15))
        .padding(.top, 40)
    }

    @ViewBuilder
    private func blockView(_ block: UnlimitedOcrDocument.Block) -> some View {
        switch block.label {
        case "title":
            MathText(content: block.content, size: 20).font(.system(size: 20, weight: .bold))
        case "sub_title":
            MathText(content: block.content, size: 16).font(.system(size: 16, weight: .semibold))
        case "equation", "formula":
            EquationView(content: block.content)
        case "table":
            TableView(content: block.content)
        case "code":
            Text(block.content)
                .font(.system(size: 12, design: .monospaced))
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
        case "image", "figure", "chart":
            Label("figure", systemImage: "photo").font(.system(size: 12)).foregroundStyle(.secondary)
        case "image_caption", "table_caption", "footnote", "page_footnote", "page_footer", "page_header", "header",
            "footer":
            MathText(content: block.content, size: 12).font(.system(size: 12)).foregroundStyle(.secondary)
        default:
            MathText(content: block.content, size: 14).font(.system(size: 14)).lineSpacing(2)
        }
    }

    private func openPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.pdf]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { model.open(url) }
    }
}
