import SwiftUI

@available(macOS 15.0, *)
struct ContentView: View {
    @EnvironmentObject private var model: WriterModel

    var body: some View {
        VStack(spacing: 0) {
            Header()
            Divider()
            HSplitView {
                TaskList().frame(minWidth: 340, idealWidth: 400, maxWidth: 520)
                VStack(spacing: 0) {
                    Editor()
                    Divider()
                    Console().frame(height: 150)
                }
                .frame(minWidth: 520)
            }
            Divider()
            CustomTaskBar()
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

@available(macOS 15.0, *)
struct Header: View {
    @EnvironmentObject private var model: WriterModel

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Python writer").font(.title2.bold())
                Text("Qwen2.5-Coder 0.5B · Neural Engine + GPU · offline, nothing leaves this Mac")
                    .font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 14) {
                    Link(destination: URL(string: "https://github.com/FluidInference/FluidUse")!) {
                        Label(
                            "github.com/FluidInference/FluidUse", systemImage: "chevron.left.forwardslash.chevron.right"
                        )
                    }
                    Link(destination: URL(string: "https://huggingface.co/FluidInference/qwen2.5-coder-0.5b-coreml")!) {
                        Label("huggingface.co/FluidInference/qwen2.5-coder-0.5b-coreml", systemImage: "shippingbox")
                    }
                }
                .font(.caption.weight(.medium))
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(model.passedCount)").font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(.green).contentTransition(.numericText())
                    Text("/ \(model.doneCount) written pass their tests")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .contentTransition(.numericText())
                }
                Text(status).font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Button {
                    model.play()
                } label: {
                    Label(model.paused ? "Resume" : "Play", systemImage: "play.fill")
                }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(!model.canPlay)
                Button {
                    model.pause()
                } label: {
                    Label("Pause", systemImage: "pause.fill")
                }
                .disabled(!model.isRunning || model.paused)
                Button {
                    model.replay()
                } label: {
                    Label("Replay", systemImage: "arrow.counterclockwise")
                }
                .keyboardShortcut("r", modifiers: [.command])
                .disabled(model.writerMissing)
            }
        }
        .padding(16)
    }

    private var status: String {
        let speed = model.tokensPerSecond > 0 ? String(format: " · %.0f tokens/s", model.tokensPerSecond) : ""
        switch model.step {
        case .loading(let message): return message
        case .idle: return model.paused ? "Paused" + speed : "Ready · press Play (⌘↩)"
        case .writing: return (model.paused ? "Pausing after this task" : "Writing…") + speed
        case .finished: return "Done · \(model.tasks.count) tasks" + speed
        case .failed(let message): return message
        }
    }
}

@available(macOS 15.0, *)
struct TaskList: View {
    @EnvironmentObject private var model: WriterModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(model.tasks) { task in
                        TaskRow(
                            task: task, status: model.statuses[task.id] ?? .pending, active: model.current == task.id
                        )
                        .id(task.id)
                        .onTapGesture { model.select(task) }
                        Divider().opacity(0.4)
                    }
                }
            }
            .onChange(of: model.current) {
                if let current = model.current { withAnimation { proxy.scrollTo(current, anchor: .center) } }
            }
        }
    }
}

@available(macOS 15.0, *)
struct TaskRow: View {
    let task: CodingTask
    let status: WriterModel.Status
    let active: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            StatusIcon(status: status).frame(width: 22)
            VStack(alignment: .leading, spacing: 4) {
                Text(task.topic).font(.system(size: 10, weight: .bold, design: .rounded)).foregroundStyle(.cyan)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(Color.cyan.opacity(0.15)))
                Text(task.prompt).font(.system(size: 13)).foregroundStyle(.primary.opacity(0.9))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14).padding(.vertical, 9)
        .background(active ? Color.accentColor.opacity(0.14) : .clear)
        .overlay(alignment: .leading) { if active { Rectangle().fill(Color.accentColor).frame(width: 3) } }
        .contentShape(Rectangle())
    }
}

@available(macOS 15.0, *)
struct StatusIcon: View {
    let status: WriterModel.Status

    var body: some View {
        switch status {
        case .pending:
            Image(systemName: "circle").foregroundStyle(.secondary)
        case .writing:
            Image(systemName: "pencil.line").foregroundStyle(Color.accentColor).symbolEffect(.pulse)
        case .testing:
            ProgressView().controlSize(.small)
        case .passed:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundStyle(.orange)
        case .broken:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
        }
    }
}

@available(macOS 15.0, *)
struct Editor: View {
    @EnvironmentObject private var model: WriterModel

    static let background = Color(red: 0.12, green: 0.12, blue: 0.14)
    static let gutterBackground = Color(red: 0.10, green: 0.10, blue: 0.12)
    private let lineHeight: CGFloat = 21

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            TabBar(title: headerTitle)
            let lines = PythonHighlighter.lines(model.code)
            let typing = model.isWriting
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                            let isLast = index == lines.count - 1
                            HStack(spacing: 0) {
                                Text("\(index + 1)")
                                    .foregroundStyle(isLast && typing ? Color(white: 0.85) : Color(white: 0.40))
                                    .frame(width: 44, alignment: .trailing)
                                    .padding(.trailing, 14)
                                    .frame(maxHeight: .infinity)
                                    .background(Self.gutterBackground)
                                HStack(spacing: 0) {
                                    Text(line).lineLimit(1)
                                    if isLast && typing { Cursor() }
                                }
                                .padding(.leading, 12)
                                Spacer(minLength: 0)
                            }
                            .frame(height: lineHeight)
                            .background(isLast && typing ? Color.white.opacity(0.05) : .clear)
                        }
                        Color.clear.frame(height: 1).id("end")
                    }
                    .font(.system(size: 14, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                }
                .defaultScrollAnchor(.top)
                .onChange(of: lines.count) { proxy.scrollTo("end", anchor: .bottom) }
            }
            .background(alignment: .leading) { Self.gutterBackground.frame(width: 58) }
        }
        .background(Self.background)
        .environment(\.colorScheme, .dark)
    }

    private var headerTitle: String? {
        if let custom = model.customTitle { return custom }
        return model.tasks.first { $0.id == model.current }?.prompt
    }
}

/// Editor tab strip: one open file, with the task it is answering on the right.
@available(macOS 15.0, *)
struct TabBar: View {
    let title: String?

    var body: some View {
        HStack(spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.left.forwardslash.chevron.right").font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.yellow)
                Text("solution.py").font(.system(size: 12))
                Image(systemName: "xmark").font(.system(size: 8, weight: .bold)).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(Editor.background)
            .overlay(alignment: .top) { Rectangle().fill(Color.accentColor).frame(height: 2) }
            Spacer()
            if let title {
                Text(title).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                    .padding(.horizontal, 12)
            }
        }
        .background(Color(red: 0.09, green: 0.09, blue: 0.10))
    }
}

/// Blinking block cursor at the end of the line being written.
struct Cursor: View {
    @State private var visible = true

    var body: some View {
        Rectangle().fill(Color(white: 0.9)).frame(width: 2, height: 17)
            .opacity(visible ? 1 : 0)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.5).repeatForever()) { visible = false }
            }
    }
}

@available(macOS 15.0, *)
struct Console: View {
    @EnvironmentObject private var model: WriterModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                Text("$ python3 solution.py").foregroundStyle(.secondary)
                ForEach(model.checks) { check in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: check.passed ? "checkmark" : "xmark")
                            .foregroundStyle(check.passed ? .green : .orange).frame(width: 14)
                        Text(check.test).foregroundStyle(.primary.opacity(0.85)).lineLimit(2)
                    }
                }
                if let error = model.runError {
                    Text(error).foregroundStyle(.red)
                } else if model.customTitle != nil, !model.isBusyCustom, !model.code.isEmpty {
                    Text("✓ valid Python (typed task: no tests)").foregroundStyle(.green)
                }
            }
            .font(.system(size: 12, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .leading).padding(12)
        }
        .background(Color(red: 0.08, green: 0.08, blue: 0.09))
        .environment(\.colorScheme, .dark)
    }
}

@available(macOS 15.0, *)
struct CustomTaskBar: View {
    @EnvironmentObject private var model: WriterModel

    var body: some View {
        HStack {
            Image(systemName: "text.cursor").foregroundStyle(.secondary)
            TextField("Or type your own: “Write a function that …”", text: $model.customTask)
                .textFieldStyle(.plain).font(.title3)
                .onSubmit { model.writeCustom() }
                .disabled(model.isRunning || model.isBusyCustom || model.writerMissing)
            if model.isBusyCustom { ProgressView().controlSize(.small) }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
    }
}
