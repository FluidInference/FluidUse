import CoreGraphics
import Foundation
import ImageIO
import ImageSort
import SwiftUI

/// Plays a video in real time and keeps labeling the newest frame as fast as the model allows.
@MainActor
final class VideoSortModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case running
        case failed(String)
    }

    struct Label: Identifiable {
        let name: String
        let color: Color
        var id: String { name }
    }

    /// What to play and how to label it.
    struct Scene {
        let title: String
        let file: String
        let start: Double
        let labels: [Label]
        let template: String
        let columns: Int
        let rows: Int
        let credit: String
        /// JSON next to the video listing each single-label clip's `animal`, `clip_start` and `clip_end` seconds.
        let clipsFile: String?
        var isGrid: Bool { columns * rows > 1 }
    }

    /// An animal the stream settled on, with the frame it was first seen in.
    struct Spot: Identifiable {
        let id: Int
        let label: Int
        let share: Float
        let seconds: Double
        let thumbnail: CGImage?
        let correct: Bool?
    }

    static let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("FluidUse/video-sort")

    static let animals: Scene = {
        let names = [
            "elephant", "giraffe", "zebra", "tiger", "penguin", "flamingo", "brown bear", "wolf", "deer", "horse",
            "goat",
            "duck", "owl", "eagle", "monkey", "hippopotamus", "rhinoceros", "seal", "bison", "moose", "fox", "pelican",
            "swan", "camel", "cat", "dog",
        ]
        let labels = names.enumerated().map { index, name in
            Label(
                name: name, color: Color(hue: Double(index) / Double(names.count), saturation: 0.65, brightness: 0.95))
        }
        return Scene(
            title: "Name the animal", file: "animals.mp4", start: 0, labels: labels, template: "a photo of a {}.",
            columns: 1, rows: 1,
            credit:
                "Video: 26 clips from Wikimedia Commons (CC BY / CC BY-SA / CC0 / public domain; animals-credits.json)",
            clipsFile: "animals-credits.json")
    }()

    static let potatoes = Scene(
        title: "Label every frame", file: "potatoes.mp4", start: 9.5,
        labels: [
            Label(name: "potatoes", color: Color(red: 0.95, green: 0.78, blue: 0.35)),
            Label(name: "a gloved hand", color: .red), Label(name: "a person", color: .purple),
            Label(name: "a conveyor belt", color: .blue), Label(name: "a truck", color: .orange),
            Label(name: "the sky", color: .cyan), Label(name: "a metal machine", color: .gray),
            Label(name: "gravel ground", color: .brown), Label(name: "a wire basket", color: .green),
            Label(name: "a car", color: .pink),
        ],
        template: "a photo of {}.", columns: 6, rows: 4,
        credit: "Video: \"Potatoes Sorting Montana2026\", USDA (Brien Aho), public domain, via Wikimedia Commons",
        clipsFile: nil)

    let scene: Scene
    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published private(set) var frame: CGImage?
    @Published private(set) var grid: GridClassifier.Grid?
    @Published private(set) var framesPerSecond: Double = 0
    @Published private(set) var framesLabeled = 0
    @Published private(set) var framesCorrect = 0
    @Published private(set) var spots: [Spot] = []
    /// Label the current clip really shows, when the scene knows it.
    @Published private(set) var truth: Int?

    private let environment = ProcessInfo.processInfo.environment
    private let logFrames = ProcessInfo.processInfo.environment["VIDEO_SORT_LOG"] == "1"
    private var classifier: GridClassifier?
    private var player: Task<Void, Never>?
    private var labeler: Task<Void, Never>?
    private var latest: VideoFrames.Frame?
    private var labelTimes: [Date] = []
    private var streak: (label: Int, count: Int) = (-1, 0)
    /// (start, end, label) per clip, when the scene has single-label clips.
    private var clips: [(start: Double, end: Double, label: Int)] = []
    private var preparing = false

    init() {
        scene = ProcessInfo.processInfo.environment["VIDEO_SORT_SCENE"] == "potatoes" ? Self.potatoes : Self.animals
    }

    var counts: [Int] {
        var counts = [Int](repeating: 0, count: scene.labels.count)
        for cell in grid?.cells ?? [] { counts[cell.label] += 1 }
        return counts
    }

    var accuracy: Double? {
        framesLabeled > 0 && !clips.isEmpty ? Double(framesCorrect) / Double(framesLabeled) : nil
    }

    private func expectedLabel(at seconds: Double) -> Int? {
        clips.first { seconds >= $0.start && seconds < $0.end }?.label
    }

    private func loadClips() {
        struct Clip: Decodable {
            let animal: String
            let clipStart: Double
            let clipEnd: Double
            enum CodingKeys: String, CodingKey {
                case animal
                case clipStart = "clip_start"
                case clipEnd = "clip_end"
            }
        }
        guard let file = scene.clipsFile,
            let data = try? Data(contentsOf: Self.directory.appendingPathComponent(file)),
            let decoded = try? JSONDecoder().decode([Clip].self, from: data)
        else { return }
        let names = scene.labels.map(\.name)
        clips = decoded.compactMap { clip in
            names.firstIndex(of: clip.animal).map { (clip.clipStart, clip.clipEnd, $0) }
        }
    }

    var speciesSpotted: Int { Set(spots.filter { $0.correct != false }.map(\.label)).count }

    func prepare() async {
        guard !preparing else { return }
        preparing = true
        do {
            loadClips()
            phase = .loading("Loading SigLIP 2 and embedding \(scene.labels.count) labels…")
            classifier = try await GridClassifier.load(
                labels: scene.labels.map(\.name), template: scene.template, columns: scene.columns, rows: scene.rows)
            phase = .ready
            startPlayback()
            print("ready at \(Date().timeIntervalSince1970)")
            // VIDEO_SORT_WAIT=1 shows the first frame and waits for Start (Space).
            if environment["VIDEO_SORT_WAIT"] != "1" { toggle() }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func startPlayback() {
        let path = environment["VIDEO_SORT_FILE"] ?? Self.directory.appendingPathComponent(scene.file).path
        let start = environment["VIDEO_SORT_START"].flatMap(Double.init) ?? scene.start
        player?.cancel()
        latest = nil
        player = Task { [weak self] in
            do {
                for try await frame in VideoFrames.stream(
                    url: URL(fileURLWithPath: path), start: start, width: 1280, height: 720)
                {
                    guard let self else { return }
                    self.frame = frame.image
                    self.latest = frame
                    truth = expectedLabel(at: frame.seconds)
                    // Hold the first frame until Start.
                    if phase == .ready, environment["VIDEO_SORT_WAIT"] == "1" { break }
                }
            } catch {
                self?.phase = .failed("Video: \(error.localizedDescription)")
            }
        }
    }

    func toggle() {
        if phase == .running {
            labeler?.cancel()
            player?.cancel()
            phase = .ready
            return
        }
        guard phase == .ready, let classifier else { return }
        phase = .running
        framesLabeled = 0
        framesCorrect = 0
        spots = []
        streak = (-1, 0)
        labelTimes = []
        startPlayback()
        let inFlight = scene.isGrid ? 6 : 1
        labeler = Task { [weak self] in
            var lastNumber = -1
            while !Task.isCancelled {
                guard let frame = self?.latest, frame.number != lastNumber else {
                    try? await Task.sleep(for: .milliseconds(3))
                    continue
                }
                lastNumber = frame.number
                guard let grid = try? await Self.label(frame.image, with: classifier, inFlight: inFlight) else {
                    continue
                }
                self?.apply(grid, frame: frame)
            }
        }
    }

    private nonisolated static func label(
        _ image: CGImage, with classifier: GridClassifier, inFlight: Int
    )
        async throws -> GridClassifier.Grid
    {
        try await Task.detached { try await classifier.classify(image, inFlight: inFlight) }.value
    }

    private func apply(_ grid: GridClassifier.Grid, frame: VideoFrames.Frame) {
        self.grid = grid
        framesLabeled += 1
        let clipTruth = expectedLabel(at: frame.seconds)
        if let clipTruth, grid.cells.first?.label == clipTruth { framesCorrect += 1 }
        let now = Date()
        labelTimes.append(now)
        labelTimes.removeAll { now.timeIntervalSince($0) > 1 }
        framesPerSecond = Double(labelTimes.count)
        if !scene.isGrid, let cell = grid.cells.first { track(cell, frame: frame, truth: clipTruth) }
        if logFrames { log(grid, frame: frame, truth: clipTruth) }
    }

    /// Adds a spot once the same confident label holds for several labeled frames in a row.
    private func track(_ cell: GridClassifier.Cell, frame: VideoFrames.Frame, truth: Int?) {
        streak = cell.label == streak.label ? (cell.label, streak.count + 1) : (cell.label, 1)
        guard streak.count == 6, cell.share >= 0.5, spots.last?.label != cell.label else { return }
        let thumbnail = frame.image.cropping(
            to: CGRect(
                x: (frame.image.width - frame.image.height) / 2, y: 0, width: frame.image.height,
                height: frame.image.height))
        spots.append(
            Spot(
                id: spots.count, label: cell.label, share: cell.share, seconds: frame.seconds, thumbnail: thumbnail,
                correct: truth.map { $0 == cell.label }))
    }

    private func log(_ grid: GridClassifier.Grid, frame: VideoFrames.Frame, truth: Int?) {
        let (cyan, yellow, red, green, dim, reset) =
            ("\u{1B}[1;36m", "\u{1B}[33m", "\u{1B}[31m", "\u{1B}[32m", "\u{1B}[2m", "\u{1B}[0m")
        let time = String(format: "%.2f", frame.seconds)
        if scene.isGrid {
            let summary = counts.enumerated().filter { $0.element > 0 }.sorted { $0.element > $1.element }
                .map { "\(scene.labels[$0.offset].name) \($0.element)" }.joined(separator: " · ")
            print(
                "\(cyan)▶ frame \(frame.number) @ \(time) s\(reset)  \(yellow)\(summary)\(reset)  "
                    + "\(red)\(grid.cells.count) cells in \(String(format: "%.0f", grid.milliseconds)) ms\(reset)")
            return
        }
        guard let cell = grid.cells.first else { return }
        let mark: String =
            truth.map { $0 == cell.label ? "\(green)✓\(reset)" : "\(red)✗ clip: \(scene.labels[$0].name)\(reset)" }
            ?? ""
        print(
            "\(cyan)▶ frame \(frame.number) @ \(time) s\(reset)  \(yellow)→ \(scene.labels[cell.label].name)\(reset) · "
                + "\(Int(cell.share * 100))% · \(red)model call \(String(format: "%.1f", grid.milliseconds)) ms\(reset)  "
                + "\(mark)\(dim)\(reset)")
    }
}
