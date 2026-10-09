@preconcurrency import AVFoundation
@preconcurrency import CoreML
import Foundation

/// EmbeddingGemma 2 audio embeddings: 10 s windows of 16 kHz mono audio go through `EmbeddingGemma2Audio` (log-mel +
/// USM conformer, on the GPU, where it runs 3x faster than on the Neural Engine) and then the text model (Neural
/// Engine), so audio lands in the same space as text and can be searched with a text query.
public final class EmbeddingGemma2Audio: Sendable {
    public static let sampleRate = 16_000
    public static let windowSamples = 160_000
    static let leftPad = 160
    static let frames = 1_000
    public static let maxInFlight = 4

    /// One embedded window of a recording.
    public struct Window: Sendable {
        public let start: TimeInterval
        public let duration: TimeInterval
        public let embedding: [Float]
    }

    public let text: EmbeddingGemma2Manager
    private let model: MLModel

    init(text: EmbeddingGemma2Manager, model: MLModel) {
        self.text = text
        self.model = model
    }

    /// Downloads (once, checksum-verified) and loads the audio model next to an already loaded text model.
    public static func load(
        text: EmbeddingGemma2Manager, computeUnits: MLComputeUnits = .cpuAndGPU,
        progress: EmbeddingGemma2ModelStore.Progress? = nil
    ) async throws -> EmbeddingGemma2Audio {
        let directory: URL
        if let path = ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_MODEL_DIR"], !path.isEmpty {
            directory = URL(fileURLWithPath: path)
        } else {
            directory = try await EmbeddingGemma2ModelStore.ensureAudio(progress: progress)
        }
        let compiled = directory.appendingPathComponent("EmbeddingGemma2Audio.mlmodelc")
        let package = directory.appendingPathComponent("EmbeddingGemma2Audio.mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            url = try await MLModel.compileModel(at: package)
        } else {
            throw EmbeddingGemma2Error.invalidAsset("Missing EmbeddingGemma2Audio in \(directory.path)")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        return EmbeddingGemma2Audio(
            text: text, model: try await MLModel.load(contentsOf: url, configuration: configuration))
    }

    /// Decodes any file AVFoundation reads to 16 kHz mono Float samples.
    public static func samples(contentsOf url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard
            let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: file.processingFormat, to: target)
        else { throw EmbeddingGemma2Error.invalidAsset("Cannot convert \(url.lastPathComponent) to 16 kHz mono") }
        let chunk: AVAudioFrameCount = 65_536
        var result: [Float] = []
        result.reserveCapacity(Int(Double(file.length) * Double(sampleRate) / file.processingFormat.sampleRate) + 1)
        while true {
            let capacity =
                AVAudioFrameCount(Double(chunk) * Double(sampleRate) / file.processingFormat.sampleRate) + 1024
            guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { break }
            var error: NSError?
            // Once the file is exhausted every read returns no frames, which ends the stream; no state to share.
            let status = converter.convert(to: output, error: &error) { _, outStatus in
                guard let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: chunk),
                    (try? file.read(into: input, frameCount: chunk)) != nil, input.frameLength > 0
                else {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return input
            }
            if let error { throw error }
            if let channel = output.floatChannelData?[0], output.frameLength > 0 {
                result.append(contentsOf: UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
            }
            if status == .endOfStream || status == .error { break }
        }
        return result
    }

    /// Embeddings of consecutive 10 s windows of `samples` (16 kHz mono), in order. `progress` reports windows done.
    public func embed(
        samples: [Float], progress: (@Sendable (_ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> [Window] {
        try await embed(recordings: [samples], progress: progress).first ?? []
    }

    /// Windows of several recordings, embedded together so short files still keep the GPU and the Neural Engine
    /// busy (one file at a time leaves a single window in flight). Result: one window list per recording, in order.
    /// `onWindow` receives each window as soon as it is embedded (completion order), for indexes that grow live;
    /// `beforeWindow` is awaited before each window starts (a pause gate, for example).
    public func embed(
        recordings: [[Float]], progress: (@Sendable (_ done: Int, _ total: Int) -> Void)? = nil,
        onWindow: (@Sendable (_ recording: Int, _ window: Window) -> Void)? = nil,
        beforeWindow: (@Sendable () async -> Void)? = nil
    ) async throws -> [[Window]] {
        var jobs: [(recording: Int, start: Int)] = []
        for (recording, samples) in recordings.enumerated() {
            var starts = Array(stride(from: 0, to: samples.count, by: Self.windowSamples))
            // A tail under 0.5 s carries almost nothing and may be too short to produce a token.
            if starts.count > 1, let last = starts.last, samples.count - last < Self.sampleRate / 2 {
                starts.removeLast()
            }
            jobs += starts.map { (recording, $0) }
        }
        return try await withThrowingTaskGroup(of: (Int, Window).self) { group in
            var windows = [Window?](repeating: nil, count: jobs.count)
            var next = 0
            var done = 0
            func addJob() {
                guard next < jobs.count else { return }
                let index = next
                next += 1
                let job = jobs[index]
                let samples = recordings[job.recording]
                let slice = Array(samples[job.start..<min(job.start + Self.windowSamples, samples.count)])
                group.addTask {
                    await beforeWindow?()
                    let embedding = try await self.embed(window: slice)
                    return (
                        index,
                        Window(
                            start: Double(job.start) / Double(Self.sampleRate),
                            duration: Double(slice.count) / Double(Self.sampleRate), embedding: embedding)
                    )
                }
            }
            for _ in 0..<min(Self.maxInFlight, jobs.count) { addJob() }
            for try await (index, window) in group {
                windows[index] = window
                onWindow?(jobs[index].recording, window)
                try Task.checkCancellation()
                done += 1
                progress?(done, jobs.count)
                addJob()
            }
            var result = [[Window]](repeating: [], count: recordings.count)
            for (index, job) in jobs.enumerated() {
                if let window = windows[index] { result[job.recording].append(window) }
            }
            return result
        }
    }

    /// Mel frames with real audio in a window of `samples` samples: frame i covers padded samples [160 i, 160 i + 321)
    /// (the feature extractor's 160-sample left pad included) and counts only when all of them are real.
    static func validFrames(samples: Int) -> Int {
        let real = leftPad + min(samples, windowSamples)
        return real < 321 ? 0 : min(frames, (real - 321) / 160 + 1)
    }

    /// Audio tokens for `validFrames` frames: two stride-2 convolutions, so token k is valid when frame 4k is.
    static func tokenCount(validFrames: Int) -> Int { (validFrames + 3) / 4 }

    /// Embedding of up to 10 s of 16 kHz mono audio.
    public func embed(window: [Float]) async throws -> [Float] {
        let count = min(window.count, Self.windowSamples)
        let waveform = try MLMultiArray(
            shape: [1, NSNumber(value: Self.windowSamples + Self.leftPad)], dataType: .float32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: Self.frames)], dataType: .float16)
        let wave = waveform.dataPointer.assumingMemoryBound(to: Float.self)
        wave.update(repeating: 0, count: Self.windowSamples + Self.leftPad)
        window.withUnsafeBufferPointer { (wave + Self.leftPad).update(from: $0.baseAddress!, count: count) }
        let maskPointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        let validFrames = Self.validFrames(samples: count)
        for frame in 0..<Self.frames { maskPointer[frame] = frame < validFrames ? 1 : 0 }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "waveform": MLFeatureValue(multiArray: waveform), "frame_mask": MLFeatureValue(multiArray: mask),
        ])
        let output = try await model.prediction(from: input)
        guard let tokens = output.featureValue(for: "audio_tokens")?.multiArrayValue else {
            throw EmbeddingGemma2Error.predictionFailed("missing audio_tokens output")
        }
        let tokenCount = Self.tokenCount(validFrames: validFrames)
        guard tokenCount > 0 else { throw EmbeddingGemma2Error.predictionFailed("window too short to embed") }
        return try await text.embed(audioTokens: tokens, count: tokenCount)
    }
}
