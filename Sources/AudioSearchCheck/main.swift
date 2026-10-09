import Foundation
import FluidUse

/// Indexes audio files with EmbeddingGemma 2 (10 s windows) and searches them with text.
///
///     swift run -c release AudioSearchCheck file.wav [more files…] [--query="…"]… [--dump=vectors.jsonl]
@main
struct AudioSearchCheck {
    static func main() async throws {
        let arguments = CommandLine.arguments.dropFirst()
        // Folders expand to the audio files inside them, sorted.
        let audioExtensions: Set<String> = ["wav", "flac", "mp3", "m4a", "aiff", "caf"]
        let files = arguments.filter { !$0.hasPrefix("--") }.flatMap { path -> [URL] in
            let url = URL(fileURLWithPath: path)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
                return [url]
            }
            let names = (try? FileManager.default.contentsOfDirectory(atPath: path)) ?? []
            return names.filter { audioExtensions.contains(($0 as NSString).pathExtension.lowercased()) }.sorted()
                .map { url.appendingPathComponent($0) }
        }
        let queries = arguments.filter { $0.hasPrefix("--query=") }.map { String($0.dropFirst(8)) }
        let dump = arguments.first { $0.hasPrefix("--dump=") }.map { String($0.dropFirst(7)) }
        guard !files.isEmpty else {
            print("usage: AudioSearchCheck file.wav [more files…] [--query=\"…\"]… [--dump=vectors.jsonl]")
            exit(2)
        }
        var start = DispatchTime.now().uptimeNanoseconds
        let text = try await EmbeddingGemma2Manager.loadDefault()
        let audio = try await EmbeddingGemma2Audio.load(text: text)
        _ = try await audio.embed(window: [Float](repeating: 0, count: 16_000))
        print(String(format: "models loaded in %.1f s", seconds(since: start)))
        start = DispatchTime.now().uptimeNanoseconds
        let recordings = try files.map { try EmbeddingGemma2Audio.samples(contentsOf: $0) }
        let decodeSeconds = seconds(since: start)
        start = DispatchTime.now().uptimeNanoseconds
        let embedded = try await audio.embed(recordings: recordings)
        let embedSeconds = seconds(since: start)
        let audioSeconds = recordings.reduce(0.0) { $0 + Double($1.count) } / Double(EmbeddingGemma2Audio.sampleRate)
        let index: [(file: String, window: EmbeddingGemma2Audio.Window)] = zip(files, embedded).flatMap {
            file, windows in
            windows.map { (file: file.lastPathComponent, window: $0) }
        }
        print(
            String(
                format: "%.1f min of audio, %d windows: decode %.1f s, embed %.1f s = %.0fx real time (embed only)",
                audioSeconds / 60, index.count, decodeSeconds, embedSeconds, audioSeconds / max(embedSeconds, 1e-6)))
        if let dump {
            let lines = index.map { item -> String in
                let object: [String: Any] = ["file": item.file, "start": item.window.start, "v": item.window.embedding]
                return String(decoding: try! JSONSerialization.data(withJSONObject: object), as: UTF8.self)
            }
            try lines.joined(separator: "\n").write(toFile: dump, atomically: true, encoding: .utf8)
        }
        for query in queries {
            let vector = try await text.embed(query, prompt: .searchQuery)
            let ranked = index.map { item in
                (item, zip(vector, item.window.embedding).reduce(Float(0)) { $0 + $1.0 * $1.1 })
            }
            .sorted { $0.1 > $1.1 }.prefix(3)
            print("\(query):")
            for (item, score) in ranked {
                print(String(format: "   %.3f  %@ @ %@", score, item.file, clock(item.window.start)))
            }
        }
    }

    static func seconds(since start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }

    static func clock(_ seconds: TimeInterval) -> String {
        String(format: "%d:%02d", Int(seconds) / 60, Int(seconds) % 60)
    }
}
