import CodeSearch
import Foundation
import FluidUse

/// Indexes a repository's Swift declarations with EmbeddingGemma 2 and searches them in plain English.
///
///     swift run -c release CodeSearchCheck [--repo=path] [--tokens=64,128,256] [--eval] [--query="…"]…
///
/// `--eval` scores FluidAudio questions whose answer is known (top-1 / top-5 by file and name).
@main
struct CodeSearchCheck {
    struct Question {
        let text: String
        let file: String
        let name: String?
    }

    /// Questions about FluidAudio, worded without the function's own name.
    static let questions = [
        Question(
            text: "resample an audio file to 16 kHz mono", file: "AudioConverter.swift", name: "resampleAudioFile"),
        Question(
            text: "download every file of a Hugging Face model repository", file: "DownloadUtils.swift",
            name: "downloadRepo"),
        Question(text: "variational Bayes clustering of speaker embeddings", file: "VBxClustering.swift", name: nil),
        Question(text: "softmax over an array of floats", file: "VDSPOperations.swift", name: "softmax"),
        Question(
            text: "merge the transcripts of overlapping audio chunks", file: "ChunkProcessor.swift", name: "mergeChunks"
        ),
        Question(
            text: "cosine distance between two speaker embeddings", file: "SpeakerOperations.swift",
            name: "cosineDistance"),
        Question(
            text: "edit distance between two sequences", file: "LevenshteinDistance.swift", name: "levenshteinDistance"),
        Question(text: "write float samples to a WAV file", file: "AudioConverter.swift", name: "AudioWAV"),
        Question(text: "agglomerative hierarchical clustering", file: "AHCClustering.swift", name: nil),
        Question(text: "convert a word into phonemes", file: "G2PModel.swift", name: "phonemize"),
        Question(
            text: "text to speech with the Kokoro model on the Neural Engine", file: "KokoroAneManager.swift",
            name: "synthesize"),
        Question(
            text: "greedy transducer decoding that also returns token timestamps", file: "AsrManager.swift",
            name: "tdtDecodeWithTimings"),
        Question(text: "k-means clustering", file: "KMeansClustering.swift", name: nil),
        Question(
            text: "stretch a weight curve to a new length with interpolation", file: "WeightInterpolation.swift",
            name: "resample"),
        Question(
            text: "load the reference recording used to clone a voice", file: "PocketTtsVoiceCloner.swift",
            name: "loadAudio"),
        Question(
            text: "split a phoneme string into tokens for SSML", file: "SSMLProcessor.swift", name: "tokenizePhonemes"),
    ]

    static func main() async throws {
        let arguments = CommandLine.arguments.dropFirst()
        func value(_ name: String) -> String? {
            arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
        }
        let repo = ((value("repo") ?? "~/Documents/FluidAudio") as NSString).expandingTildeInPath
        let caps = (value("tokens") ?? "128").split(separator: ",").compactMap { Int($0) }
        let queries = arguments.filter { $0.hasPrefix("--query=") }.map { String($0.dropFirst(8)) }
        var start = DispatchTime.now().uptimeNanoseconds
        let chunks = CodeChunker.chunks(repository: URL(fileURLWithPath: repo))
        print(
            String(
                format: "%d chunks from %d files in %.2f s", chunks.count, Set(chunks.map(\.path)).count,
                seconds(since: start)))
        let manager = try await EmbeddingGemma2Manager.loadDefault()
        _ = try await manager.embed(Array(repeating: "warm up", count: 16), prompt: .none)
        for cap in caps {
            start = DispatchTime.now().uptimeNanoseconds
            var vectors: [[Float]] = []
            for batch in stride(from: 0, to: chunks.count, by: 256) {
                let slice = chunks[batch..<min(batch + 256, chunks.count)]
                vectors += try await manager.embed(slice.map(\.document), prompt: .none, maxTokens: cap)
            }
            let indexSeconds = seconds(since: start)
            print(
                String(
                    format: "cap %d tokens: indexed %d chunks in %.1f s (%.0f chunks/s)", cap, chunks.count,
                    indexSeconds, Double(chunks.count) / indexSeconds))
            func top(_ query: String, _ k: Int) async throws -> [CodeChunk] {
                let q = try await manager.embed(query, prompt: .codeRetrieval)
                let scored = vectors.indices.map { ($0, zip(q, vectors[$0]).reduce(Float(0)) { $0 + $1.0 * $1.1 }) }
                return scored.sorted { $0.1 > $1.1 }.prefix(k).map { chunks[$0.0] }
            }
            if arguments.contains("--eval") {
                var hit1 = 0
                var hit5 = 0
                for question in questions {
                    let found = try await top(question.text, 5)
                    let match = { (chunk: CodeChunk) in
                        chunk.path.hasSuffix(question.file)
                            && (question.name.map { chunk.name.hasSuffix($0) || chunk.name.contains($0) } ?? true)
                    }
                    if let first = found.first, match(first) { hit1 += 1 }
                    if found.contains(where: match) { hit5 += 1 }
                    if !arguments.contains("--quiet") {
                        print(
                            "  \(found.first.map(match) == true ? "✓" : found.contains(where: match) ? "~" : "✗") \(question.text) → \(found.first.map { "\($0.name) (\(($0.path as NSString).lastPathComponent):\($0.line))" } ?? "-")"
                        )
                    }
                }
                print("  top-1 \(hit1)/\(questions.count), top-5 \(hit5)/\(questions.count)")
            }
            for query in queries {
                print("\(query):")
                for chunk in try await top(query, 3) { print("   \(chunk.name)  \(chunk.path):\(chunk.line)") }
            }
        }
    }

    static func seconds(since start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }
}
