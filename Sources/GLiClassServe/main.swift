import FluidUse
import Foundation

/// GLiClass decisions over stdin/stdout, one JSON object per line, for harnesses outside Swift
/// (Tools/doom). Prints `{"ready": true, ...}` once the model is loaded.
///
///     in:  {"text": "...", "labels": ["a", "b"], "prompt": "..."}
///     out: {"index": 0, "probabilities": [0.9, 0.1], "ms": 1.8}
///
///     swift run -c release GLiClassServe [--precision lut8] [--model-dir <dir>]
@main
struct GLiClassServe {
    struct Request: Decodable {
        let text: String
        let labels: [String]
        let prompt: String?
    }

    struct Response: Encodable {
        var index: Int?
        var probabilities: [Float]?
        var ms: Double?
        var error: String?
    }

    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        func option(_ name: String) -> String? {
            arguments.firstIndex(of: name).flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil }
        }
        let precision = option("--precision") ?? "lut8"
        let configuration = GLiClassManager.Configuration(lengths: [128], precision: precision)
        let started = Date()
        let manager: GLiClassManager
        if let directory = option("--model-dir") {
            manager = try await GLiClassManager.load(
                from: URL(fileURLWithPath: directory), configuration: configuration)
        } else {
            manager = try await GLiClassManager.load(configuration: configuration)
        }
        _ = try await manager.classify(text: "warm up", labels: ["yes", "no"])
        let loadSeconds = Date().timeIntervalSince(started)
        emit("{\"ready\": true, \"precision\": \"\(precision)\", \"load_s\": \(loadSeconds)}")

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        while let line = readLine() {
            guard !line.isEmpty else { continue }
            var response = Response()
            do {
                let request = try decoder.decode(Request.self, from: Data(line.utf8))
                let begin = DispatchTime.now().uptimeNanoseconds
                let answer = try await manager.classify(
                    text: request.text, labels: request.labels, prompt: request.prompt)
                response.ms = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e6
                response.index = answer.selectedIndex
                response.probabilities = answer.probabilities
            } catch {
                response.error = error.localizedDescription
            }
            emit(String(decoding: try encoder.encode(response), as: UTF8.self))
        }
    }

    private static func emit(_ line: String) {
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }
}
