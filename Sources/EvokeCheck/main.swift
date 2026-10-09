import CoreML
import FluidUse
import Foundation

/// Granite-Embedding-30M-Sparse (Evoke) parity against Evoke's shipped ONNX compilers.
///
///     swift run -c release EvokeCheck parity Tests/FluidUseTests/Fixtures/evoke-reference.json [gpu]
///
/// Fixtures (model-lab evoke-coreml `swift_fixtures.py`): Hugging Face RoBERTa token ids, and ONNX query/document
/// terms for NFCorpus queries and titles. Models come from `EVOKE_MODEL_DIR` or the pinned Hub revision (L64).
@main
struct EvokeCheck {
    struct Reference: Decodable {
        struct TokenCase: Decodable {
            let text: String
            let ids: [Int]
        }
        struct TermCase: Decodable {
            let kind: String
            let ids: [Int]
            let terms: [String: Float]
        }
        let tokens: [TokenCase]
        let atoms: [TermCase]
    }

    static func main() async throws {
        let a = Array(CommandLine.arguments.dropFirst())
        guard a.count >= 2, a[0] == "parity" else {
            fputs("usage: EvokeCheck parity <evoke-reference.json> [gpu]\n", stderr)
            exit(2)
        }
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: a[1])))
        let units: MLComputeUnits = a.count > 2 && a[2] == "gpu" ? .cpuAndGPU : .cpuAndNeuralEngine
        let manager = try await EvokeManager.loadDefault(lengths: [64], computeUnits: units)

        var tokenBad = 0
        for row in reference.tokens where try manager.tokenize(row.text) != row.ids {
            tokenBad += 1
            print(
                "token mismatch: \(row.text.debugDescription)\n  want \(row.ids)\n  got  \(try manager.tokenize(row.text))"
            )
        }

        _ = try await manager.terms(for: "warm up", kind: .query)
        var exact = 0
        var overlaps: [Double] = []
        var dots: [Double] = []
        var latencies: [Double] = []
        for row in reference.atoms {
            let kind = EvokeTextKind(rawValue: row.kind)!
            let want = Dictionary(uniqueKeysWithValues: row.terms.map { (Int($0.key)!, $0.value) })
            let (got, ms) = try await manager.terms(tokenIds: row.ids, kind: kind)
            latencies.append(ms)
            if Set(got.keys) == Set(want.keys) { exact += 1 }
            overlaps.append(Double(Set(got.keys).intersection(want.keys).count) / Double(max(want.count, 1)))
            dots.append(Double(EvokeTerms.score(want, got) / max(EvokeTerms.score(want, want), 1e-9)))
        }
        latencies.sort()
        print("tokenizer: \(reference.tokens.count - tokenBad)/\(reference.tokens.count) cases identical")
        print(
            String(
                format: "terms (%@): exact id sets %d/%d, id overlap min %.3f mean %.4f, self-dot ratio min %.4f",
                units == .cpuAndGPU ? "GPU" : "Neural Engine", exact, reference.atoms.count, overlaps.min()!,
                overlaps.reduce(0, +) / Double(overlaps.count), dots.min()!))
        print(String(format: "prediction p50 %.2f ms", latencies[latencies.count / 2]))
        let passed = tokenBad == 0 && overlaps.min()! >= 0.9 && dots.min()! >= 0.98
        print(passed ? "PASS" : "FAIL")
        exit(passed ? 0 : 1)
    }
}
