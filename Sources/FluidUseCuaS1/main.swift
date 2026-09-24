import CoreML
import Foundation
import FluidUse
import ImageIO

/// `fluiduse-cua-s1` -- Cua-S1-4B-0.2 Core ML runtime checks.
///
///   parity --models <dir> --fixtures <swift-text.json|swift-multimodal.json> [--screens <dir>] [--variant w8]
///
/// Rebuilds each fixture prompt with `CuaS1FourBPrompt`, checks the chat string and token ids against the
/// Python reference, runs the Core ML model and compares the letter softmax with the fp32 reference.

struct FixtureOption: Decodable {
    let elementId: String
    let role: String
    let label: String
    let action: String
    let entityId: String?
}

struct FixtureTask: Decodable {
    let id: String
    let app: String
    let taskFamily: String
    let goal: String?
    let axTree: String?
    let screenshot: String?
    let options: [FixtureOption]
    let expected: [String: String]
    let chat: String
    let inputIds: [Int]
    let letterLogits: [Float]
}

struct FixtureFile: Decodable {
    let modality: String
    let tasks: [FixtureTask]
}

func snakeCaseDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}

func value(_ flag: String, in args: [String]) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    return args[i + 1]
}

func softmax(_ x: [Float]) -> [Float] {
    let m = x.max() ?? 0
    let e = x.map { expf($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}

func loadImage(_ url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { throw CuaS1FourBError.invalidInput("cannot read \(url.path)") }
    return image
}

func parity(_ args: [String]) async throws {
    guard let models = value("--models", in: args), let fixturesPath = value("--fixtures", in: args) else {
        print("usage: parity --models <dir> --fixtures <swift-*.json> [--screens <dir>] [--variant w8]")
        exit(2)
    }
    let fixtures = try snakeCaseDecoder().decode(
        FixtureFile.self, from: Data(contentsOf: URL(fileURLWithPath: fixturesPath)))
    guard let modality = CuaS1FourBModality(rawValue: fixtures.modality) else { exit(2) }
    let screens = URL(fileURLWithPath: value("--screens", in: args) ?? "fixtures/screens")
    var configuration = CuaS1FourBManager.Configuration(modality: modality)
    configuration.variant = value("--variant", in: args) ?? ""
    if let lengths = value("--lengths", in: args) {
        configuration.lengths = lengths.split(separator: ",").compactMap { Int($0) }
    }
    let loadStart = Date()
    let manager = try await CuaS1FourBManager.load(from: URL(fileURLWithPath: models), configuration: configuration)
    print(String(format: "loaded %@ in %.1f s", modality.rawValue, Date().timeIntervalSince(loadStart)))
    let warmStart = Date()
    try await manager.prewarm()
    print(String(format: "prewarmed in %.1f s", Date().timeIntervalSince(warmStart)))

    var chatOK = 0
    var idsOK = 0
    var argmaxOK = 0
    var maxDp: Float = 0
    var times: [Double] = []
    for task in fixtures.tasks {
        let state = CuaS1FourBState(
            app: task.app, taskFamily: task.taskFamily, goal: task.goal, accessibilityTree: task.axTree,
            screenshot: try task.screenshot.map { try loadImage(screens.appendingPathComponent($0)) },
            options: task.options.map {
                CuaS1FourBOption(
                    elementId: $0.elementId, role: $0.role, label: $0.label, action: $0.action, entityId: $0.entityId)
            })
        let chat = try CuaS1FourBPrompt.chat(state: state, modality: modality)
        if chat == task.chat { chatOK += 1 } else { print("  chat mismatch: \(task.id)") }
        if modality == .text {
            let ids = manager.tokenizer.encode(task.chat)
            if ids == task.inputIds {
                idsOK += 1
            } else {
                print("  token mismatch: \(task.id) \(ids.count) vs \(task.inputIds.count)")
            }
        }
        let start = Date()
        let decision = try await manager.decide(state)
        times.append(Date().timeIntervalSince(start) * 1000)
        if modality == .multimodal {
            if decision.tokens == task.inputIds.count {
                idsOK += 1
            } else {
                print("  length mismatch: \(task.id) \(decision.tokens) vs \(task.inputIds.count)")
            }
        }
        let got = softmax(decision.options.map(\.logit))
        let want = softmax(task.letterLogits)
        let dp = zip(got, want).map { abs($0 - $1) }.max() ?? 0
        maxDp = max(maxDp, dp)
        let gotArg = got.indices.max { got[$0] < got[$1] }!
        let wantArg = want.indices.max { want[$0] < want[$1] }!
        if gotArg == wantArg { argmaxOK += 1 } else { print("  argmax mismatch: \(task.id) dp=\(dp)") }
    }
    let sorted = times.dropFirst().sorted()
    let median = sorted.isEmpty ? times[0] : sorted[sorted.count / 2]
    let n = fixtures.tasks.count
    print("chat \(chatOK)/\(n)  tokens \(idsOK)/\(n)  argmax \(argmaxOK)/\(n)  max|dp| \(maxDp)")
    print(String(format: "median decision %.0f ms (first after prewarm %.0f ms)", median, times[0]))
}

let args = Array(CommandLine.arguments.dropFirst())
switch args.first {
case "parity":
    try await parity(Array(args.dropFirst()))
default:
    print("usage: fluiduse-cua-s1 parity ...")
    exit(2)
}
