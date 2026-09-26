import FluidUse
import Foundation

/// Kev-0.8B checks.
///
///     swift run -c release KevCheck tokenizer <tokenizer.json> <fixtures.json>
///     swift run -c release KevCheck parity <model directory> <records.json>
///     swift run -c release KevCheck serving <model directory>   (Kev's scripts/serving_bench.py cases)
///     swift run -c release KevCheck fast-parity <model directory> <records.json>   (state-cache runtime)
///     swift run -c release KevCheck fast-serving <model directory>
@main
struct KevCheck {
    struct Fixture: Decodable {
        let text: String
        let ids: [Int]
    }

    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if #available(macOS 15.0, *), arguments.first == "fast-serving", arguments.count == 2 {
            let manager = try await KevFastManager.load(from: URL(fileURLWithPath: arguments[1]))
            try await manager.warm()
            try await serving { try await manager.answer(state: $0, questions: $1, maxStateTokens: 8192) }
            return
        }
        if #available(macOS 15.0, *), arguments.first == "fast-parity", arguments.count == 3 {
            let manager = try await KevFastManager.load(from: URL(fileURLWithPath: arguments[1]))
            try await parity(records: URL(fileURLWithPath: arguments[2])) {
                try await manager.answer(state: $0, questions: $1)
            }
            return
        }
        if arguments.first == "serving", arguments.count == 2 {
            try await serving(directory: URL(fileURLWithPath: arguments[1]))
            return
        }
        if arguments.first == "parity", arguments.count == 3 {
            try await parity(directory: URL(fileURLWithPath: arguments[1]), records: URL(fileURLWithPath: arguments[2]))
            return
        }
        guard arguments.first == "tokenizer", arguments.count == 3 else {
            fputs("usage: KevCheck tokenizer <tokenizer.json> <fixtures.json> | parity <dir> <records.json>\n", stderr)
            exit(2)
        }
        let start = Date()
        let tokenizer = try QwenBPETokenizer(tokenizerJsonURL: URL(fileURLWithPath: arguments[1]))
        print(String(format: "loaded tokenizer in %.2f s", Date().timeIntervalSince(start)))
        let fixtures = try JSONDecoder().decode(
            [Fixture].self, from: Data(contentsOf: URL(fileURLWithPath: arguments[2])))
        var mismatches = 0
        var tokens = 0
        let encodeStart = Date()
        for fixture in fixtures {
            let ids = try tokenizer.encode(fixture.text)
            tokens += ids.count
            if ids != fixture.ids {
                mismatches += 1
                if mismatches <= 5 {
                    let prefix = zip(ids, fixture.ids).prefix { $0 == $1 }.count
                    print("MISMATCH at token \(prefix): \(fixture.text.prefix(80).debugDescription)")
                    print(
                        "  swift \(Array(ids.dropFirst(prefix).prefix(8)))  hf \(Array(fixture.ids.dropFirst(prefix).prefix(8)))"
                    )
                }
            }
        }
        print(
            String(
                format: "fixtures %d, tokens %d, mismatches %d, encode %.2f s", fixtures.count, tokens, mismatches,
                Date().timeIntervalSince(encodeStart)))
        exit(mismatches == 0 ? 0 : 1)
    }

    /// Kev's System One question JSON -> KevQuestion, as `kev.api.to_record` reads it.
    static func question(_ json: [String: Any]) throws -> KevQuestion {
        let instructions = json["instructions"] as? String ?? ""
        func text(_ value: Any?) throws -> String? {
            switch value {
            case nil, is NSNull: return nil
            case let string as String: return string
            case let number as NSNumber: return number.stringValue
            default: throw KevError.invalidAsset("non-text criteria are not supported by this check")
            }
        }
        switch json["type"] as? String {
        case "choice":
            guard let criteria = json["criteria"] as? [[Any]] else { throw KevError.invalidAsset("choice criteria") }
            return .choice(
                instructions, options: try criteria.map { (key: $0[0] as? String ?? "", description: try text($0[1])) })
        case "noul":
            let criteria = json["criteria"] as? [String: Any] ?? [:]
            return .noul(instructions, no: try text(criteria["false"]), yes: try text(criteria["true"]))
        case "score":
            guard let levels = json["criteria"] as? [Any] else { throw KevError.invalidAsset("score levels") }
            return .score(instructions, levels: try levels.map { try text($0) ?? "" })
        default:
            throw KevError.invalidAsset("unknown question type")
        }
    }

    static func parity(directory: URL, records: URL) async throws {
        let start = Date()
        let manager = try await KevManager.load(from: directory)
        print(String(format: "loaded in %.1f s", Date().timeIntervalSince(start)))
        try await parity(records: records) { try await manager.answer(state: $0, questions: $1) }
    }

    static func parity(
        records: URL, answer: @Sendable (String, [KevQuestion]) async throws -> [KevAnswer]
    ) async throws {
        guard let rows = try JSONSerialization.jsonObject(with: Data(contentsOf: records)) as? [[String: Any]] else {
            throw KevError.invalidAsset("records.json")
        }
        var questions = 0
        var flips = 0
        var worst: Float = 0
        var skipped = 0
        let runStart = Date()
        for row in rows {
            guard let state = row["state"] as? String, let specs = row["questions"] as? [[String: Any]],
                let expected = row["probabilities"] as? [[Double]]
            else { continue }
            let typed: [KevQuestion]
            do {
                typed = try specs.map(question)
            } catch {
                skipped += 1
                continue
            }
            let answers = try await answer(state, typed)
            for (answer, reference) in zip(answers, expected) {
                questions += 1
                let difference = zip(answer.probabilities, reference).map { abs($0 - Float($1)) }.max() ?? 0
                worst = max(worst, difference)
                if difference > 0.01 {
                    print(
                        "  |dp| \(difference) keys \(answer.keys) swift \(answer.probabilities.map { ($0 * 1000).rounded() / 1000 }) "
                            + "python \(reference.map { ($0 * 1000).rounded() / 1000 }) tokens \(answer.tokens)")
                }
                let referenceBest = reference.indices.max { reference[$0] < reference[$1] } ?? 0
                if answer.bestIndex != referenceBest { flips += 1 }
            }
        }
        print(
            String(
                format: "questions %d, top-answer flips %d, max |dp| %.5f, skipped records %d, %.1f ms/question",
                questions,
                flips, worst, skipped, 1000 * Date().timeIntervalSince(runStart) / Double(max(questions, 1))))
    }

    /// The request shapes of Kev's `scripts/serving_bench.py`, with the same texts and questions.
    static func servingCases() -> [(name: String, state: String, questions: [KevQuestion])] {
        let department = KevQuestion.choice(
            "Which team should handle this?",
            options: [
                ("returns", "Exchanges, refunds, wrong or damaged items"),
                ("shipping", "Delivery status, delays, lost packages"),
                ("billing", "Charges, invoices, payment problems"),
            ])
        let returnReason = KevQuestion.choice(
            "If the customer wants to return something, why?",
            options: [
                ("wrong_size", "The item doesn't fit"), ("wrong_item", "A different product was delivered"),
                ("damaged", "The item arrived broken or faulty"),
                ("changed_mind", "The item is fine, the customer no longer wants it"),
                ("other", "A return reason that fits none of the above"),
            ])
        let resolution = KevQuestion.choice(
            "What does the customer want to happen?",
            options: [
                ("exchange", "Swap the item for a different one"), ("refund", "Money back"),
                ("replacement", "The same item sent again"), ("information", "Just an answer, no action needed"),
            ])
        let tone = KevQuestion.choice(
            "What is the customer's tone?", options: [("calm", nil), ("frustrated", nil), ("angry", nil)])
        let escalate = KevQuestion.noul("Does this message require urgent human attention?")
        let frustration = KevQuestion.score(
            "How frustrated is the customer?", levels: ["Calm", "Frustrated", "Very angry"])
        let ticket =
            "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?"
        let paragraph =
            "I ordered a pair of running shoes on the first of the month and paid with my credit card. The confirmation email said "
            + "delivery in three to five business days, but the tracking page did not update for over a week, and when the package "
            + "finally arrived the box was crushed on one side. The shoes inside were a size ten instead of the size nine I ordered. "
        let five = [department, returnReason, resolution, escalate, frustration]
        return [
            ("2 questions, short state", ticket, [department, escalate]),
            ("6 questions, short state", ticket, [department, returnReason, resolution, tone, escalate, frustration]),
            ("5 questions, 370-token state", String(repeating: paragraph, count: 5), five),
            ("5 questions, 2,200-token state", String(repeating: paragraph, count: 30), five),
        ]
    }

    static func serving(answer: @Sendable (String, [KevQuestion]) async throws -> [KevAnswer]) async throws {
        for (name, state, questions) in servingCases() {
            do {
                _ = try await answer(state, questions)
            } catch {
                print("\(name): \(error)")
                continue
            }
            var times: [Double] = []
            for i in 1...22 {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = try await answer("Ticket \(i). \(state)", questions)
                if i > 2 { times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6) }
            }
            times.sort()
            print(
                "\(name.padding(toLength: 32, withPad: " ", startingAt: 0)) median \(String(format: "%.1f", times[times.count / 2])) ms"
            )
        }
    }

    static func serving(directory: URL) async throws {
        let manager = try await KevManager.load(from: directory)
        for (name, state, questions) in servingCases() {
            _ = try await manager.answer(state: state, questions: questions, maxStateTokens: 8192)
            for concurrent in [false, true] {
                var times: [Double] = []
                for i in 1...22 {
                    let text = "Ticket \(i). \(state)"
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ =
                        concurrent
                        ? try await manager.answerConcurrently(state: text, questions: questions, maxStateTokens: 8192)
                        : try await manager.answer(state: text, questions: questions, maxStateTokens: 8192)
                    if i > 2 { times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6) }
                }
                times.sort()
                print(
                    "\(name.padding(toLength: 32, withPad: " ", startingAt: 0)) \(concurrent ? "concurrent" : "sequential")  median \(String(format: "%.1f", times[times.count / 2])) ms"
                )
            }
        }
    }
}
