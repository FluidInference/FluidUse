import Foundation

/// A deterministic backlog of fictional support tickets (mock data, no real customers): issue templates × products ×
/// details × openers / closers, shuffled with a fixed seed so every run shows the same queue.
enum MockTickets {
    static let plans = ["Pro plan", "Team plan", "Starter plan", "Business plan"]
    static let products = ["mobile app", "desktop app", "dashboard", "web app", "browser extension"]
    static let features = [
        "dark mode", "CSV export", "Spanish language support", "date filters in search", "a Slack integration",
        "bulk editing", "keyboard shortcuts", "an offline mode", "custom report templates",
        "two-factor via authenticator apps",
    ]
    static let amounts = ["$12", "$29", "$49", "$99", "$240", "$1,200"]
    static let times = [
        "since this morning", "for the last hour", "since yesterday", "since the last update", "for two days",
    ]

    static let openers = ["", "Hi, ", "Hello team, ", "Hey, ", "Urgent: ", "Quick one: "]
    static let closers = [
        "", " Thanks.", " Please help.", " Can someone look at this today?", " This is really frustrating.",
    ]

    /// Templates: {plan} {product} {feature} {amount} {time}. Templates starting with "+" are positive: no complaint closer.
    static let templates = [
        "I was charged twice for my {plan} this month. Can you refund the duplicate {amount}?",
        "Our invoice shows {amount} but we downgraded to the {plan} weeks ago.",
        "Can I get an invoice with our VAT number for last quarter?",
        "I cancelled my {plan} but was still billed {amount}. I want my money back.",
        "Is there a discount for nonprofits on the {plan}?",
        "We were billed for 50 seats but only have 12 users. Please fix and credit us.",
        "The {product} has been throwing 500 errors {time} and nobody on our team can log in.",
        "Our checkout is down {time} and customers can't pay.",
        "Webhooks stopped firing for new orders {time}, nothing in the logs.",
        "The {product} crashes every time I open the calendar tab.",
        "API requests started timing out {time} after we upgraded the SDK.",
        "Exports from the {product} come out empty {time}.",
        "Sync between the {product} and our calendar has been broken {time}.",
        "I forgot my password and the reset email never arrives.",
        "Someone logged into my account from another country last night. Please lock it.",
        "How do I change the email address on my account? I left my old job.",
        "Two-factor codes aren't arriving by SMS and I'm locked out.",
        "Please delete my account and all my data.",
        "I need to transfer ownership of our workspace to a colleague.",
        "+Would love {feature} in the {product}.",
        "Do you support {feature}?",
        "Any plans to add {feature}? Half our team would use it.",
        "Is {feature} on the roadmap for the {product}?",
        "Your latest update deleted my saved drafts. Two weeks of work gone.",
        "+Thanks for the fast fix yesterday, everything works great now!",
    ]

    static func make(count: Int = 1000, seed: UInt64 = 7) -> [String] {
        var generator = SplitMix64(seed: seed)
        var tickets: [String] = []
        var seen: Set<String> = []
        while tickets.count < count {
            var text = templates[Int(generator.next() % UInt64(templates.count))]
            let positive = text.hasPrefix("+")
            if positive { text.removeFirst() }
            for (slot, values) in [
                ("{plan}", plans), ("{product}", products), ("{feature}", features), ("{amount}", amounts),
                ("{time}", times),
            ] {
                text = text.replacingOccurrences(of: slot, with: values[Int(generator.next() % UInt64(values.count))])
            }
            var opener = openers[Int(generator.next() % UInt64(openers.count))]
            if !opener.isEmpty, let first = text.first {
                // "Hi, the dashboard…" but keep "API…" and "I…" capitalised
                let second = text.dropFirst().first
                if second?.isLowercase == true { text = first.lowercased() + text.dropFirst() }
            } else {
                opener = ""
            }
            let closer = closers[Int(generator.next() % UInt64(closers.count))]
            let full = opener + text + (positive ? "" : closer)
            if seen.insert(full).inserted { tickets.append(full) }
        }
        return tickets
    }
}

/// Small deterministic PRNG (the mock queue must be identical across runs).
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
