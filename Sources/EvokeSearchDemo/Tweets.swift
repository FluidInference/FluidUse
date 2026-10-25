import SwiftUI

/// Fictional accounts and posts for the mock timeline.
struct Tweet: Identifiable, Sendable {
    let id: Int
    let name: String
    let handle: String
    let age: String
    let text: String
    let replies: Int
    let reposts: Int
    let likes: Int
    let hue: Double
}

enum MockTimeline {
    private struct Post: Decodable {
        let text: String
    }

    private static let firstNames = [
        "Alex", "Sam", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Jamie", "Avery", "Quinn", "Rowan", "Skyler",
        "Drew", "Parker", "Reese", "Emerson", "Hayden", "Logan", "Peyton", "Sage",
    ]
    private static let lastNames = [
        "Rivera", "Chen", "Patel", "Nguyen", "Kowalski", "Okafor", "Silva", "Haddad", "Larsen", "Moreau", "Tanaka",
        "Reyes", "Fischer", "Novak", "Ibrahim", "Costa", "Walsh", "Yilmaz", "Sato", "Mensah",
    ]

    /// Posts from a JSON array of `{"text": …}` (see `fetch-posts.py`) with fictional authors;
    /// falls back to the built-in 40 when the file is missing.
    static func load(from url: URL) -> [Tweet] {
        guard let data = try? Data(contentsOf: url), let posts = try? JSONDecoder().decode([Post].self, from: data)
        else { return tweets }
        return posts.enumerated().map { i, post in
            let first = firstNames[i % firstNames.count]
            let last = lastNames[(i / firstNames.count + i * 7) % lastNames.count]
            let minutes = (i * 37) % 2880 + 1
            return Tweet(
                id: i, name: "\(first) \(last)", handle: "\(first)\(last)\(i % 97)".lowercased(),
                age: minutes < 60 ? "\(minutes)m" : "\(minutes / 60)h", text: post.text,
                replies: (i * 37 + 11) % 240, reposts: (i * 53 + 7) % 900, likes: (i * 211 + 41) % 9800,
                hue: Double((i * 47) % 360) / 360)
        }
    }

    // (name, handle, age, text)
    private static let rows: [(String, String, String, String)] = [
        (
            "Priya Raman", "priyabuilds", "2m",
            "Rented an H100 for $1.89/hr overnight and fine-tuned the whole thing before breakfast. Wild that this used to cost a lab budget."
        ),
        (
            "Marco Lenz", "marcolenz", "14m",
            "Spot instances are back. Grabbed eight A100s at a third of on-demand pricing for the eval run."
        ),
        (
            "Dana Okafor", "danaruns", "21m",
            "Crossed the finish line of my first 26.2 today. Legs are gone, heart is full."
        ),
        (
            "Leo Park", "leoparkdev", "33m",
            "Finally rewrote the hot loop in Rust. p99 went from 40ms to 6ms and I can sleep again."
        ),
        (
            "Hannah Weiss", "hannahwcooks", "41m",
            "Sunday project: sourdough with a 72-hour cold ferment. The crumb is unreal."
        ),
        (
            "Sam Ortiz", "samotravels", "52m",
            "Booked Lisbon to Tokyo with one stop for under 600 bucks. Error fare or not, I'm going."
        ),
        (
            "Grace Lin", "gracelin_md", "1h",
            "Reminder for your parents: the updated flu and RSV shots are recommended for everyone over 60. Pharmacies have walk-ins."
        ),
        (
            "Owen Hale", "owenhale", "1h",
            "Our pupper finally learned to sit and stay without treats. Six weeks of patience paid off."
        ),
        (
            "Ines Duarte", "inesduarte", "1h",
            "Groceries are up again this month. Eggs, coffee, olive oil, everything. My paycheck did not get the memo."
        ),
        ("Ravi Menon", "ravimenon", "2h", "The Fed held rates steady. Mortgage people, hold on a bit longer."),
        (
            "Chloe Martin", "chloeplants", "2h",
            "Repotted the monstera and it immediately threw out a new leaf. Plants are dramatic."
        ),
        ("Tom Becker", "tombecker", "2h", "Every on-call shift is the same: DNS. It is always DNS."),
        (
            "Aisha Bello", "aishabello", "2h",
            "Started lifting three times a week in January. Deadlift went from 60 to 110 kg. Strength training changed how I sleep."
        ),
        (
            "Ken Watanabe", "kenwphoto", "3h",
            "Caught the aurora over the lake at 2am. Camera settings: 8 seconds, f/1.8, ISO 1600."
        ),
        (
            "Maya Cohen", "mayacohen", "3h",
            "My cardiologist says: walk 30 minutes a day, cut the salt, and keep your cholesterol in check. Simple, not easy."
        ),
        (
            "Felix Grant", "felixgrant", "3h",
            "Taught my 9-year-old Python with a turtle drawing game. She made a spiral and screamed."
        ),
        ("Nora Quinn", "noraquinn", "4h", "Heat wave number four this summer. The city opened cooling centers again."),
        (
            "Ben Adler", "benadler", "4h",
            "Switched the whole team from Jira to a plain markdown file in the repo. Velocity unclear, morale way up."
        ),
        ("Lucia Romero", "luciaromero", "4h", "Wildfire smoke rolled in overnight. AQI is 180, masks back on."),
        (
            "Jake Morris", "jakemorris", "5h",
            "Local model on my laptop now writes better commit messages than I do. No cloud, no API key."
        ),
        (
            "Emma Novak", "emmanovak", "5h",
            "Our rescue cat has decided the keyboard is her bed. Productivity: zero. Happiness: high."
        ),
        (
            "Yusuf Kaya", "yusufkaya", "5h",
            "Swapped coffee for green tea for a month. Fewer crashes in the afternoon, same focus."
        ),
        (
            "Olivia Brooks", "oliviabrooks", "6h",
            "Paid off the last of my student loans today. Eleven years. I cried in the bank app."
        ),
        (
            "Daniel Reyes", "danreyes", "6h",
            "The new phone chip runs a 3B model at 30 tokens a second fully offline. On-device is here."
        ),
        (
            "Sofia Rossi", "sofiarossi", "6h",
            "Made my nonna's ragu for the first time. Four hours, one pot, zero regrets."
        ),
        (
            "Arjun Shah", "arjunshah", "7h",
            "Took the train from Paris to Milan instead of flying. Mountains the whole way and I got work done."
        ),
        (
            "Mia Fischer", "miafischer", "7h",
            "Seven hours of sleep for 30 days straight. Resting heart rate dropped by 6 bpm."
        ),
        (
            "Chris Young", "chrisyoung", "8h",
            "Postgres full text search got us 90% of the way. Did not need a separate search cluster after all."
        ),
        ("Zara Ahmed", "zaraahmed", "8h", "Booked a cabin with no wifi for the weekend. Phone stays in the glovebox."),
        (
            "Paul Dubois", "pauldubois", "9h",
            "Rent went up 18% at renewal. Looking at moving further out and commuting."
        ),
        (
            "Kim Tran", "kimtran", "9h",
            "Our golden retriever passed her therapy dog certification! She visits the children's hospital on Fridays."
        ),
        (
            "Ethan Clark", "ethanclark", "10h",
            "Battery storage on the grid hit a new record during the evening peak. Solar plus batteries is the story of the decade."
        ),
        (
            "Ruth Adams", "ruthadams", "10h",
            "Grandma got her shingles vaccine this week. Painless, she said, and she's 84."
        ),
        (
            "Ivan Petrov", "ivanpetrov", "11h",
            "Quantized the 7B to 4-bit and it fits in 5 GB. Quality loss is barely noticeable on our evals."
        ),
        (
            "Lily Evans", "lilyevans", "12h",
            "Hiked the ridge trail to the summit at sunrise. 1,100 m of climbing and the clouds were below us."
        ),
        (
            "Omar Haddad", "omarhaddad", "12h",
            "Index funds, automatic monthly contributions, don't check it. That's the whole plan."
        ),
        (
            "Jess Park", "jesspark", "13h",
            "First week at the bootcamp done. I can finally read JavaScript without panicking."
        ),
        (
            "Victor Hugo Lima", "victorlima", "14h",
            "Flooding on the coast road again after the storm surge. Sea walls are not keeping up."
        ),
        (
            "Anna Berg", "annaberg", "15h",
            "Meal prep Sunday: lentil curry, roasted veg, rice. Eight lunches for under 20 dollars."
        ),
        (
            "Noah Silva", "noahsilva", "16h",
            "Benchmarks are in: the open model matches the closed one on our retrieval set at a twentieth of the size."
        ),
    ]

    static let tweets: [Tweet] = rows.enumerated().map { i, row in
        Tweet(
            id: i, name: row.0, handle: row.1, age: row.2, text: row.3,
            replies: (i * 37 + 11) % 240, reposts: (i * 53 + 7) % 900, likes: (i * 211 + 41) % 9800,
            hue: Double((i * 47) % 360) / 360)
    }
}
