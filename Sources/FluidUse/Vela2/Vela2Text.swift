import Foundation

/// Python-exact text rules of the Vela 2.0 engine (`vela2_inference.py`): the word units spans are read on (`_WORD`), the
/// units the span decoder votes over (`_UNIT`), edge / URL trimming and the decoder itself. Written as scanners rather
/// than `NSRegularExpression` because ICU's `\w` / `\s` differ from Python's (combining marks, `No` digits, U+001C…).
/// All offsets are Unicode scalar indices, as Python's.
enum Vela2Text {
    // MARK: character classes

    /// The engine's `_CJK` class as it actually compiles (one range starts at U+8C48 in the release source).
    static func isCJK(_ c: Unicode.Scalar) -> Bool {
        switch c.value {
        case 0x2E80...0x2FFF, 0x3000...0x303F, 0x3040...0x30FF, 0x3100...0x31FF, 0x3400...0x4DBF, 0x4E00...0x9FFF,
            0xA000...0xA4CF, 0xAC00...0xD7AF, 0x8C48...0xFAFF, 0xFE30...0xFE4F, 0xFF00...0xFFEF:
            return true
        default: return false
        }
    }

    /// Python `str.isspace` (= `re`'s `\s` for str patterns).
    static func isSpace(_ c: Unicode.Scalar) -> Bool {
        switch c.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
            return true
        default: return false
        }
    }

    /// Python `str.isalnum` for one character: a letter (L*) or any numeric character.
    static func isAlnum(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return c.properties.numericType != nil
        }
    }

    /// `re`'s `\w` for str patterns.
    static func isWord(_ c: Unicode.Scalar) -> Bool { c == "_" || isAlnum(c) }

    /// ASCII case folding as `re.IGNORECASE` applies it to ASCII classes (incl. the Kelvin sign, long s, dotted / dotless i).
    static func fold(_ c: Unicode.Scalar) -> Unicode.Scalar {
        switch c.value {
        case 0x41...0x5A: return Unicode.Scalar(c.value + 32)!
        case 0x212A: return "k"
        case 0x017F: return "s"
        case 0x0130, 0x0131: return "i"
        default: return c
        }
    }

    static func isAsciiLetter(_ c: Unicode.Scalar) -> Bool { ("a"..."z").contains(fold(c)) }
    static func isAsciiDigit(_ c: Unicode.Scalar) -> Bool { ("0"..."9").contains(c) }

    static let trail: Set<Unicode.Scalar> = [".", ",", ";", ":", "!", "?", ")", "]", "}", ">", "\"", "'", "\u{201D}", "\u{2019}", "\u{BB}"]
    static let unitTrail: Set<Unicode.Scalar> = [".", ",", ";", ":", "!", "?"]
    static let urlTrail: Set<Unicode.Scalar> = [".", ",", ";", ":", "!", "?", ")"]
    static let edge: Set<Unicode.Scalar> = Set("[](){}<>「」『』【】《》〈〉\"'“”‘’«»‹›".unicodeScalars)
    static let unitBodyExcluded: Set<Unicode.Scalar> = Set("])}>「」『』\"'“”".unicodeScalars)

    /// `(?:https?://|www\.)` at `i`; returns the index after the prefix.
    static func urlPrefix(_ s: [Unicode.Scalar], _ i: Int, ignoreCase: Bool) -> Int? {
        func lit(_ word: String) -> Int? {
            let w = Array(word.unicodeScalars)
            guard i + w.count <= s.count else { return nil }
            for k in 0..<w.count where (ignoreCase ? fold(s[i + k]) : s[i + k]) != w[k] { return nil }
            return i + w.count
        }
        return lit("https://") ?? lit("http://") ?? lit("www.")
    }

    /// A lazy URL body followed by `[trail]*` and then a non-body character: the body run minus trailing `trail`
    /// characters, at least one character long.
    static func urlBody(_ s: [Unicode.Scalar], from p: Int, body: (Unicode.Scalar) -> Bool, trail: Set<Unicode.Scalar>) -> Int? {
        var run = p
        while run < s.count, body(s[run]) { run += 1 }
        guard run > p else { return nil }
        var end = run
        while end > p + 1, trail.contains(s[end - 1]) { end -= 1 }
        return end
    }

    // MARK: _WORD (GLiNER2 whitespace splitter + CJK per character + URL rule), re.IGNORECASE

    static func splitWords(_ s: [Unicode.Scalar]) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        var i = 0
        while i < s.count {
            if let end = wordMatch(s, i) {
                out.append((i, end))
                i = end
            } else {
                i += 1
            }
        }
        return out
    }

    static func wordMatch(_ s: [Unicode.Scalar], _ i: Int) -> Int? {
        let n = s.count
        // URL
        if let p = urlPrefix(s, i, ignoreCase: true),
            let end = urlBody(s, from: p, body: { !isSpace($0) && !isCJK($0) }, trail: trail)
        {
            return end
        }
        // e-mail: [a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        func local(_ c: Unicode.Scalar) -> Bool { isAsciiLetter(c) || isAsciiDigit(c) || "._%+-".unicodeScalars.contains(c) }
        func domain(_ c: Unicode.Scalar) -> Bool { isAsciiLetter(c) || isAsciiDigit(c) || c == "." || c == "-" }
        var j = i
        while j < n, local(s[j]) { j += 1 }
        if j > i, j < n, s[j] == "@" {
            let d0 = j + 1
            var m = d0
            while m < n, domain(s[m]) { m += 1 }
            var dot = m - 1
            while dot > d0 {
                if s[dot] == "." {
                    var k = dot + 1
                    while k < n, isAsciiLetter(s[k]) { k += 1 }
                    if k - dot - 1 >= 2 { return k }
                }
                dot -= 1
            }
        }
        // @handle
        if s[i] == "@" {
            var k = i + 1
            while k < n, isAsciiLetter(s[k]) || isAsciiDigit(s[k]) || s[k] == "_" { k += 1 }
            if k > i + 1 { return k }
        }
        if isCJK(s[i]) { return i + 1 }
        // [^\W CJK]+(?:[-_][^\W CJK]+)*
        func wordChar(_ c: Unicode.Scalar) -> Bool { isWord(c) && !isCJK(c) }
        if wordChar(s[i]) {
            var k = i
            while k < n, wordChar(s[k]) { k += 1 }
            while k + 1 < n, s[k] == "-" || s[k] == "_", wordChar(s[k + 1]) {
                k += 1
                while k < n, wordChar(s[k]) { k += 1 }
            }
            return k
        }
        if !isSpace(s[i]) { return i + 1 }
        return nil
    }

    /// `word_first_tokens`: words that overlap a token, with the index of the first overlapping token.
    static func wordFirstTokens(_ s: [Unicode.Scalar], offsets: [(Int, Int)]) -> (words: [(Int, Int)], first: [Int]) {
        let w = splitWords(s)
        guard !w.isEmpty, !offsets.isEmpty else { return ([], []) }
        var cummax: [Int] = []
        var running = Int.min
        for (st, en) in offsets {
            running = max(running, en > st ? en : -1)
            cummax.append(running)
        }
        var words: [(Int, Int)] = []
        var first: [Int] = []
        for (a, b) in w {
            var lo = 0
            var hi = cummax.count
            while lo < hi {  // searchsorted(side="right"): first index with cummax > a
                let mid = (lo + hi) / 2
                if cummax[mid] <= a { lo = mid + 1 } else { hi = mid }
            }
            var t = lo
            while t < offsets.count, offsets[t].1 <= offsets[t].0 { t += 1 }
            if t < offsets.count, offsets[t].0 < b {
                words.append((a, b))
                first.append(t)
            }
        }
        return (words, first)
    }

    // MARK: _UNIT and trimming

    static func units(_ s: [Unicode.Scalar]) -> [Int] {
        var u = [Int](repeating: -1, count: s.count + 1)
        var i = 0
        var k = 0
        func unitBody(_ c: Unicode.Scalar) -> Bool { !isSpace(c) && !unitBodyExcluded.contains(c) && !isCJK(c) }
        func unitChar(_ c: Unicode.Scalar) -> Bool { (!isCJK(c) && isAlnum(c)) || "@._-+".unicodeScalars.contains(c) }
        while i < s.count {
            var end: Int?
            if let p = urlPrefix(s, i, ignoreCase: false), let e = urlBody(s, from: p, body: unitBody, trail: unitTrail) {
                end = e
            } else if unitChar(s[i]) {
                var j = i
                while j < s.count, unitChar(s[j]) { j += 1 }
                end = j
            } else if !isSpace(s[i]) {
                end = i + 1
            }
            if let end {
                for x in i..<end { u[x] = k }
                k += 1
                i = end
            } else {
                i += 1
            }
        }
        return u
    }

    static func trim(_ s: [Unicode.Scalar], _ start: Int, _ end: Int) -> (Int, Int) {
        var a = start
        var b = end
        while a < b, isSpace(s[a]) || edge.contains(s[a]) { a += 1 }
        while b > a, isSpace(s[b - 1]) || edge.contains(s[b - 1]) { b -= 1 }
        guard urlPrefix(s, a, ignoreCase: true) != nil else { return (a, b) }
        if let cjk = (a..<b).first(where: { isCJK(s[$0]) }) { b = cjk }
        while b > a, urlTrail.contains(s[b - 1]) || edge.contains(s[b - 1]) || isSpace(s[b - 1]) { b -= 1 }
        return (a, b)
    }

    // MARK: decoder (span_decode_r4.decode_v2, extend=False, fill and unit vote on)

    struct Span {
        let start: Int
        let end: Int
        let label: Int
        let probability: Double
    }

    /// `probs` [words][labels] (sigmoid probabilities), `offsets` the words' character spans.
    static func decodeSpans(_ probs: [[Double]], offsets: [(Int, Int)], text s: [Unicode.Scalar], threshold thr: Double) -> [Span] {
        let T = offsets.count
        guard T > 0 else { return [] }
        let valid = offsets.map { $0.1 > $0.0 }
        var lab = [Int](repeating: -1, count: T)
        var pk = [Double](repeating: 0, count: T)
        for t in 0..<T {
            var best = 0
            for c in probs[t].indices where Float(probs[t][c]) > Float(probs[t][best]) { best = c }
            pk[t] = Double(Float(probs[t][best]))
            if valid[t], Float(pk[t]) > Float(thr) { lab[t] = best }  // numpy: float32 array vs a weak Python float
        }
        let u = units(s)
        var tu = [Int](repeating: -1, count: T)
        for t in 0..<T where valid[t] {
            var f = offsets[t].0
            while f < offsets[t].1, isSpace(s[f]) { f += 1 }
            tu[t] = f < offsets[t].1 && f < s.count ? u[f] : -1
        }
        var t = 0
        while t < T {
            if tu[t] < 0 {
                t += 1
                continue
            }
            var t2 = t
            while t2 + 1 < T, tu[t2 + 1] == tu[t] || !valid[t2 + 1] { t2 += 1 }
            let idx = (t...t2).filter { valid[$0] }
            let labelled = idx.filter { lab[$0] >= 0 }
            if !labelled.isEmpty {
                if Set(labelled.map { lab[$0] }).count > 1 {
                    var score: [Int: Double] = [:]
                    var order: [Int] = []
                    for i in labelled {
                        if score[lab[i]] == nil { order.append(lab[i]) }
                        score[lab[i], default: 0] += pk[i]
                    }
                    let winner = order.max { score[$0]! < score[$1]! }!  // Python max(): first of equal maxima
                    for i in labelled { lab[i] = winner }
                }
                for c in Set(labelled.map { lab[$0] }).sorted() {  // a Python set of small ints iterates ascending
                    let pos = labelled.filter { lab[$0] == c }
                    for i in idx where pos.first! < i && i < pos.last! && lab[i] < 0 { lab[i] = c }
                }
            }
            t = t2 + 1
        }
        var spans: [(c: Int, s: Int, e: Int, ts: [Int])] = []
        var cur: (c: Int, s: Int, e: Int, ts: [Int])?
        for t in 0..<T where valid[t] {
            let c = lab[t]
            if var open = cur, c == open.c {
                open.e = offsets[t].1
                open.ts.append(t)
                cur = open
                continue
            }
            if let open = cur { spans.append(open) }
            cur = c >= 0 ? (c, offsets[t].0, offsets[t].1, [t]) : nil
        }
        if let open = cur { spans.append(open) }
        var out: [Span] = []
        for span in spans {
            let (a, b) = trim(s, span.s, span.e)
            guard b > a else { continue }
            let mean = span.ts.reduce(0.0) { $0 + Double(Float(probs[$1][span.c])) } / Double(span.ts.count)
            out.append(Span(start: a, end: b, label: span.c, probability: mean))
        }
        return out
    }
}
