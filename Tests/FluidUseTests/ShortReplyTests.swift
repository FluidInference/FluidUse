import XCTest

@testable import FluidUse

final class ShortReplyTests: XCTestCase {
    func testCleanReplyMirrorsThePythonCleanup() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("ShortReplyManager needs macOS 15") }
        XCTAssertEqual(ShortReplyManager.cleanReply("Reply: Nice job!\nExtra text"), "Nice job!")
        XCTAssertEqual(ShortReplyManager.cleanReply("  \n- “Congrats on the 5K!”  "), "Congrats on the 5K!")
        XCTAssertEqual(ShortReplyManager.cleanReply(" \n "), "")
    }

    func testLinksAreStrippedFromPosts() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("ShortReplyManager needs macOS 15") }
        XCTAssertEqual(
            ShortReplyManager.stripLinks("Translation with pic.x.com/2eKrZU3eEr and https://example.com/x done"),
            "Translation with  and  done")
    }

    /// The Swift prompt must tokenize exactly as `reply.py` + `apply_chat_template(enable_thinking=False)`.
    /// Reference ids come from the Python tokenizer for the post "I finished my first 5K today." (78 tokens).
    func testPromptTokensMatchPythonReference() throws {
        guard let directory = ProcessInfo.processInfo.environment["SHORT_REPLY_MODEL_DIR"], !directory.isEmpty else {
            throw XCTSkip("Set SHORT_REPLY_MODEL_DIR (folder with tokenizer.json) to run")
        }
        let tokenizer = try QwenBPETokenizer(
            tokenizerJsonURL: URL(fileURLWithPath: directory).appendingPathComponent("tokenizer.json"))
        let system =
            "Write one natural, very short reply to the social post as a separate person. React to a concrete detail in "
            + "the post when possible. Do not invent facts or personal experiences. Use at most 12 words. Output only the reply."
        let text =
            "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\nPost: I finished my first 5K today.\nReply:<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let reference = [
            151644, 8948, 198, 7985, 825, 5810, 11, 1602, 2805, 9851, 311, 279, 3590, 1736, 438, 264, 8651, 1697, 13,
            3592,
            311, 264, 14175, 7716, 304, 279, 1736, 979, 3204, 13, 3155, 537, 17023, 13064, 476, 4345, 11449, 13, 5443,
            518,
            1429, 220, 16, 17, 4244, 13, 9258, 1172, 279, 9851, 13, 151645, 198, 151644, 872, 198, 4133, 25, 358, 8060,
            847,
            1156, 220, 20, 42, 3351, 624, 20841, 25, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
        ]
        XCTAssertEqual(try tokenizer.encode(text), reference)
        XCTAssertEqual(
            tokenizer.decode(try tokenizer.encode("Congrats on finishing your first 5K!")),
            "Congrats on finishing your first 5K!")
        XCTAssertEqual(tokenizer.decode([151645, 8948]), "system", "special tokens are dropped when decoding")
    }
}
