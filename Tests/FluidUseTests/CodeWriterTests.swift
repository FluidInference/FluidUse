import XCTest

@testable import FluidUse

final class CodeWriterTests: XCTestCase {
    func testExtractCodeTakesTheFirstFencedBlock() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("CodeWriterManager needs macOS 15") }
        let text =
            "Here you go:\n\n```python\ndef f(x):\n    return x[::-1]\n```\n\nIt reverses.\n```python\nprint(1)\n```"
        XCTAssertEqual(CodeWriterManager.extractCode(text), "def f(x):\n    return x[::-1]")
    }

    func testExtractCodeKeepsAnUnclosedBlockAndPlainText() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("CodeWriterManager needs macOS 15") }
        // Mid-stream (or cut off at the token limit): no closing fence yet.
        XCTAssertEqual(CodeWriterManager.extractCode("```python\ndef f():\n    retu"), "def f():\n    retu")
        XCTAssertEqual(CodeWriterManager.extractCode("```\nx = 1\n```"), "x = 1")
        XCTAssertEqual(CodeWriterManager.extractCode("  def g(): pass\n"), "def g(): pass")
    }

    /// The Swift prompt must tokenize exactly as `apply_chat_template(..., add_generation_prompt=True)` in
    /// transformers for Qwen/Qwen2.5-Coder-0.5B-Instruct.
    func testPromptTokensMatchPythonReference() throws {
        guard let directory = ProcessInfo.processInfo.environment["CODE_WRITER_MODEL_DIR"], !directory.isEmpty else {
            throw XCTSkip("Set CODE_WRITER_MODEL_DIR (folder with tokenizer.json) to run")
        }
        let tokenizer = try QwenBPETokenizer(
            tokenizerJsonURL: URL(fileURLWithPath: directory).appendingPathComponent("tokenizer.json"))
        let text =
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\nWrite a function to reverse each string in a list.\nYour code should pass this test:\n"
            + "assert rev([\"ab\"]) == [\"ba\"]<|im_end|>\n<|im_start|>assistant\n"
        let reference = [
            151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847,
            13,
            151645, 198, 151644, 872, 198, 7985, 264, 729, 311, 9931, 1817, 914, 304, 264, 1140, 624, 7771, 2038, 1265,
            1494, 419, 1273, 510, 2207, 5772, 19065, 370, 14013, 621, 4383, 4645, 1341, 151645, 198, 151644, 77091, 198,
        ]
        XCTAssertEqual(try tokenizer.encode(text), reference)
    }
}
