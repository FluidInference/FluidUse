import XCTest

@testable import FluidUse

final class CuaS1FourBTests: XCTestCase {
    private func snakeCaseDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    private struct PromptFixture: Decodable {
        struct Option: Decodable {
            let elementId: String
            let role: String
            let label: String
            let action: String
            let entityId: String?
        }
        let app: String
        let taskFamily: String
        let goal: String?
        let axTree: String?
        let options: [Option]
        let chat: String
    }

    /// Chat string rendered by `cua_s1.four_b.build_prompt` + the Qwen3.5 chat template.
    func testChatMatchesReference() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "cua-s1-4b-prompt", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try snakeCaseDecoder().decode(PromptFixture.self, from: Data(contentsOf: url))
        let state = CuaS1FourBState(
            app: fixture.app, taskFamily: fixture.taskFamily, goal: fixture.goal,
            accessibilityTree: fixture.axTree,
            options: fixture.options.map {
                CuaS1FourBOption(
                    elementId: $0.elementId, role: $0.role, label: $0.label, action: $0.action, entityId: $0.entityId)
            })
        XCTAssertEqual(try CuaS1FourBPrompt.chat(state: state, modality: .text), fixture.chat)
    }

    func testFillOptionNamesEntityAndMultimodalPlaceholder() throws {
        let state = CuaS1FourBState(
            app: "portal", taskFamily: "form_filling", goal: "Sign up",
            options: [
                CuaS1FourBOption(elementId: "el_0", role: "Edit", label: "Email", action: "fill", entityId: "ent_1"),
                CuaS1FourBOption(elementId: "el_0", role: "Edit", label: "Email", action: "skip"),
            ])
        let chat = try CuaS1FourBPrompt.chat(state: state, modality: .multimodal)
        XCTAssertTrue(chat.contains("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Goal: Sign up\n\n"))
        XCTAssertTrue(chat.contains("A. Edit \"Email\" -> fill (with entity 'ent_1')\nB. Edit \"Email\" -> skip\n"))
        XCTAssertTrue(chat.contains("The current screenshot is attached.\n\n"))
        XCTAssertTrue(chat.hasSuffix("<|im_start|>assistant\n<think>\n"))
        XCTAssertThrowsError(try CuaS1FourBPrompt.chat(state: state, modality: .text))
    }

    func testRejectsMoreThanTwentySixOptions() {
        let options = (0..<27).map {
            CuaS1FourBOption(elementId: "el_\($0)", role: "Button", label: "B", action: "skip")
        }
        let state = CuaS1FourBState(app: "a", taskFamily: "f", accessibilityTree: "-", options: options)
        XCTAssertThrowsError(try CuaS1FourBPrompt.chat(state: state, modality: .text))
    }

    /// `Qwen3_5Model.get_rope_index` on 3 text tokens, a 4x6-patch image (2x3 merged) and 2 text tokens.
    func testMRopePositionsMatchReference() {
        let pad = 248_056
        let ids = [1, 2, 3] + Array(repeating: pad, count: 6) + [4, 5]
        let positions = CuaS1FourBManager.mropePositions(
            ids: ids, padTokenId: pad, imageTokens: 6, gridRows: 2, gridCols: 3)
        XCTAssertEqual(
            positions,
            [
                [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 4], [3, 3, 5], [3, 4, 3], [3, 4, 4], [3, 4, 5],
                [6, 6, 6], [7, 7, 7],
            ])
    }

    /// `smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=...)` from transformers.
    func testSmartResizeMatchesReference() {
        let cases: [(Int, Int, Int, Int, Int)] = [
            (580, 760, 16_777_216, 576, 768), (1056, 760, 16_777_216, 1056, 768), (316, 760, 16_777_216, 320, 768),
            (1080, 1920, 16_777_216, 1088, 1920), (1080, 1920, 1_048_576, 768, 1344), (100, 100, 16_777_216, 256, 256),
        ]
        for (h, w, maxPixels, wantH, wantW) in cases {
            let got = CuaS1FourBVision.smartResize(
                height: h, width: w, factor: 32, minPixels: 65_536, maxPixels: maxPixels)
            XCTAssertEqual(got.height, wantH, "\(h)x\(w)")
            XCTAssertEqual(got.width, wantW, "\(h)x\(w)")
        }
    }
}
