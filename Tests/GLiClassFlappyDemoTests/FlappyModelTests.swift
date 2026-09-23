import Testing

@testable import GLiClassFlappyDemo

struct FlappyModelTests {
    @MainActor
    @Test func everyModelControlSelectsItsOwnBackend() {
        #expect(FlappyModel.Control.gliclass.model == .gliclass)
        #expect(FlappyModel.Control.laya.model == .laya)
        #expect(FlappyModel.Control.gliner2Base.model == .gliner2Base)
        #expect(FlappyModel.Control.gliner2Multilingual.model == .gliner2Multilingual)
        #expect(FlappyModel.Control.heuristic.model == nil)
        #expect(FlappyModel.Control.manual.model == nil)
    }

    @MainActor
    @Test func manualLifecycleAndHeuristicWorkWithoutModel() {
        let model = FlappyModel()
        #expect(model.control.model == .gliclass)
        #expect(model.safetyGuard)
        #expect(!model.canPlay)
        model.control = .manual
        model.toggle()
        model.flap()
        model.advanceFrame()
        #expect(model.game.frames == 1)
        #expect(model.game.velocity < 0)
        model.pause()
        let frame = model.game.frames
        model.advanceFrame()
        #expect(model.game.frames == frame)
        model.reset()
        #expect(model.game.frames == 0)
        model.control = .heuristic
        model.toggle()
        model.advanceFrame()
        #expect(model.game.frames == 1)
        model.pause()
    }
}
