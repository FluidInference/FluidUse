import Game2048
import Testing

@testable import Decision2048BenchDemo

@MainActor
struct Decision2048BenchModelTests {
    @Test func defaultsToRawTopTwoComparisonSeed() {
        let model = Decision2048BenchModel()

        #expect(model.seed == 3)
        #expect(model.moveDelayMs == 10)
        #expect(model.gliClass.board == model.laya.board)
        #expect(model.gliClass.detail.contains("1 call/move"))
        #expect(model.laya.detail.contains("2 calls/move"))
    }
}
