import Foundation
import XCTest

@testable import FluidUse

final class GLiClassModelStoreTests: XCTestCase {
    private var publishedConfig: GLiClassModelStore.RepositoryConfig {
        get throws {
            let url = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/gliclass-hub-config.json")
            return try JSONDecoder().decode(GLiClassModelStore.RepositoryConfig.self, from: Data(contentsOf: url))
        }
    }

    func testPublishedConfigSelectsAvailablePackages() throws {
        let config = try publishedConfig
        XCTAssertEqual(
            try config.package(length: 128, precision: "lut8"),
            "gliclass_edge_apps_lut8_kmeans_per_tensor_L128_options25.mlpackage")
        XCTAssertEqual(
            try config.package(length: 256, precision: "fp16"),
            "gliclass_edge_apps_fp16_L256_options25.mlpackage")
        XCTAssertThrowsError(try config.package(length: 256, precision: "lut8"))
        XCTAssertThrowsError(try config.package(length: 1024, precision: "fp16"))
        XCTAssertThrowsError(try config.package(length: 128, precision: "lut6"))
    }

    func testPublishedConfigChecksumMatchesHubManifest() throws {
        let fixture = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/gliclass-hub-config.json")
        XCTAssertEqual(
            GLiClassModelStore.sha256(try Data(contentsOf: fixture)),
            "350f819e00eb40ae5415458c320ca2e8dab2c226961ec4342f46de6b1299663f")
    }
}
