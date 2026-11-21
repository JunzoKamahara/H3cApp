import XCTest
import H3Engine
@testable import H3cApp

/// What the app reads from a pi-H3 adapter (Lakonik/pi-H3, the single-file
/// ComfyUI layout) and how it files the engine's "pixel head" progress.
/// The adapter test is skipped unless H3_PI_H3_ADAPTER names the file.
final class PiH3AdapterTests: XCTestCase {
    func testPixelHeadIsTheDecodeStage() {
        XCTAssertEqual(JobStage.stage(forPhase: "pixel head"), .decode)
        XCTAssertEqual(JobStage.stage(forPhase: "video VAE load"), .decode)
    }

    func testAdapterIsRecognised() throws {
        let path = ProcessInfo.processInfo.environment["H3_PI_H3_ADAPTER"]
        try XCTSkipIf(path == nil, "set H3_PI_H3_ADAPTER to pi_h3_*_dwd_adapter_int8_convrot.safetensors")
        let info = try H3LoRAInfo.inspect(path: path!)
        XCTAssertEqual(info.piH3, .pixels)
        XCTAssertEqual(info.rankMax, 64)
        XCTAssertEqual(info.blocks, 46)
        let name = URL(fileURLWithPath: path!).lastPathComponent
        XCTAssertTrue([2, 4].contains(LoRAEntry.detectedSteps(fromFileName: name) ?? 0),
                      "the file name gives the adapter's step count")
    }
}
