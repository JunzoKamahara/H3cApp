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

    func testStepCountsAreRefusedUpFront() {
        XCTAssertNil(PiH3Rules.problem(adapterName: "a", steps: 2))
        XCTAssertNil(PiH3Rules.problem(adapterName: "a", steps: 4))
        for steps in [3, 5, 8, 20] {
            let problem = PiH3Rules.problem(adapterName: "pi_h3_4step", steps: steps)
            XCTAssertNotNil(problem, "\(steps) steps")
            XCTAssertTrue(problem?.contains("pi_h3_4step") ?? false)
        }
        // The app's own range reaches the adapter's smallest count.
        XCTAssertTrue(stepsRange.contains(2))
    }

    func testAnAdaptersOwnStepCountIsRequired() {
        XCTAssertNil(PiH3Rules.problem(adapterName: "a", steps: 4, adapterSteps: 4))
        XCTAssertNil(PiH3Rules.problem(adapterName: "a", steps: 2, adapterSteps: 2))
        let two = PiH3Rules.problem(adapterName: "pi_h3_2step", steps: 4, adapterSteps: 2)
        XCTAssertNotNil(two, "a 2-step adapter at 4 steps")
        XCTAssertNotNil(PiH3Rules.problem(adapterName: "a", steps: 2, adapterSteps: 4))
        // A count the name does not give, or one that is not pi-H3's, leaves
        // only the 2-or-4 rule.
        XCTAssertNil(PiH3Rules.problem(adapterName: "a", steps: 2, adapterSteps: nil))
        XCTAssertNil(PiH3Rules.problem(adapterName: "a", steps: 4, adapterSteps: 8))
        XCTAssertNotNil(PiH3Rules.problem(adapterName: "a", steps: 8, adapterSteps: 8))
    }

    func testDiffusersFileIsListedByItsFolder() {
        let diffusers = "/models/pi_h3_2step_adapter/diffusion_pytorch_model.safetensors"
        XCTAssertEqual(LoRAEntry.listedName(forFileAt: diffusers), "pi_h3_2step_adapter")
        XCTAssertEqual(LoRAEntry.detectedSteps(fromFileName: LoRAEntry.listedName(forFileAt: diffusers)), 2)
        let single = "/models/pi_h3_4step_dwd_adapter_int8_convrot.safetensors"
        XCTAssertEqual(LoRAEntry.listedName(forFileAt: single), "pi_h3_4step_dwd_adapter_int8_convrot.safetensors")
        // Moved into the LoRA folder, it is named after the folder it came
        // from, so the list still tells it apart and reads its steps.
        XCTAssertEqual(ModelLibrary.loraFolderStem(forFileAt: diffusers), "pi_h3_2step_adapter")
        XCTAssertEqual(LoRAEntry.detectedSteps(
            fromFileName: LoRAEntry.listedName(forFileAt: "/LoRA/pi_h3_2step_adapter.safetensors")), 2)
        XCTAssertEqual(ModelLibrary.loraFolderStem(forFileAt: single), "pi_h3_4step_dwd_adapter_int8_convrot")
    }

    /// The adapter without the pixel head comes only in the diffusers
    /// layout; H3_PI_H3_LATENT_ADAPTER names its file.
    func testDiffusersAdapterIsRecognised() throws {
        let path = ProcessInfo.processInfo.environment["H3_PI_H3_LATENT_ADAPTER"]
        try XCTSkipIf(path == nil, "set H3_PI_H3_LATENT_ADAPTER to pi_h3_*_adapter/diffusion_pytorch_model.safetensors")
        let info = try H3LoRAInfo.inspect(path: path!)
        XCTAssertEqual(info.piH3, .latent)
        XCTAssertEqual(info.rankMax, 64)
        XCTAssertEqual(info.blocks, 50)
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
