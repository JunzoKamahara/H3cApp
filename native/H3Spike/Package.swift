// swift-tools-version:5.9
import Foundation
import PackageDescription

// Resolve the h3.c engine repo root (two levels up from this package: native/H3Spike -> native -> repo root)
// so the linker flags below work regardless of which directory `swift build` is invoked from.
let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let repoRoot = packageDir.deletingLastPathComponent().deletingLastPathComponent().path

// Optional, opt-in: when CCV_DIR is set (matching the same env var the
// top-level Makefile reads - see tools/ccv_eval/README.md), link in the
// experimental H3_ATTENTION_BACKEND=ccv_dense path. Unset (the default),
// this package builds exactly as before - `swift build` doesn't read
// CCV_DIR from the shell unless told to (SwiftPM only forwards variables
// SwiftPM itself defines), so pass it explicitly:
// `CCV_DIR=/path swift build -c release` (after `make libh3.a CCV_DIR=/path`
// at the repo root at least once, which also builds the .o referenced
// below as an order-only prerequisite).
//
// The backend's own object file (h3_gpu_ccv_attention.o) is referenced
// directly here rather than through libh3.a: it isn't part of that archive
// on purpose (see the Makefile's comment) since nothing references its
// symbols directly - it self-registers via a static constructor - and a
// static archive only links in a member that resolves some other object's
// undefined symbol, so archived it would be silently dropped by `-lh3`. A
// loose object file passed directly to the linker is always included.
let ccvDir = ProcessInfo.processInfo.environment["CCV_DIR"]
var h3LinkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-L\(repoRoot)/build", "-lh3upscale", "-lh3", "-licucore"]),
    .linkedFramework("Foundation"),
    .linkedFramework("Metal"),
    .linkedFramework("MetalPerformanceShaders"),
    .linkedFramework("MetalPerformanceShadersGraph"),
    .linkedFramework("Accelerate"),
    .linkedFramework("AVFoundation"),
    .linkedFramework("CoreMedia"),
    .linkedFramework("CoreVideo"),
    .linkedFramework("CoreAudio"),
    .linkedFramework("CoreGraphics"),
    .linkedFramework("ImageIO"),
]
if let ccvDir {
    h3LinkerSettings.append(.unsafeFlags([
        "\(repoRoot)/build/obj/h3_gpu_ccv_attention.o",
        "\(ccvDir)/lib/libccv.a", "-lblas", "-lc++",
    ]))
    h3LinkerSettings.append(.linkedFramework("CoreML"))
    h3LinkerSettings.append(.linkedFramework("IOSurface"))
    h3LinkerSettings.append(.linkedFramework("QuartzCore"))
}

let package = Package(
    name: "H3Spike",
    platforms: [.macOS("15.0")],
    targets: [
        .target(
            name: "CH3",
            path: "Sources/CH3"
        ),
        .target(
            name: "H3Engine",
            dependencies: ["CH3"],
            path: "Sources/H3Engine"
        ),
        // The upscalers' Swift wrappers (SwiftVR, SeedVR2, the latent upscaler):
        // apart from the generation engine's, as libh3upscale.a is from libh3.a.
        .target(
            name: "H3Upscale",
            dependencies: ["CH3"],
            path: "Sources/H3Upscale"
        ),
        .executableTarget(
            name: "H3Spike",
            dependencies: ["CH3"],
            path: "Sources/H3Spike",
            linkerSettings: h3LinkerSettings
        ),
        .executableTarget(
            name: "H3cApp",
            dependencies: ["H3Engine", "H3Upscale"],
            path: "Sources/H3cApp",
            linkerSettings: h3LinkerSettings
        ),
    ]
)
