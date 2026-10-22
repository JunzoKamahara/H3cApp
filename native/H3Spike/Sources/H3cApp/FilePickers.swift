import AppKit
import AVFoundation
import UniformTypeIdentifiers

let imageTypes: [UTType] = [.image]
// Ref2VA reference slots accept a still image, a video, or an audio-only
// file (H3ReferenceKind.image/.video/.audio) - h3_av_reader.m decodes all
// three natively. An audio reference can't stand on its own though - h3.c
// rejects any reference set with audio but no image/video among them
// ("reference audio requires an image or video reference").
let referenceMediaTypes: [UTType] = [.image, .movie, .audio]
// Lip sync takes the audio track of an audio file or of a video.
let forcedAudioTypes: [UTType] = [.audio, .movie]
// "safetensors" has no registered system UTI, so this synthesizes a dynamic
// one from the extension - still filters the open panel correctly.
let safetensorsTypes: [UTType] = [UTType(filenameExtension: "safetensors") ?? .data]

func chooseFile(allowedContentTypes: [UTType], startingIn directory: String? = nil) -> String? {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowedContentTypes = allowedContentTypes
    if let directory { panel.directoryURL = URL(fileURLWithPath: directory, isDirectory: true) }
    return panel.runModal() == .OK ? panel.url?.path : nil
}

func chooseFiles(allowedContentTypes: [UTType]) -> [String] {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowedContentTypes = allowedContentTypes
    return panel.runModal() == .OK ? panel.urls.map(\.path) : []
}

/// For registering an H3 checkpoint directory (e.g. a MiniMax-H3 snapshot
/// containing FL2VA/Ref2VA) in ModelLibrary.
func chooseDirectory() -> String? {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.canCreateDirectories = false
    return panel.runModal() == .OK ? panel.url?.path : nil
}

/// Which H3ReferenceKind a picked reference file should be tagged as -
/// image, video, or audio, by extension (via the same UTType conformance
/// checks referenceMediaTypes filters the open panel with).
func isVideoFile(path: String) -> Bool {
    let ext = (path as NSString).pathExtension
    guard let type = UTType(filenameExtension: ext) else { return false }
    return type.conforms(to: .movie)
}

func isAudioFile(path: String) -> Bool {
    let ext = (path as NSString).pathExtension
    guard let type = UTType(filenameExtension: ext) else { return false }
    return type.conforms(to: .audio) && !type.conforms(to: .movie)
}

/// An audio file's length in seconds, nil if it can't be told. Read the way
/// the engine reads the file (AVAsset), not with AVAudioFile, which gives 0
/// frames for some files that play fine - Opus in .m4a, for one - and so
/// made every clip after the first look past the end of the audio.
func audioFileSeconds(_ path: String) -> Double? {
    final class Result: @unchecked Sendable { var seconds: Double? }
    let result = Result()
    let done = DispatchSemaphore(value: 0)
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    Task.detached {
        if let duration = try? await asset.load(.duration), duration.isNumeric, duration.seconds > 0 {
            result.seconds = duration.seconds
        }
        done.signal()
    }
    done.wait()
    return result.seconds
}
