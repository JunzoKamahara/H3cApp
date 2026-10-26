import AVFoundation
import Foundation

/// How the app compresses video. Everything kept in a project (generated,
/// joined, enlarged and imported videos) is HEVC at a constant quality:
/// H.264 at an average bitrate spent a fifth of it on dark scenes, which
/// then broke into 8-pixel blocks. 動画を書き出す… can turn a copy into
/// H.264 for players without HEVC; quality lost in a master could not be
/// won back there, the other way round.
enum VideoCodec {
    case hevc
    case h264

    /// AVAssetExportSession's preset for re-encoding a master (a join, an
    /// import, the label): HEVC, which keeps dark scenes nearly as they were
    /// (1.9 Mbps and 0.4 dB off for a dark 1920x1088 clip).
    static let masterExportPreset = AVAssetExportPresetHEVCHighestQuality

    /// AVAssetWriter's settings, as the engine writes its videos
    /// (h3_av_writer.m): HEVC at quality 0.85. For H.264 also quality 0.85:
    /// its export preset gave the dark clip 1.7 Mbps (47.1 dB), this 3.4
    /// (47.8 dB). A key frame every 30 frames either way.
    func writerSettings(width: Int, height: Int) -> [String: Any] {
        var compression: [String: Any] = [
            AVVideoQualityKey: 0.85,
            AVVideoMaxKeyFrameIntervalKey: 30,
            AVVideoAllowFrameReorderingKey: true,
        ]
        if self == .h264 { compression[AVVideoProfileLevelKey] = AVVideoProfileLevelH264HighAutoLevel }
        return [
            AVVideoCodecKey: self == .hevc ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
        ]
    }

    /// The codec of `track`'s video, if it is one of these.
    static func of(_ track: AVAssetTrack) async -> VideoCodec? {
        guard let format = try? await track.load(.formatDescriptions).first else { return nil }
        switch CMFormatDescriptionGetMediaSubType(format) {
        case kCMVideoCodecType_HEVC: return .hevc
        case kCMVideoCodecType_H264: return .h264
        default: return nil
        }
    }
}
