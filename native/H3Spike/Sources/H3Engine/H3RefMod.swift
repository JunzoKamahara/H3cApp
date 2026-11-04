import CH3
import Foundation

/// What a RefMod file holds. Image and video members are the Ref2VA video
/// VAE's latent (F16 [1,24,T,H,W]); audio members the audio VAE's
/// ([1,32,2,T]). See h3_refmod.h.
public enum H3RefModKind: Sendable, Equatable {
    case image, video, audio

    var cValue: h3_refmod_kind {
        switch self {
        case .image: return H3_REFMOD_IMAGE
        case .video: return H3_REFMOD_VIDEO
        case .audio: return H3_REFMOD_AUDIO
        }
    }

    init?(cValue: h3_refmod_kind) {
        switch cValue {
        case H3_REFMOD_IMAGE: self = .image
        case H3_REFMOD_VIDEO: self = .video
        case H3_REFMOD_AUDIO: self = .audio
        default: return nil
        }
    }
}

public enum H3RefModMode: Sendable, Equatable {
    /// The encoder's latent as it is ("encode" upstream).
    case full
    /// Pooled to a small grid and refined ("training" upstream).
    case compressed

    var cValue: h3_refmod_mode {
        switch self {
        case .full: return H3_REFMOD_FULL
        case .compressed: return H3_REFMOD_COMPRESSED
        }
    }
}

public struct H3RefModMember: Sendable, Equatable, Identifiable {
    /// The member's index in its file (0 for a single-latent file).
    public let id: Int
    public let kind: H3RefModKind
    public let mode: H3RefModMode
    public let latentT: Int
    public let latentH: Int
    public let latentW: Int
    /// DiT rows this member adds: T x H/2 x W/2 for a visual member, 2T
    /// for audio.
    public let tokens: Int
    public let formatVersion: Int
    public let optimizeSteps: Int
    public let name: String
    public let source: String
    public let sourceShape: String
    public let pool: String
    public let description: String
    public let supported: Bool

    /// For a video or audio member: the seconds it spans (5 latent frames
    /// per 17 video frames at 24 fps; 40 latent frames per second of audio).
    public var seconds: Double? {
        switch kind {
        case .image: return nil
        case .video: return latentT <= 2 ? nil : Double((latentT - 2) / 5 * 17 + 5) / 24
        case .audio: return Double(latentT) / 40
        }
    }
}

public struct H3RefModInfo: Sendable, Equatable {
    public let formatVersion: Int
    public let isBundle: Bool
    public let name: String
    public let members: [H3RefModMember]

    /// Reads only the safetensors header - cheap enough to call from the UI.
    public static func inspect(path: String) throws -> H3RefModInfo {
        var info = h3_refmod_info()
        var errorBuffer = [CChar](repeating: 0, count: 512)
        let ok = errorBuffer.withUnsafeMutableBufferPointer { errorBuf in
            h3_refmod_inspect(path, &info, errorBuf.baseAddress, errorBuf.count)
        }
        guard ok != 0 else { throw H3EngineError.unusableRefMod(String(cString: errorBuffer)) }
        defer { h3_refmod_info_free(&info) }
        var members: [H3RefModMember] = []
        for index in 0..<Int(info.member_count) {
            let member = info.members[index]
            guard let kind = H3RefModKind(cValue: member.kind) else { continue }
            members.append(H3RefModMember(
                id: index, kind: kind,
                mode: member.mode == Int32(H3_REFMOD_COMPRESSED.rawValue) ? .compressed : .full,
                latentT: Int(member.latent_t), latentH: Int(member.latent_h), latentW: Int(member.latent_w),
                tokens: Int(member.tokens), formatVersion: Int(member.format_version),
                optimizeSteps: Int(member.optimize_steps),
                name: fixedCString(member.name), source: fixedCString(member.source),
                sourceShape: fixedCString(member.source_shape), pool: fixedCString(member.pool),
                description: fixedCString(member.description), supported: member.supported != 0))
        }
        return H3RefModInfo(formatVersion: Int(info.format_version), isBundle: info.bundle != 0,
                            name: fixedCString(info.name), members: members)
    }
}

/// What H3Engine.extractRefMod makes: a single-latent RefMod file from one
/// source, with the upstream extractor's defaults.
public struct H3RefModExtractRequest: Sendable, Equatable {
    public var sourcePath: String
    public var outputPath: String
    public var kind: H3RefModKind
    /// The RefMod's name in its metadata; empty = the source file's stem.
    public var name: String = ""
    public var mode: H3RefModMode = .full
    /// Short-edge cap in pixels (the source is only ever shrunk).
    public var resolutionCap: Int = 1024
    /// Video: how many frames are kept (spread evenly, snapped to 17k+5).
    public var latentFrames: Int = 16
    /// Compressed: cells on the latent's longer side.
    public var poolLongEdge: Int = 16
    /// Compressed: Adam steps refining the pooled latent (0 = none).
    public var optimizeSteps: Int = 500
    /// Audio: seconds kept from the start.
    public var maxSeconds: Int = 15
    public var description: String = ""

    public init(sourcePath: String, outputPath: String, kind: H3RefModKind) {
        self.sourcePath = sourcePath
        self.outputPath = outputPath
        self.kind = kind
    }
}
