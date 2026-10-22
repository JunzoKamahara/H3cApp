import Foundation
import H3Engine

// カスタムクリップ: a つなげて生成 laid out clip by clip - each clip's
// length, what happens in it, its images and its lip-sync audio - instead of
// equal clips cut from one length. Each clip after the first continues the
// one before (its last 22 frames carried over), unless it starts from a
// first frame of its own: then it is joined to it with a cut.

/// One clip of カスタムクリップ, as the form and the project keep it.
struct ChainClipSpec: Codable, Equatable, Identifiable {
    /// The clip's images (clips after the first; the first uses the form's).
    enum Images: String, Codable, CaseIterable {
        /// The form's reference images (none if the form has none).
        case same
        /// Its own reference images, videos or audio.
        case references
        /// Its own first and/or last frame.
        case frames
    }

    /// The clip's lip-sync audio (clips after the first).
    enum Audio: String, Codable, CaseIterable {
        /// The audio of the clip before, from where that clip left it.
        case carryOn
        /// Another file, from audioOffset seconds.
        case file
        /// No lip sync: the clip makes its own sound.
        case none
    }

    var id = UUID()
    var seconds = 7
    var prompt = ""
    var images = Images.same
    var references: [ProjectDraft.Reference] = []
    var firstFrame: String? = nil
    var lastFrame: String? = nil
    var audio = Audio.carryOn
    var audioPath: String? = nil
    var audioOffset = 0.0

    static let secondsRange = 1 ... 15
    /// A continuation must be longer than the 22 frames it carries over.
    static let continuingSecondsRange = 2 ... 15

    var frames: Int { Int(h3AlignedFrameCount(seconds: Double(seconds))) }

    /// Starts from its own first frame instead of continuing the clip before.
    var startsFresh: Bool { images == .frames && firstFrame != nil }

    /// The same clip with every file path passed through `map`.
    func mappingPaths(_ map: (String) -> String) -> ChainClipSpec {
        var clip = self
        clip.references = references.map { .init(kind: $0.kind, path: map($0.path)) }
        clip.firstFrame = firstFrame.map(map)
        clip.lastFrame = lastFrame.map(map)
        clip.audioPath = audioPath.map(map)
        return clip
    }

    /// The files the clip uses.
    var paths: [String] {
        references.map(\.path) + [firstFrame, lastFrame, audioPath].compactMap { $0 }
    }

    var referenceInputs: [H3ReferenceInput] {
        references.compactMap { reference in
            H3ReferenceKind(storedName: reference.kind).map { H3ReferenceInput(kind: $0, path: reference.path) }
        }
    }
}

/// One clip of a つなげて生成 as it will be made (ChainRun).
struct ChainClipPlan {
    enum Images {
        /// What the form has: its references, and on the first clip its
        /// first and last frame too.
        case form
        /// None - the form's references switched off for this clip.
        case none
        case references([H3ReferenceInput])
        case frames(first: String?, last: String?)
    }

    var frames: Int
    /// What the clip adds to the prompt (chainClipPrompt).
    var text: String?
    var images = Images.form
    /// Carries over the last frames of the clip before.
    var continues: Bool
    /// The lip-sync audio and where in it the clip starts; nil without.
    var audioPath: String?
    var audioOffset = 0.0
}

/// Where each clip starts in the joined video, in frames: a continuation
/// overlaps the clip before by the frames it carries over, a cut doesn't.
func chainClipStarts(frames: [Int], continues: [Bool], overlap: Int = continuationFrames) -> [Int] {
    var starts: [Int] = []
    var start = 0
    for index in frames.indices {
        if index > 0 { start += frames[index - 1] - (continues[index] ? overlap : 0) }
        starts.append(start)
    }
    return starts
}

extension GenerationViewModel {
    /// カスタムクリップ is what the chained length is made of.
    var usesCustomClips: Bool { chained && chainCustom && !customClips.isEmpty }

    /// The clips the next つなげて生成 makes: カスタムクリップ, or equal
    /// clips cut from the length (with any クリップごとの指示 and
    /// reference switches). Lip sync carries on from clip to clip, a clip
    /// starting past the end of its audio going without it.
    func chainClipPlans() -> [ChainClipPlan] {
        var plans: [ChainClipPlan]
        var audio: [ChainClipSpec.Audio] = []
        var audioFiles: [(path: String?, offset: Double)] = []
        if usesCustomClips {
            plans = customClips.enumerated().map { index, clip in
                var plan = ChainClipPlan(frames: clip.frames, text: clip.prompt, continues: index > 0 && !clip.startsFresh)
                if index > 0 {
                    switch clip.images {
                    case .same: plan.images = .form
                    case .references:
                        let inputs = clip.referenceInputs
                        plan.images = inputs.isEmpty ? .none : .references(inputs)
                    case .frames:
                        plan.images = clip.firstFrame == nil && clip.lastFrame == nil
                            ? .none : .frames(first: clip.firstFrame, last: clip.lastFrame)
                    }
                }
                return plan
            }
            audio = customClips.map(\.audio)
            audioFiles = customClips.map { ($0.audioPath, max(0, $0.audioOffset)) }
        } else {
            plans = currentChainPlan.enumerated().map { index, frames in
                var plan = ChainClipPlan(frames: frames,
                                         text: index < clipPrompts.count ? clipPrompts[index] : nil,
                                         continues: index > 0)
                if !chainClipUsesReferences(clipReferences, index: index) { plan.images = .none }
                return plan
            }
        }
        var lengths: [String: Double?] = [:]
        func length(_ path: String) -> Double? {
            if let known = lengths[path] { return known }
            let seconds = audioFileSeconds(path)
            lengths[path] = seconds
            return seconds
        }
        var current: (path: String, offset: Double)? = forcedAudioPath.map { ($0, max(0, forcedAudioOffset)) }
        for index in plans.indices {
            if index > 0 {
                switch index < audio.count ? audio[index] : .carryOn {
                case .carryOn:
                    let advance = plans[index - 1].frames - (plans[index].continues ? continuationFrames : 0)
                    current = current.map { ($0.path, $0.offset + Double(advance) / 24) }
                case .file:
                    current = audioFiles[index].path.map { ($0, audioFiles[index].offset) }
                case .none:
                    current = nil
                }
            }
            if let current, length(current.path).map({ current.offset < $0 - 0.1 }) ?? true {
                plans[index].audioPath = current.path
                plans[index].audioOffset = current.offset
            }
        }
        return plans
    }

    /// Turns the length's equal clips (and their クリップごとの指示) into
    /// カスタムクリップ, so they can be edited one by one.
    func customClipsFromCurrentPlan() -> [ChainClipSpec] {
        let plan = isChained ? currentChainPlan : [Int(h3AlignedFrameCount(seconds: Double(seconds)))]
        return plan.enumerated().map { index, frames in
            var clip = ChainClipSpec()
            clip.seconds = max(index == 0 ? 1 : 2, Int((Double(frames) / 24).rounded()))
                .clamped(to: ChainClipSpec.secondsRange)
            clip.prompt = index < clipPrompts.count ? clipPrompts[index] : ""
            if !chainClipUsesReferences(clipReferences, index: index) {
                clip.images = .references
            }
            return clip
        }
    }

    /// Switches the length to カスタムクリップ (one clip if there were none).
    func useCustomClips() {
        if customClips.isEmpty { customClips = customClipsFromCurrentPlan() }
        chained = true
        chainCustom = true
    }

    /// What is missing from カスタムクリップ's clips, if anything.
    var customClipsValidationMessage: String? {
        guard usesCustomClips else { return nil }
        for (index, clip) in customClips.enumerated() where index > 0 {
            let number = index + 1
            if let missing = clip.paths.first(where: { !FileManager.default.fileExists(atPath: $0) }) {
                let name = URL(fileURLWithPath: missing).lastPathComponent
                return String(localized: "\(number)本目のファイルが見つかりません: \(name)")
            }
            if clip.images == .references, !clip.references.isEmpty,
               !clip.references.contains(where: { $0.kind == "image" || $0.kind == "video" }) {
                return String(localized: "\(number)本目：音声だけの参照はできません。画像か動画の参照も追加してください")
            }
            if clip.audio == .file && clip.audioPath == nil {
                return String(localized: "\(number)本目のリップシンクの音声を選んでください")
            }
            if !clip.startsFresh && clip.seconds < ChainClipSpec.continuingSecondsRange.lowerBound {
                return String(localized: "\(number)本目は、前の動画の続きなので2秒以上にしてください")
            }
        }
        return nil
    }

    /// What カスタムクリップ makes, in seconds.
    var customClipsTotalSeconds: Double {
        let frames = customClips.map(\.frames)
        let starts = chainClipStarts(frames: frames, continues: customClips.indices.map {
            $0 > 0 && !customClips[$0].startsFresh
        })
        guard let last = starts.last, let lastFrames = frames.last else { return 0 }
        return Double(last + lastFrames) / 24
    }
}
