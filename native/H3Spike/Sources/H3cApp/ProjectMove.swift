import Foundation

// 別のプロジェクトに移動: a project video, with everything that makes it
// more than a file - its record (.json), its latent (.h3lt, for continuing
// it and for Open-H3 再生成) and, for a つなげて生成, the clips it was
// joined from (chains/<name>) - moves into another project. Paths in the
// records are relative to the project they sit in, so each is rewritten:
// a moved file to where it went, an input file (references, frames,
// lip-sync audio) to a copy in the new project's references folder.

extension ProjectFiles {
    /// What one move takes along (worked out on the main actor, where the
    /// project's videos are known).
    struct VideoMove: Sendable {
        let video: URL
        let sourceProject: URL
        let destinationProject: URL
        /// The つなげて生成's own clip folder (chains/<name>), moved along.
        let chainFolder: URL?
        /// The latent the video continues from: its own, or for an enlarged
        /// copy made before latents were copied along, its original's
        /// (copied, as the original stays).
        let latent: URL?
        /// Enlarged copies in the source that use this video's latent and
        /// keep a copy of it when it leaves.
        let enlargedCopies: [URL]
    }

    enum MoveError: LocalizedError {
        case notInProject(String)

        var errorDescription: String? {
            switch self {
            case .notInProject(let name):
                return String(localized: "プロジェクトのフォルダの中にない動画は移動できません: \(name)")
            }
        }
    }

    /// Moves the video as `move` says and returns where it went. Either all
    /// of it moves or, on a failure, what had moved goes back. Copies of
    /// input files already made stay in the new project (they are copies).
    /// Reads and moves files: call it off the main actor.
    static func moveVideo(_ move: VideoMove) throws -> URL {
        let fm = FileManager.default
        let source = move.sourceProject, destination = move.destinationProject
        let video = move.video
        let stem = video.deletingPathExtension().lastPathComponent
        guard stored(video.path, in: source) != video.path else { throw MoveError.notInProject(video.lastPathComponent) }

        // Names free in the destination: the video's (with its record and
        // latent), and its clip folder's.
        var base = stem
        var counter = 2
        while ["mp4", "json", "h3lt"].contains(where: {
            fm.fileExists(atPath: destination.appendingPathComponent(base).appendingPathExtension($0).path)
        }) {
            base = "\(stem)-\(counter)"
            counter += 1
        }
        let movedVideo = destination.appendingPathComponent(base + ".mp4")
        var movedChain: URL?
        if let chain = move.chainFolder {
            let chains = destination.appendingPathComponent("chains", isDirectory: true)
            var name = chain.lastPathComponent
            counter = 2
            while fm.fileExists(atPath: chains.appendingPathComponent(name).path) {
                name = "\(chain.lastPathComponent)-\(counter)"
                counter += 1
            }
            movedChain = chains.appendingPathComponent(name, isDirectory: true)
        }

        // A path as a record in the source stores it, as the destination's
        // record should: the moved files where they went; input files from
        // inside the source copied into the destination; anything else
        // (outside the project, not there, not copied) still naming the
        // file it named: an absolute path as it was, one relative to the
        // source as an absolute path, since the destination's record would
        // read it as a file in the destination.
        func whereMoved(_ path: String) -> String? {
            let resolved = URL(fileURLWithPath: resolved(path, in: source)).standardizedFileURL
            if resolved.path == video.standardizedFileURL.path { return base + ".mp4" }
            if let chain = move.chainFolder?.standardizedFileURL, let movedChain,
               resolved.path.hasPrefix(chain.path + "/") {
                let rest = String(resolved.path.dropFirst(chain.path.count + 1))
                return stored(movedChain.appendingPathComponent(rest).path, in: destination)
            }
            return nil
        }
        var copied: [String: String] = [:]
        func carried(_ path: String) -> String {
            if let moved = whereMoved(path) { return moved }
            let resolvedPath = resolved(path, in: source)
            let left = path.hasPrefix("/") || path.isEmpty ? path
                : stored(URL(fileURLWithPath: resolvedPath).standardizedFileURL.path, in: destination)
            // Inside it with links resolved too: a record can be anyone's.
            guard stored(resolvedPath, in: source) != resolvedPath, isInside(resolvedPath, folder: source),
                  fm.fileExists(atPath: resolvedPath) else {
                return left
            }
            if let copy = copied[resolvedPath] { return copy }
            guard let copy = try? importReference(resolvedPath, into: destination, inside: source) else { return left }
            let storedCopy = stored(copy, in: destination)
            copied[resolvedPath] = storedCopy
            return storedCopy
        }
        func rewritten(_ record: ProjectVideoRecord) -> ProjectVideoRecord {
            var record = record
            var draft = record.draft
            draft.firstFrame = draft.firstFrame.map(carried)
            draft.lastFrame = draft.lastFrame.map(carried)
            draft.forcedAudio = draft.forcedAudio.map(carried)
            draft.references = draft.references.map { $0.mappingPath(carried) }
            draft = draft.mappingClipPaths(carried)
            // The video it continued moves only if it was one of the clips;
            // otherwise the name stays, as a note of where it came from.
            draft.continueFrom = draft.continueFrom.map { whereMoved($0) ?? $0 }
            record.draft = draft
            record.joinedClips = record.joinedClips?.map { whereMoved($0) ?? $0 }
            return record
        }

        // Every record is rewritten now, while the files are where the paths
        // say: path standardizing depends on what exists (a "/private"
        // prefix is dropped only for a file that is there).
        let record = ProjectFiles.record(for: video).map(rewritten)
        var clipRecords: [(name: String, original: Data, rewritten: Data)] = []
        if let chain = move.chainFolder {
            let names = (try? fm.contentsOfDirectory(atPath: chain.path)) ?? []
            for name in names.sorted() where name.lowercased().hasSuffix(".json") {
                guard let data = try? Data(contentsOf: chain.appendingPathComponent(name)),
                      let clip = try? decoder.decode(ProjectVideoRecord.self, from: data) else { continue }
                clipRecords.append((name, data, try encoder.encode(rewritten(clip))))
            }
        }
        let latent = latentURL(for: video)
        let movedLatent = latentURL(for: movedVideo)
        let sourceRecord = video.deletingPathExtension().appendingPathExtension("json")
        let movedRecord = movedVideo.deletingPathExtension().appendingPathExtension("json")

        // Copies that use this video's latent keep one of their own.
        if fm.fileExists(atPath: latent.path) {
            for copy in move.enlargedCopies where !fm.fileExists(atPath: latentURL(for: copy).path) {
                try? fm.copyItem(at: latent, to: latentURL(for: copy))
            }
        }

        // The moves, undone in reverse on a failure.
        var undo: [() -> Void] = []
        func moveItem(_ from: URL, _ to: URL) throws {
            try fm.moveItem(at: from, to: to)
            undo.append { try? fm.moveItem(at: to, to: from) }
        }
        do {
            try moveItem(video, movedVideo)
            if fm.fileExists(atPath: latent.path) {
                try moveItem(latent, movedLatent)
            } else if let other = move.latent, fm.fileExists(atPath: other.path) {
                try fm.copyItem(at: other, to: movedLatent)
                undo.append { try? fm.removeItem(at: movedLatent) }
            }
            if let chain = move.chainFolder, let movedChain {
                try fm.createDirectory(at: movedChain.deletingLastPathComponent(), withIntermediateDirectories: true)
                try moveItem(chain, movedChain)
                // The clips' own records, rewritten above; put back as they
                // were before the folder goes back on a failure.
                for clip in clipRecords {
                    let url = movedChain.appendingPathComponent(clip.name)
                    try clip.rewritten.write(to: url, options: .atomic)
                    undo.append { try? clip.original.write(to: url, options: .atomic) }
                }
            }
            if var record {
                record.video = movedVideo.lastPathComponent
                try encoder.encode(record).write(to: movedRecord, options: .atomic)
                undo.append { try? fm.removeItem(at: movedRecord) }
            } else if fm.fileExists(atPath: sourceRecord.path) {
                // A record this build can't read goes as it is.
                try moveItem(sourceRecord, movedRecord)
            }
        } catch {
            for step in undo.reversed() { step() }
            throw error
        }
        if record != nil { try? fm.removeItem(at: sourceRecord) }
        // Nothing left in the source's chains folder: it goes too.
        if let chains = move.chainFolder?.deletingLastPathComponent(),
           ((try? fm.contentsOfDirectory(atPath: chains.path)) ?? ["?"]).isEmpty {
            try? fm.removeItem(at: chains)
        }
        return movedVideo
    }
}
