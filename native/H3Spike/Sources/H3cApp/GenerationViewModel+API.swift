import AppKit
import Foundation
import H3Engine

/// The local automation API this app exposes while it's running - the
/// direct replacement for the old Python gui/server.py, which this app no
/// longer ships (removed along with the Python dependency it needed). The
/// difference in shape isn't incidental: gui/server.py had to accept
/// browser file uploads because its client was a browser; a script driving
/// this API runs on the same Mac, so it can just pass filesystem paths.
///
/// POST /api/generate fully replaces the current draft from its JSON body
/// (same fields the form itself edits) and, if valid, starts a job exactly
/// as pressing the window's generate button would - there is only ever one job at a time,
/// shared with the UI, so a request while one is already running gets 409.
/// GET /api/status polls progress. GET /api/result/video streams the
/// current result until the next generate() call deletes it.
let h3APIPort: UInt16 = 8420

/// The app's one API server, started at launch and kept until quit. It
/// always talks to the primary model (AppModels), which outlives its window,
/// so the API keeps working with every window closed while the app runs.
@MainActor
final class APIHost {
    static let shared = APIHost()

    private var server: HTTPServer?
    private var startError: String?

    func start() {
        guard server == nil, startError == nil else { return }
        let server = HTTPServer(port: h3APIPort) { request in
            await AppModels.shared.primary.handleAPIRequest(request)
        }
        do {
            try server.start()
            self.server = server
        } catch {
            startError = String(describing: error)
        }
    }

    func stop() {
        server?.stop()
        server = nil
    }

    /// What 「APIサーバー」 says in `model`'s window.
    func status(for model: GenerationViewModel) -> String {
        if let startError { return String(localized: "起動できませんでした（\(startError)）") }
        guard server != nil else { return String(localized: "起動しています…") }
        let address = "http://127.0.0.1:\(h3APIPort)"
        return model === AppModels.shared.primary
            ? address : String(localized: "\(address)（別のウィンドウで受け付け中）")
    }
}

extension GenerationViewModel {
    func handleAPIRequest(_ request: HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/api/status"):
            return statusResponse()
        case ("POST", "/api/generate"):
            return handleGenerateRequest(request)
        case ("POST", "/api/cancel"):
            cancel()
            return .json(200, ["ok": true])
        case ("GET", "/api/result/video"):
            return resultVideoResponse()
        case ("GET", "/api/preview"):
            // 途中の様子 as PNG: the latest picture of the running generation.
            guard let image = denoisePreviewImage,
                  let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
                return .error(404, "no denoise preview yet (turn it on with POST /api/preview/settings)")
            }
            return HTTPResponse(status: 200, headers: ["Content-Type": "image/png"], body: png)
        case ("POST", "/api/preview/settings"):
            // {"enabled": bool} - Advanced Settings' 途中の様子; turning it on
            // downloads the decoder (22.7 MB) when it is not there yet.
            let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
            guard let enabled = json["enabled"] as? Bool else { return .error(400, "\"enabled\" must be true or false") }
            showDenoisePreview = enabled
            if enabled && !DenoisePreviewWeights.isPresent { denoisePreviewDownloader.download() }
            return .json(200, denoisePreviewStatus)
        case ("GET", "/api/models"):
            return .json(200, library.models.map {
                ["id": $0.id.uuidString, "name": $0.name, "path": $0.path, "active": $0.id == library.activeModelID]
            })
        case ("POST", "/api/models/import"):
            return await handleModelImport(request)
        case ("GET", "/api/models/detail"):
            // Model Management's 詳細 as JSON (?id=<model id>).
            guard let id = request.query["id"].flatMap(UUID.init(uuidString:)),
                  let entry = library.models.first(where: { $0.id == id }) else {
                return .error(400, "\"id\" must be a model id from GET /api/models")
            }
            let path = entry.path
            let detail = await Task.detached { ModelDetail.inspect(path: path, id: id) }.value
            return .json(200, ["name": entry.name, "path": path, "total_bytes": detail.totalBytes,
                               "sections": detail.sections.map { section in
                                   ["title": section.title, "note": section.note ?? NSNull(),
                                    "rows": section.rows.map { row in
                                        ["title": row.title, "status": "\(row.status)", "lines": row.lines] as [String: Any]
                                    }] as [String: Any]
                               }])
        case ("GET", "/api/models/import/downloads"):
            // What the import wizard would offer to download for a part
            // (?kind=dit|ref2va|textEncoder|vision|shared); lists only.
            guard let kind = request.query["kind"].flatMap(ImportDownloadKind.init(rawValue:)) else {
                return .error(400, "\"kind\" must be one of dit, ref2va, textEncoder, vision, shared")
            }
            do {
                let repo = request.query["repo"] ?? kind.savedRepo
                let choices = try await ImportDownloadChoice.list(kind, repo: repo)
                return .json(200, ["repo": repo, "destination": kind.destination,
                                   "choices": choices.map {
                                       ["title": $0.title, "files": $0.entries.count, "bytes": $0.bytes,
                                        "verified": $0.verified,
                                        "unsupported": $0.unsupported ?? NSNull()] as [String: Any]
                                   }])
            } catch {
                return .error(502, error.localizedDescription)
            }
        case ("POST", "/api/models/remove"):
            // Unregisters a model and deletes its generation caches.
            let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
            guard !isGenerating else { return .error(409, "a generation is running") }
            guard let text = json["id"] as? String, let id = UUID(uuidString: text),
                  let entry = library.models.first(where: { $0.id == id }) else {
                return .error(400, "\"id\" must be a model id from GET /api/models")
            }
            // "delete_files": true also moves the model's files to the Trash,
            // like Model Management's 削除… ("dry_run": true only reports
            // what would go and the space it frees).
            // The model's generation caches are always deleted.
            let others = library.models.filter { $0.id != id }.map { (path: $0.path, id: $0.id) }
            let path = entry.path, files = json["delete_files"] as? Bool == true
            let plan = await Task.detached {
                ModelRemovalPlan.make(modelPath: path, id: id, others: others, includeFiles: files)
            }.value
            let report: [String: Any] = [
                "targets": plan.targets.map { ["path": $0.path, "label": $0.label, "bytes": $0.bytes] },
                "caches": plan.caches.map { ["path": $0.path, "label": $0.label, "bytes": $0.bytes] },
                "freed_bytes": plan.freedBytes, "cache_bytes": plan.cacheBytes,
                "shared_bytes": plan.sharedBytes, "shared_files": plan.sharedFiles, "refused": plan.refused]
            if json["dry_run"] as? Bool == true { return .json(200, report) }
            guard id != library.activeModelID else { return .error(409, "that model is in use") }
            library.removeModel(id: id)
            let failed = await Task.detached { plan.execute() }.value
            return .json(200, report.merging(["ok": failed.isEmpty, "failed": failed]) { $1 })
        case ("POST", "/api/models/add_ref2va"):
            return await handleAddRef2VA(request)
        case ("POST", "/api/models/add"):
            // Registers a model folder, like Model Management's フォルダを追加….
            let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
            guard let path = json["path"] as? String, FileManager.default.fileExists(atPath: path + "/FL2VA") else {
                return .error(400, "\"path\" must be a folder holding FL2VA")
            }
            library.addModel(path: path, name: json["name"] as? String)
            return .json(200, ["id": library.models.first { $0.path == path }?.id.uuidString ?? ""])
        case ("POST", "/api/models/select"):
            let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
            guard !isGenerating else { return .error(409, "a generation is running") }
            guard let text = json["id"] as? String, let id = UUID(uuidString: text),
                  library.models.contains(where: { $0.id == id }) else {
                return .error(400, "\"id\" must be a model id from GET /api/models")
            }
            switchModel(to: id)
            return .json(200, ["ok": true])
        case ("GET", "/api/project"):
            return projectResponse()
        case ("POST", "/api/project/new"), ("POST", "/api/project/open"), ("POST", "/api/project/close"),
             ("POST", "/api/project/restore"), ("POST", "/api/project/use-as-reference"),
             ("POST", "/api/project/delete-video"):
            return handleProjectRequest(request)
        case ("POST", "/api/project/join"):
            return await handleJoinRequest(request)
        case ("GET", "/api/loras"):
            return .json(200, library.loras.map {
                ["id": $0.id.uuidString, "name": $0.name, "path": $0.path,
                 "strength": $0.strength, "recommended_steps": $0.recommendedSteps.map { $0 as Any } ?? NSNull(),
                 "enabled": $0.enabled, "recommended_sampling": recommendedSampling($0),
                 "sampler": $0.sampler.rawValue, "audio_shift": $0.audioShift.map { $0 as Any } ?? NSNull()]
            })
        default:
            return .error(404, "no such endpoint: \(request.method) \(request.path)")
        }
    }

    /// POST /api/models/import: {"files": [GGUF paths]?, "folder"? (every H3
    /// GGUF in it is added), "ref2va": [paths that are the Ref2VA DiT]?
    /// (default: DiT files named ref2va), "name"?, "base_model"? (a
    /// registered model's id), "base_path"? (a folder, registered or not)} -
    /// the same import as Model Management's
    /// wizard: files are sorted by what they hold (of several for one part,
    /// the first is used and "notes" says so), a text encoder without a
    /// vision tower gets the base model's. "dry_run": true returns the plan
    /// (parts, duplicates, problems) without importing. Returns the plan's problems as a
    /// 400, else registers the new model (not selected) and returns its id.
    private func handleModelImport(_ request: HTTPRequest) async -> HTTPResponse {
        let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
        guard !isGenerating else { return .error(409, "a generation is running") }
        var picks = (json["files"] as? [String] ?? []).map(ImportPick.init(path:))
        var skipped: [String] = []
        if let folder = json["folder"] as? String {
            let found = importableGGUFs(in: folder)
            picks += found.picks.filter { pick in !picks.contains { $0.path == pick.path } }
            skipped = found.skipped
        }
        guard !picks.isEmpty else {
            return .error(400, "\"files\" (GGUF paths) or \"folder\" (holding H3 GGUFs) is required")
        }
        var bases = preferredImportBases(library.models) + (downloadedImportBase().map { [$0] } ?? [])
        if let folder = json["base_path"] as? String {
            guard let entry = unregisteredImportBase(path: folder, registered: library.models) ??
                    library.models.first(where: { $0.path == folder }) else {
                return .error(400, "\"base_path\" has no FL2VA tokenizer, VAEs and configs")
            }
            bases.insert(entry, at: 0)
        }
        let base = (json["base_model"] as? String).flatMap(UUID.init(uuidString:))
            .map { id in bases.first { $0.id == id } } ?? bases.first
        let ref2va = json["ref2va"] as? [String] ??
            picks.filter { $0.info?.role == .dit && $0.looksLikeRef2VA }.map(\.path)
        let plan = ImportPlan(files: picks, ref2vaPaths: Set(ref2va), base: base,
                              availableBytes: availableBytes(at: importedModelsDirectory))
        if json["dry_run"] as? Bool == true {
            // What the wizard's review page would show, nothing written.
            return .json(200, [
                "can_import": plan.canImport,
                "files": picks.map(\.path), "skipped": skipped,
                "base": base.map { ["name": $0.name, "path": $0.path] } ?? NSNull(),
                "items": plan.items.map { item in
                    ["part": item.title, "detail": item.detail,
                     "candidates": item.candidates.map(\.fileName),
                     "hint": item.hint ?? NSNull(),
                     "hint_url": item.hintURL?.absoluteString ?? NSNull(),
                     "download": item.download?.rawValue ?? NSNull()] as [String: Any]
                },
                "problems": plan.problems, "notes": plan.notes,
                "unused": plan.unused.map { ["file": ($0.key as NSString).lastPathComponent, "reason": $0.value] },
                "bytes_written": plan.bytesWritten])
        }
        guard let dit = plan.dit, let base, plan.canImport else {
            let missing = plan.items.filter { $0.status == .missing }.map(\.title)
            return .error(400, (plan.problems + missing.map { "missing: \($0)" }).joined(separator: "; "))
        }
        let importer = ModelImporter()
        do {
            let path = try await importer.importNow(.init(
                name: json["name"] as? String ?? "MiniMax-H3 GGUF", dit: dit,
                ref2vaDit: plan.ref2vaDit, textEncoder: plan.textEncoder,
                vision: plan.vision, base: base))
            library.addModel(path: path)
            let entry = library.models.first { $0.path == path }
            return .json(200, ["id": entry?.id.uuidString ?? "", "path": path,
                               "items": plan.items.map { ["part": $0.title, "detail": $0.detail] },
                               "notes": plan.notes])
        } catch {
            return .error(500, error.localizedDescription)
        }
    }

    /// POST /api/models/add_ref2va: {"id", "file"? (a Ref2VA DiT GGUF),
    /// "dry_run"?} - Model Manager's Ref2VAを追加… for a model without
    /// Ref2VA. A GGUF model takes "file" and clones everything else from its
    /// FL2VA. An official model's Ref2VA (about 66 GB) is downloaded from
    /// Model Manager only; "dry_run" reports what either would download
    /// and clone, nothing written.
    private func handleAddRef2VA(_ request: HTTPRequest) async -> HTTPResponse {
        let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
        guard let text = json["id"] as? String, let id = UUID(uuidString: text),
              let entry = library.models.first(where: { $0.id == id }) else {
            return .error(400, "\"id\" must be a model id from GET /api/models")
        }
        guard !modelHasRef2VA(entry) else { return .error(409, "that model already has Ref2VA") }
        let gguf = isGGUFModel(entry)
        let plan: Ref2VAAddPlan
        do {
            plan = gguf ? try await .gguf(modelPath: entry.path) : try await .official(modelPath: entry.path)
        } catch {
            return .error(502, error.localizedDescription)
        }
        var clones = plan.clones
        var part: ModelImportManifest.Part?
        if let file = json["file"] as? String {
            guard gguf else { return .error(400, "\"file\" is for a model made from GGUF files") }
            guard let local = Ref2VAAddPlan.localDiT(ImportPick(path: file), modelPath: entry.path) else {
                return .error(400, "\"file\" must be a DiT GGUF")
            }
            clones.append(local.clone)
            part = local.part
        }
        let report: [String: Any] = [
            "gguf_model": gguf,
            "downloads": plan.downloads.map { ["path": $0.path, "bytes": $0.size] },
            "download_bytes": plan.downloadBytes,
            "clones": clones.map { ["from": $0.from, "to": $0.to] }, "clone_bytes": plan.cloneBytes]
        if json["dry_run"] as? Bool == true { return .json(200, report) }
        guard gguf else {
            return .error(400, "an official model's Ref2VA is downloaded from Model Manager (Ref2VAを追加…)")
        }
        guard part != nil else { return .error(400, "\"file\" (a Ref2VA DiT GGUF) is required") }
        let path = entry.path
        do {
            try FileManager.default.createDirectory(atPath: path + "/" + ref2vaStagingFolderName,
                                                    withIntermediateDirectories: true)
            try await Ref2VAAddPlan.downloadSmall(plan.downloads, modelPath: path)
            let manifestPart = part
            try await Task.detached(priority: .userInitiated) {
                try Ref2VAAddPlan.finish(modelPath: path, clones: clones, manifestPart: manifestPart)
            }.value
            return .json(200, report.merging(["ok": true, "path": path + "/Ref2VA"]) { $1 })
        } catch {
            return .error(500, error.localizedDescription)
        }
    }

    /// What the adapter's metadata asks for (see samplingSettings).
    private func recommendedSampling(_ entry: LoRAEntry) -> Any {
        guard case .success(let info)? = library.loraInfo(for: entry) else { return NSNull() }
        return ["sampler": info.recommendsRenoise ? "renoise" : "euler",
                "audio_shift": info.recommendedAudioShift.map { $0 as Any } ?? NSNull()]
    }

    private func statusResponse() -> HTTPResponse {
        let engineStateText: String
        switch engineState {
        case .loading: engineStateText = "loading"
        case .ready: engineStateText = "ready"
        case .failed(let message): engineStateText = "failed: \(message)"
        }
        var body: [String: Any] = [
            "engine_state": engineStateText,
            "device": deviceLine,
            "is_generating": isGenerating,
            "is_cancelling": isCancelling,
            "phase": phase,
            "elapsed_seconds": elapsedSeconds,
            "stage_title": stageTitle,
            "stage_detail": stageDetail,
            "result_available": resultURL != nil,
        ]
        if let estimatedRemainingSeconds { body["estimated_remaining_seconds"] = estimatedRemainingSeconds }
        if let progressBarFraction { body["progress_fraction"] = progressBarFraction }
        if let errorMessage { body["error_message"] = errorMessage }
        if let project { body["project"] = project.name }
        body["denoise_preview"] = denoisePreviewStatus
        if let batchProgress {
            var batch: [String: Any] = ["index": batchProgress.index, "total": batchProgress.total,
                                        "chained": batchProgress.chained]
            if let batchTiming {
                batch["elapsed_seconds"] = batchTiming.elapsed
                if let remaining = batchTiming.remaining { batch["estimated_remaining_seconds"] = remaining }
            }
            body["batch"] = batch
        }
        return .json(200, body)
    }

    // MARK: Projects (see Projects.swift)

    private func projectResponse() -> HTTPResponse {
        guard let project else { return .json(200, ["open": false]) }
        let formatter = ISO8601DateFormatter()
        return .json(200, [
            "open": true,
            "name": project.name,
            "path": project.url.path,
            "batch_count": batchCount,
            "videos": projectVideos.map { video -> [String: Any] in
                var item: [String: Any] = ["file": video.url.lastPathComponent, "path": video.url.path]
                if let record = video.record {
                    item["seed"] = record.seed
                    item["completed_at"] = formatter.string(from: record.completedAt)
                    item["generation_seconds"] = record.generationSeconds
                    item["continued_from"] = record.draft.continueFrom
                        .map { URL(fileURLWithPath: $0).lastPathComponent as Any } ?? NSNull()
                    if let clips = record.joinedClips {
                        item["joined_clips"] = clips
                        item["planned_clips"] = record.chainPlannedClips as Any? ?? NSNull()
                    }
                }
                item["can_continue"] = canContinue(video)
                return item
            },
        ])
    }

    /// POST /api/project/join {"video": name}: joins the continuation chain
    /// ending at that video into joined/<name>_joined.mp4 and waits for it.
    private func handleJoinRequest(_ request: HTTPRequest) async -> HTTPResponse {
        let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
        guard project != nil else { return .error(400, "no project is open") }
        guard let name = json["video"] as? String,
              let video = projectVideos.first(where: { $0.url.lastPathComponent == name }) else {
            return .error(400, "\"video\" must name a file listed by GET /api/project")
        }
        guard !isExportingJoined else { return .error(409, "a joined video is being exported") }
        switch await exportJoinedVideo(endingAt: video, reveal: false) {
        case .success(let url):
            let chain = continuationChain(endingAt: video) ?? []
            return .json(200, ["path": url.path, "clips": chain.map { $0.url.lastPathComponent }])
        case .failure(let error):
            return .error(400, error.localizedDescription)
        }
    }

    private func handleProjectRequest(_ request: HTTPRequest) -> HTTPResponse {
        let json = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]) ?? [:]
        guard !isGenerating else { return .error(409, "a generation is running") }
        projectMessage = nil
        func video() -> ProjectVideo? {
            guard let name = json["video"] as? String else { return nil }
            return projectVideos.first { $0.url.lastPathComponent == name }
        }
        switch request.path {
        case "/api/project/new":
            guard let name = json["name"] as? String else { return .error(400, "\"name\" is required") }
            let parent = (json["directory"] as? String).map { URL(fileURLWithPath: $0, isDirectory: true) }
                ?? ProjectStore.defaultParentDirectory
            createProject(name: name, parent: parent)
        case "/api/project/open":
            guard let path = json["path"] as? String else { return .error(400, "\"path\" is required") }
            openProject(at: URL(fileURLWithPath: path, isDirectory: true))
        case "/api/project/close":
            closeProject()
            return .json(200, ["ok": true])
        default:
            guard project != nil else { return .error(400, "no project is open") }
            guard let video = video() else { return .error(400, "\"video\" must name a file listed by GET /api/project") }
            switch request.path {
            case "/api/project/restore": restoreProjectVideo(video)
            case "/api/project/use-as-reference": useVideoAsReference(video.url)
            default: deleteProjectVideo(video)
            }
        }
        if let projectMessage, project == nil || request.path != "/api/project/restore" {
            return .error(400, projectMessage)
        }
        return projectResponse()
    }

    private var denoisePreviewStatus: [String: Any] {
        var status: [String: Any] = ["enabled": showDenoisePreview,
                                     "decoder_ready": DenoisePreviewWeights.isPresent,
                                     "available": denoisePreviewImage != nil]
        switch denoisePreviewDownloader.state {
        case .downloading: status["downloading"] = true
        case .failed(let message): status["download_error"] = message
        case .idle: break
        }
        if denoisePreviewImage != nil {
            status["step"] = denoisePreviewStep
            status["steps"] = denoisePreviewSteps
        }
        return status
    }

    private func resultVideoResponse() -> HTTPResponse {
        guard let resultURL else { return .error(404, "no result available yet") }
        return .file(atPath: resultURL.path, contentType: "video/mp4")
    }

    private func handleGenerateRequest(_ request: HTTPRequest) -> HTTPResponse {
        guard let json = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] else {
            return .error(400, "expected a JSON object body")
        }
        // "count": videos made one after another with different seeds
        // (needs an open project, where they are kept).
        var count = 1
        if let value = json["count"] {
            guard let value = jsonInteger(value), batchCountRange.contains(value) else {
                return .error(400, "\"count\" must be an integer from \(batchCountRange.lowerBound) to \(batchCountRange.upperBound)")
            }
            guard value == 1 || project != nil else {
                return .error(400, "\"count\" above 1 needs an open project (POST /api/project/new or /api/project/open)")
            }
            count = value
        }
        // "from_form": true generates from the form as it is (e.g. a
        // project's saved form) instead of replacing it from this body.
        if let value = json["from_form"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                return .error(400, "\"from_form\" must be true or false")
            }
            if number.boolValue {
                let extra = Set(json.keys).subtracting(["from_form", "count"])
                guard extra.isEmpty else {
                    return .error(400, "\"from_form\" can't be combined with \(extra.sorted().joined(separator: ", "))")
                }
                guard canGenerate else {
                    return .error(409, validationMessage ?? "a generation is already running")
                }
                guard !isChained || project != nil else {
                    return .error(400, "a chained length needs an open project (POST /api/project/new or /api/project/open)")
                }
                generate(count: count)
                return .json(202, ["ok": true])
            }
        }
        guard let requestedPrompt = (json["prompt"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !requestedPrompt.isEmpty else {
            return .error(400, "\"prompt\" is required")
        }
        // Checked before anything below touches the draft/LoRA state. The
        // engine params clamp to stepsRange, so an out-of-range value would
        // otherwise be accepted and silently run a different step count.
        var requestedSteps: Int?
        if let stepsValue = json["steps"] {
            guard let value = jsonInteger(stepsValue), stepsRange.contains(value) else {
                return .error(400, "\"steps\" must be an integer from \(stepsRange.lowerBound) to \(stepsRange.upperBound)")
            }
            requestedSteps = value
        }

        // Every field below is set from this request (defaulting when
        // omitted), not merged onto whatever the UI's form last held - a
        // script driving this endpoint shouldn't have to know or care what
        // a human last typed into the window.
        prompt = requestedPrompt

        if let profileRaw = json["size_profile"] as? String {
            guard let profile = SizeProfile(rawValue: profileRaw) else {
                return .error(400, "unknown size_profile \(profileRaw) - expected one of: "
                    + SizeProfile.allCases.map(\.rawValue).joined(separator: ", "))
            }
            sizeProfile = profile
        } else {
            sizeProfile = .square
        }

        // 1-15 s in one generation; 16-90 s (or "chained": true from 15 s)
        // by つなげて生成, joining clips of "chain_clip_seconds" (5 or 7).
        var wantsChain = false
        if let chainedValue = json["chained"] {
            guard let flag = chainedValue as? Bool else { return .error(400, "\"chained\" must be true or false") }
            wantsChain = flag
        }
        if let secondsValue = json["seconds"] {
            guard let value = jsonInteger(secondsValue), (1 ... chainedSecondsRange.upperBound).contains(value) else {
                return .error(400, "\"seconds\" must be an integer from 1 to \(chainedSecondsRange.upperBound)")
            }
            if wantsChain && value < chainedSecondsRange.lowerBound {
                return .error(400, "\"chained\" needs \"seconds\" of \(chainedSecondsRange.lowerBound) or more")
            }
            setDuration(seconds: value, chained: wantsChain)
        } else {
            setDuration(seconds: 5, chained: false)
        }
        if let clipValue = json["chain_clip_seconds"] {
            switch jsonInteger(clipValue) {
            case 5: chainClipLength = .five
            case 7: chainClipLength = .seven
            default: return .error(400, "\"chain_clip_seconds\" must be 5 or 7")
            }
        } else {
            chainClipLength = defaultChainClipLength
        }

        if let modeRaw = json["compute_mode"] as? String {
            guard let mode = ComputeMode(rawValue: modeRaw) else {
                return .error(400, "unknown compute_mode \(modeRaw) - expected attentionCache, resident, or ssdStreaming")
            }
            computeMode = mode
        } else {
            computeMode = defaultComputeMode
        }

        if let speedRaw = json["speed_mode"] as? String {
            guard let speed = SpeedMode(rawValue: speedRaw) else {
                return .error(400, "unknown speed_mode \(speedRaw) - expected quality, fast, or fastest")
            }
            speedMode = speed
        } else {
            speedMode = .quality
        }
        // Stored as given (default when omitted); a speed preset still runs
        // at reuse 1 - see effectiveDenoiseReuse. The engine rejects values
        // outside reuseRange and the params builder would clamp them
        // silently, so reject them here.
        if let reuse = json["reuse"] {
            guard let reuse = jsonInteger(reuse), reuseRange.contains(reuse) else {
                return .error(400, "\"reuse\" must be an integer from \(reuseRange.lowerBound) to \(reuseRange.upperBound)")
            }
            denoiseReuse = reuse
        } else {
            denoiseReuse = defaultReuse
        }

        if let layers = json["dit_layers"] {
            guard let layers = jsonInteger(layers), ditLayersRange.contains(layers) else {
                return .error(400, "\"dit_layers\" must be an integer in \(ditLayersRange.lowerBound)-\(ditLayersRange.upperBound)")
            }
            ditLayers = layers
        } else {
            ditLayers = defaultDitLayers
        }

        if let fast = json["fast_attention"] {
            guard let fast = fast as? Bool else {
                return .error(400, "\"fast_attention\" must be true or false")
            }
            if fast && !fastAttentionAvailable {
                return .error(400, "fast_attention is not available on this build/device")
            }
            fastAttention = fast
        } else {
            fastAttention = false
        }

        // A JSON number, or a decimal string for clients whose JSON numbers
        // are doubles and would round a large seed.
        if let seedValue = json["seed"] {
            guard let seed = jsonUInt64(seedValue) else {
                return .error(400, "\"seed\" must be an integer from 0 to \(UInt64.max), as a number or a decimal string")
            }
            seedFixed = true
            seedText = String(seed)
        } else {
            seedFixed = false
        }

        let firstFrame = json["first_frame_path"] as? String
        let lastFrame = json["last_frame_path"] as? String
        let referencePaths = (json["reference_paths"] as? [String]) ?? []

        if firstFrame != nil || lastFrame != nil {
            if !referencePaths.isEmpty {
                return .error(400, "first_frame_path/last_frame_path cannot be combined with reference_paths")
            }
            creationMethod = .image
            imageInputMode = .firstLastFrame
            firstFramePath = firstFrame
            lastFramePath = lastFrame
        } else if !referencePaths.isEmpty {
            creationMethod = .image
            imageInputMode = .referenceImage
            referenceImages = referencePaths.map { path in
                let kind: H3ReferenceKind = isVideoFile(path: path) ? .video : (isAudioFile(path: path) ? .audio : .image)
                return H3ReferenceInput(kind: kind, path: path)
            }
        } else {
            creationMethod = .text
        }

        // "continue_from": a video of the open project to continue (its last
        // 22 frames carried over); its size must match "size_profile".
        // Omitted means a fresh clip. Like この動画の続きを作る, but the
        // request's own prompt and settings are used as given.
        if let source = json["continue_from"] {
            guard let name = source as? String,
                  let video = projectVideos.first(where: { $0.url.lastPathComponent == name }) else {
                return .error(400, "\"continue_from\" must name a video listed by GET /api/project")
            }
            guard canContinue(video), let record = video.record else {
                return .error(400, "\(name) has no saved latents to continue from")
            }
            if creationMethod == .image {
                return .error(400, "continue_from cannot be combined with first/last frames or references")
            }
            continuation = ContinuationSource(videoURL: video.url, latentURL: ProjectFiles.latentURL(for: video.url),
                                              sizeProfileRaw: record.draft.sizeProfile)
        } else {
            continuation = nil
        }

        // Lip sync: "forced_audio_path" holds that file's audio as the
        // soundtrack, from "forced_audio_offset" seconds on (default 0).
        // Works with every creation method; omitted means off.
        if let forced = json["forced_audio_path"] {
            guard let path = forced as? String, !path.isEmpty else {
                return .error(400, "\"forced_audio_path\" must be a file path")
            }
            guard FileManager.default.fileExists(atPath: path) else {
                return .error(400, "forced_audio_path not found: \(path)")
            }
            var offset = 0.0
            if let value = json["forced_audio_offset"] {
                guard let number = value as? NSNumber, number.doubleValue.isFinite,
                      number.doubleValue >= 0 else {
                    return .error(400, "\"forced_audio_offset\" must be a number of seconds >= 0")
                }
                offset = number.doubleValue
            }
            forcedAudioPath = path
            forcedAudioOffset = offset
        } else {
            if json["forced_audio_offset"] != nil {
                return .error(400, "\"forced_audio_offset\" needs \"forced_audio_path\"")
            }
            forcedAudioPath = nil
            forcedAudioOffset = 0
        }

        // "loras": [{"name": ..., "strength": 0.8}, ...] (or bare names)
        // replaces the stack; the older single "lora_name"/"lora_scale"
        // pair still works. Neither means no LoRA. Strength multiplies the
        // adapter's own alpha/rank scale; omitted keeps the entry's own.
        // "sampler" ("auto", "euler", "renoise") and "audio_shift" (0 =
        // auto) set the entry's own step rule and audio shift, like モデル管理.
        var requested: [(name: String, strength: Any?, sampler: Any?, audioShift: Any?)] = []
        if let list = json["loras"] as? [Any] {
            for item in list {
                if let name = item as? String {
                    requested.append((name, nil, nil, nil))
                } else if let object = item as? [String: Any], let name = object["name"] as? String {
                    requested.append((name, object["strength"], object["sampler"], object["audio_shift"]))
                } else {
                    return .error(400, "each \"loras\" item must be a name or {\"name\", \"strength\"}")
                }
            }
        } else if let loraName = json["lora_name"] as? String, !loraName.isEmpty {
            requested.append((loraName, json["lora_scale"], nil, nil))
        }
        if requested.count > maxStackedLoRAs {
            return .error(400, "at most \(maxStackedLoRAs) LoRAs can be stacked")
        }
        var stack: [(entry: LoRAEntry, strength: Any?, sampler: LoRASampler?, audioShift: Double?)] = []
        for (name, strength, sampler, audioShift) in requested {
            guard let entry = library.loras.first(where: { $0.name == name }) else {
                return .error(400, "unknown LoRA \(name) - see GET /api/loras")
            }
            if let strength, !(strength is NSNumber) {
                return .error(400, "strength for \(name) must be a number")
            }
            var samplerChoice: LoRASampler?
            if let sampler {
                guard let text = sampler as? String, let choice = LoRASampler(rawValue: text) else {
                    return .error(400, "sampler for \(name) must be \"auto\", \"euler\" or \"renoise\"")
                }
                samplerChoice = choice
            }
            var shift: Double?
            if let audioShift {
                guard let number = audioShift as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                      (0 ... 100).contains(number.doubleValue) else {
                    return .error(400, "audio_shift for \(name) must be a number from 0 (auto) to 100")
                }
                shift = number.doubleValue
            }
            stack.append((entry, strength, samplerChoice, shift))
        }
        for (entry, strength, sampler, audioShift) in stack {
            if let strength { library.setLoRAScale(id: entry.id, scaleText: "\(strength)") }
            if let sampler { library.setLoRASampler(id: entry.id, sampler) }
            if let audioShift {
                library.setLoRAAudioShift(id: entry.id, text: audioShift > 0 ? audioShift.formatted(.number.grouping(.never)) : "")
            }
        }
        library.setEnabledLoRAs(Set(stack.map { $0.entry.id }))

        // After the LoRAs are settled: enabling a Turbo LoRA moves the draft
        // to its recommended steps (followTurboLoRASteps), which an explicit
        // "steps" in the request still overrides.
        let turboSteps = library.enabledLoRAs.first { $0.recommendedSteps != nil }?.recommendedSteps
        steps = requestedSteps ?? turboSteps ?? defaultStepsForAPI

        // "sampler" ("auto", "euler", "renoise") and "audio_shift" (0 =
        // auto) override, for this one generation, the step rule and audio
        // shift the enabled LoRAs' metadata recommends - for A/B tests.
        var override = SamplingOverride()
        if let sampler = json["sampler"] {
            switch sampler as? String {
            case "auto"?: break
            case "euler"?: override.renoise = false
            case "renoise"?: override.renoise = true
            default: return .error(400, "\"sampler\" must be \"auto\", \"euler\" or \"renoise\"")
            }
        }
        if let shift = json["audio_shift"] {
            guard let number = shift as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  (0 ... 100).contains(number.doubleValue) else {
                return .error(400, "\"audio_shift\" must be a number from 0 (auto) to 100")
            }
            if number.doubleValue > 0 { override.audioShift = number.floatValue }
        }

        guard canGenerate else {
            return .error(409, validationMessage ?? "a generation is already running")
        }
        guard !isChained || project != nil else {
            return .error(400, "a chained length needs an open project (POST /api/project/new or /api/project/open)")
        }
        guard !isChained || count == 1 else {
            return .error(400, "a chained length makes one video; \"count\" must be 1")
        }
        samplingOverride = override.renoise == nil && override.audioShift == nil ? nil : override
        generate(count: count)
        return .json(202, ["ok": true])
    }
}

/// h3.h's own default - kept here rather than importing CH3 into this
/// small API file just for one constant.
private let defaultStepsForAPI = 20

/// A JSON number as an Int. JSONSerialization hands booleans back as
/// NSNumber too, and `true as? Int` succeeds as 1 - so without this check
/// "reuse": true would quietly run reuse 1.
private func jsonInteger(_ value: Any) -> Int? {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
    return value as? Int
}

/// A seed: a non-negative JSON integer, or a string of decimal digits.
private func jsonUInt64(_ value: Any) -> UInt64? {
    if let text = value as? String {
        guard !text.isEmpty, text.allSatisfy({ $0.isASCII && $0.isNumber }) else { return nil }
        return UInt64(text)
    }
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
    return value as? UInt64
}
