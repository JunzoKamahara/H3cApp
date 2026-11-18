import AppKit
import Foundation
import H3Engine

// 高解像度化 of a project image (ImageUpscale.swift): the redraw runs as an
// image generation (its progress, 途中の様子 and 中止), the frame enlargers
// as an enlargement (the banner and its 中止). Either way the result is a
// new project image next to the original, with a record saying where it
// came from, and the preview shows it.

extension GenerationViewModel {
    /// Why `method` can't enlarge now, if it can't.
    func imageUpscaleBlockedReason(_ method: ImageUpscaleMethod) -> String? {
        if isGenerating { return String(localized: "生成中は高解像度化できません。生成が終わってからにしてください") }
        if isUpscalingVideo { return String(localized: "拡大しています。終わってからにしてください") }
        switch method {
        case .redraw:
            guard let model = library.activeImageModel else {
                return String(localized: "画像モデルがまだ追加されていません")
            }
            if !FileManager.default.fileExists(atPath: model.path) {
                return String(localized: "画像モデル「\(model.name)」のフォルダが見つかりません")
            }
            // As for generating: not while the text encoder is compared or
            // chosen, and not without one to read.
            return Self.imageTextEncoderProblem(model: model.path, name: model.name, videoModelPaths: videoModelPaths)
        case .frame(let upscaler):
            if !upscaler.method.isAvailable { return upscaler.method.missingReason }
            return upscaleBlockedReason(upscaler)
        }
    }

    /// The prompt a redraw uses: the image's own, or for an imported one the
    /// panel's image prompt.
    func redrawPrompt(for image: ProjectImage) -> String {
        let own = image.record?.prompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return own.isEmpty ? imagePrompt.trimmingCharacters(in: .whitespacesAndNewlines) : own
    }

    /// Starts 高解像度化 of `image`. Returns false (with projectMessage) when it
    /// cannot start.
    @discardableResult
    func upscaleImage(_ image: ProjectImage, method: ImageUpscaleMethod, factor: UpscaleFactor,
                      strength: RedrawStrength = .standard) -> Bool {
        if let reason = imageUpscaleBlockedReason(method) {
            projectMessage = reason
            return false
        }
        guard let size = image.record.map({ (width: $0.width, height: $0.height) })
                ?? ImageMetadata.loadImage(at: image.url).map({ (width: $0.width, height: $0.height) }),
              let made = ImageUpscale.choices(method, width: size.width, height: size.height)
                .first(where: { $0.factor == factor }) else {
            projectMessage = String(localized: "この画像は、この方法・倍率では高解像度化できません")
            return false
        }
        switch method {
        case .redraw:
            return redraw(image, width: made.width, height: made.height, factor: factor, strength: strength)
        case .frame(let upscaler):
            enlarge(image, upscaler: upscaler, factor: factor, width: made.width, height: made.height)
            return true
        }
    }

    // MARK: 画像モデルで描き直す

    private func redraw(_ source: ProjectImage, width: Int, height: Int, factor: UpscaleFactor,
                        strength: RedrawStrength) -> Bool {
        guard let model = library.activeImageModel else { return false }
        if project == nil { startUntitledProject() }
        guard let project else { return false }
        let prompt = redrawPrompt(for: source)
        guard !prompt.isEmpty else {
            projectMessage = String(localized: "描き直すにはプロンプトが要ります。取り込んだ画像のときは、パネルの画像のプロンプトに内容を書いてください")
            return false
        }
        let engine: H3ImageEngine
        do {
            engine = try imageEngine(for: model)
        } catch {
            errorMessage = Self.userFacingMessage(for: error)
            return false
        }
        var draft = source.record?.draft ?? currentImageDraft() ?? .blank
        draft.prompt = prompt
        draft.model = model.id.uuidString
        draft.size = ImageSizeProfile.canvas(width: Int32(width), height: Int32(height)).rawValue
        draft.references = nil
        draft.count = 1
        let seed = source.record?.seed.flatMap(UInt64.init) ?? UInt64.random(in: UInt64.min ... UInt64.max)
        let installed = H3ImageModel.installed(at: model.path)
        let projectURL = project.url
        var request = ImageStoreRequest(
            projectURL: projectURL, draft: draft, references: [], modelName: model.name,
            systemName: engine.modelName, modelSource: installed?.weightsRepo,
            modelRevision: installed?.weightsRevision,
            deviceLine: engineState == .ready ? deviceLine : nil, appVersion: Self.appVersion,
            recordsPrompt: GenerationMetadata.exportsSettings)
        request.upscale = .init(from: source.url.lastPathComponent, method: ImageUpscaleMethod.redraw.id,
                                factor: factor.rawValue, strength: Double(strength.sigma))
        let params = H3ImageParams(prompt: prompt, width: Int32(width), height: Int32(height), seed: seed,
                                   latentPreview: preferences.showImageDenoisePreview,
                                   initImagePath: source.url.path, initStrength: strength.sigma)
        imageBatchStopped = false
        displayedImage = nil
        imageGenerationBegan(index: 1, total: 1, reading: engine)
        imageGenerationTask = Task {
            let startedAt = Date()
            var finished: H3ImageResult?
            do {
                for try await event in engine.generate(params) {
                    switch event {
                    case .progress(let phase, let completed, let total):
                        self.imageGenerationProgress(phase: phase, completed: completed, total: total)
                    case .latentPreview(let preview):
                        self.showImageLatentPreview(preview, isPH3: engine.isPH3)
                    case .finished(let result):
                        finished = result
                    }
                }
            } catch {
                if case H3EngineError.cancelled = error {
                    self.phase = String(localized: "生成を中止しました")
                } else {
                    self.errorMessage = Self.userFacingMessage(for: error)
                }
            }
            if let result = finished {
                let seconds = Date().timeIntervalSince(startedAt)
                let stored = await Task.detached(priority: .userInitiated) {
                    Result { try request.store(result, seedWasRandom: false, completedAt: Date(),
                                               generationSeconds: seconds) }
                }.value
                switch stored {
                case .success(let image):
                    if self.project?.url == projectURL {
                        self.reloadProjectImages()
                        self.displayedImage = self.projectImages.first { $0.url == image.url } ?? image
                        self.resultURL = nil
                        self.lastResult = nil
                    }
                    self.projectMessage = String(localized: "\(source.url.lastPathComponent)を\("\(width)×\(height)")に描き直して、\(image.url.lastPathComponent)をつくりました（\(formatElapsed(seconds))）")
                    self.projectMessageFile = image.url
                case .failure(let error):
                    self.errorMessage = String(localized: "画像を保存できませんでした（詳細: \(error.localizedDescription)）")
                }
            }
            self.imageGenerationEnded()
            self.imageGenerationTask = nil
        }
        return true
    }

    // MARK: The frame enlargers

    private func enlarge(_ source: ProjectImage, upscaler: VideoUpscaler, factor: UpscaleFactor,
                         width: Int, height: Int) {
        guard let project else { return }
        let projectURL = project.url
        let record = source.record
        let appVersion = Self.appVersion
        let recordsPrompt = GenerationMetadata.exportsSettings
        upscalingVideoName = source.url.lastPathComponent
        upscalingFactor = factor
        upscaleProgress = 0
        upscaleCancel.reset()
        upscalingIsImage = true
        isUpscalingVideo = true
        let cancel = upscaleCancel
        let started = Date()
        Task {
            let outcome = await Task.detached(priority: .userInitiated) { () -> Result<ProjectImage, Error> in
                Result {
                    guard let picture = ImageMetadata.loadImage(at: source.url) else {
                        throw ProjectImageError.unreadable(source.url.lastPathComponent)
                    }
                    let made = try ImageUpscale.enlarge(
                        picture, upscaler: upscaler, factor: factor, width: width, height: height,
                        progress: { share in Task { @MainActor [weak self] in self?.upscaleProgress = share } },
                        isCancelled: { cancel.isRequested })
                    let now = Date()
                    // Still AI-generated when the original was: said in the
                    // PNG as the original's is.
                    let disclosure = record.flatMap { record -> ImageMetadata.Disclosure? in
                        guard !record.isImported else { return nil }
                        return ImageMetadata.Disclosure(
                            creatorTool: "H3cApp \(appVersion)", createdAt: now,
                            system: record.model?.name ?? "MiniMax H3 Image", systemVersion: record.model?.name,
                            prompt: recordsPrompt ? record.prompt : nil)
                    }
                    guard let png = ImageMetadata.pngData(made, disclosure: disclosure) else {
                        throw ProjectImageError.cannotEncode
                    }
                    let seed = record?.seed.flatMap(UInt64.init)
                    return try ProjectFiles.storeImage(png: png, latent: nil, completedAt: now, seed: seed,
                                                       into: projectURL) { name, sha256 in
                        var copy = record ?? ProjectImageRecord(
                            assetID: "", image: name, sha256: sha256, completedAt: now,
                            width: width, height: height, appVersion: appVersion)
                        copy.assetID = UUID().uuidString
                        copy.image = name
                        copy.sha256 = sha256
                        copy.completedAt = now
                        copy.width = width
                        copy.height = height
                        copy.appVersion = appVersion
                        copy.latent = nil
                        copy.latentMissingReason = String(localized: "拡大した画像です（潜在ファイルは元の画像にあります）")
                        copy.generationSeconds = Date().timeIntervalSince(started)
                        copy.upscaledFrom = source.url.lastPathComponent
                        copy.upscaler = upscaler.rawValue
                        copy.upscaleFactor = factor.rawValue
                        copy.upscaleStrength = nil
                        return copy
                    }
                }
            }.value
            self.isUpscalingVideo = false
            self.upscalingIsImage = false
            switch outcome {
            case .success(let image):
                if self.project?.url == projectURL {
                    self.reloadProjectImages()
                    if let shown = self.projectImages.first(where: { $0.url == image.url }) {
                        self.showProjectImage(shown)
                    }
                }
                self.projectMessage = String(localized: "\(source.url.lastPathComponent)を\("\(width)×\(height)")に拡大して、\(image.url.lastPathComponent)をつくりました（\(upscaler.label)、\(formatElapsed(Date().timeIntervalSince(started)))）")
                self.projectMessageFile = image.url
            case .failure(let error):
                if case VideoUpscale.UpscaleError.cancelled = error {
                    self.projectMessage = String(localized: "拡大を中止しました")
                } else {
                    NSLog("H3cApp: enlarging image %@ failed: %@", source.url.path, String(describing: error))
                    self.projectMessage = String(localized: "画像を拡大できませんでした（詳細: \(error.localizedDescription)）")
                }
            }
        }
    }
}
