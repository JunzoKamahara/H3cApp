import AppKit
import Foundation
import H3Engine

// The image side of a window (plans/IMAGE_PLAN.ja.md, phase 1): its draft,
// the project's images, bringing image files in, the preview, and handing
// an image to the video form. Generating comes with the image engine
// (phase 2); until then an image reaches a project by being brought in.

extension GenerationViewModel {
    // MARK: Draft

    /// What project.json records as the kind: nothing while making videos,
    /// so a project that never left the video side is written as before.
    var storedGenerationKind: String? {
        generationKind == .video ? nil : generationKind.rawValue
    }

    /// The image side of the form; nil while it is untouched, so project.json
    /// gains nothing until images are used.
    func currentImageDraft() -> ImageDraft? {
        let draft = ImageDraft(prompt: imagePrompt, model: nil, size: imageSizeProfile.rawValue,
                               seedFixed: imageSeedFixed, seed: imageSeedText, count: imageCount)
        return draft == .blank && lastSavedProjectFile?.imageDraft == nil ? nil : draft
    }

    func applyImageDraft(_ draft: ImageDraft) {
        imagePrompt = draft.prompt
        if let profile = ImageSizeProfile(rawValue: draft.size) { imageSizeProfile = profile }
        imageSeedFixed = draft.seedFixed
        imageSeedText = draft.seed
        imageCount = draft.count.clamped(to: imageCountRange)
    }

    // MARK: Validation and summary

    /// Why an image cannot be generated now (shown under the prompt).
    var imageValidationMessage: String? {
        if isImportingImages { return String(localized: "画像を取り込んでいます…") }
        if library.activeImageModel == nil {
            return String(localized: "画像モデルがまだ追加されていません。手元の画像は「取り込む」でプロジェクトに入れられます")
        }
        if promptIsEmpty { return String(localized: "画像の内容を入力してください") }
        if imageSeedFixed && UInt64(imageSeedText) == nil {
            return imageSeedText.isEmpty ? String(localized: "シード値を入力してください")
                                         : String(localized: "シード値は0〜18446744073709551615の整数にしてください")
        }
        // The engine that runs the model comes in phase 2 of the plan.
        return String(localized: "この版では、まだ画像を生成できません")
    }

    var canGenerateImage: Bool {
        !isGenerating && !isImportingImages && !isMovingProjectFiles && imageValidationMessage == nil
    }

    var imageDraftSummaryText: String {
        var parts = [imageSizeProfile.label, String(localized: "\(imageCount)枚")]
        if imageSizeProfile.isVideoCanvas { parts.append(String(localized: "動画の開始画像用")) }
        if imageSeedFixed { parts.append(String(localized: "シード固定")) }
        if let model = library.activeImageModel { parts.append(model.name) }
        return parts.joined(separator: summarySeparator)
    }

    // MARK: The project's images

    func reloadProjectImages() {
        projectImages = project.map { ProjectFiles.images(in: $0.url) } ?? []
        // The image in the preview (this project's, or another's shown from
        // すべてのプロジェクト) may have been moved or deleted.
        if let displayed = displayedImage {
            displayedImage = projectImages.first { $0.url == displayed.url }
                ?? (FileManager.default.fileExists(atPath: displayed.url.path) ? displayed : nil)
        }
    }

    /// 画像を取り込む…: image files become images of the project, as PNG,
    /// each with a record saying where it came from. They are not marked as
    /// AI-generated here: the app did not make them. What a file says of its
    /// own origin stays in it.
    func chooseAndImportImages() {
        let paths = chooseFiles(allowedContentTypes: imageTypes)
        guard !paths.isEmpty else { return }
        importImages(from: paths.map { URL(fileURLWithPath: $0) })
    }

    func importImages(from sources: [URL]) {
        guard !isImportingImages else { return }
        if project == nil { startUntitledProject() }
        guard let project else { return }
        let projectURL = project.url
        let appVersion = Self.appVersion
        isImportingImages = true
        Task { [weak self] in
            let results = await Task.detached(priority: .userInitiated) { () -> [Result<ProjectImage, Error>] in
                sources.map { source in
                    Result {
                        guard let image = ImageMetadata.pngForImport(at: source) else {
                            throw ProjectImageError.unreadable(source.lastPathComponent)
                        }
                        let now = Date()
                        return try ProjectFiles.storeImage(
                            png: image.png, latent: nil, completedAt: now, seed: nil, into: projectURL
                        ) { name, sha256 in
                            ProjectImageRecord(assetID: UUID().uuidString, image: name, sha256: sha256,
                                               completedAt: now, width: image.width, height: image.height,
                                               appVersion: appVersion, importedFrom: source.lastPathComponent)
                        }
                    }
                }
            }.value
            guard let self else { return }
            self.isImportingImages = false
            var imported: [ProjectImage] = []
            var failure: Error?
            for result in results {
                switch result {
                case .success(let image): imported.append(image)
                case .failure(let error): failure = error
                }
            }
            if let failure {
                self.projectMessage = String(localized: "画像を取り込めませんでした（詳細: \(failure.localizedDescription)）")
            }
            guard self.project?.url == projectURL else { return }
            self.reloadProjectImages()
            // The last one brought in goes to the preview.
            if let last = imported.last, let image = self.projectImages.first(where: { $0.url == last.url }) {
                self.showProjectImage(image)
            }
        }
    }

    // MARK: Preview

    /// Shows a project image in the preview, in place of a video.
    func showProjectImage(_ image: ProjectImage) {
        guard !isGenerating else { return }
        deleteTemporaryPreview()
        resultURL = nil
        lastResult = nil
        displayedImage = image
    }

    /// Puts an image's request back into the image form with its seed
    /// pinned. An image brought in from a file has none.
    func restoreProjectImage(_ image: ProjectImage) {
        guard var draft = image.record?.draft else { return }
        if let seed = image.record?.seed {
            draft.seedFixed = true
            draft.seed = seed
        }
        generationKind = .image
        applyImageDraft(draft)
    }

    /// Moves the image, its record, latent and thumbnail to the Trash.
    func deleteProjectImage(_ image: ProjectImage) {
        do {
            try FileManager.default.trashItem(at: image.url, resultingItemURL: nil)
            for url in [ProjectFiles.imageRecordURL(for: image.url), ProjectFiles.imageLatentURL(for: image.url)]
            where FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.trashItem(at: url, resultingItemURL: nil)
            }
            // A cache, made again when needed: just removed.
            try? FileManager.default.removeItem(at: ProjectFiles.imageThumbnailURL(for: image.url))
            if displayedImage?.url == image.url { displayedImage = nil }
        } catch {
            projectMessage = String(localized: "画像を削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        reloadProjectImages()
    }

    // MARK: Handing an image to the video form

    /// この画像から動画を作る: the video form starts from this image. The
    /// image's prompt goes into an empty video prompt. Nothing is generated.
    func makeVideo(from image: ProjectImage) {
        let path = image.url.path
        noteInputsChosen([path])
        generationKind = .video
        creationMethod = .image
        imageInputMode = .firstLastFrame
        firstFramePath = path
        // The prompt this image was made from; one brought in has none.
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let made = image.record?.prompt, !made.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            prompt = made
        }
    }

    /// 最後の画像に使う: the video form ends on this image.
    func useImageAsLastFrame(_ image: ProjectImage) {
        let path = image.url.path
        noteInputsChosen([path])
        generationKind = .video
        creationMethod = .image
        imageInputMode = .firstLastFrame
        lastFramePath = path
    }

    /// 参照画像に追加: the video form takes this image as a reference. That
    /// mode needs the model's Ref2VA; returns false when the active model
    /// has none (the image is added all the same, for when it does).
    @discardableResult
    func addImageAsReference(_ image: ProjectImage) -> Bool {
        let path = image.url.path
        noteInputsChosen([path])
        generationKind = .video
        creationMethod = .image
        imageInputMode = .referenceImage
        if !referenceImages.contains(where: { $0.path == path }) {
            referenceImages.append(H3ReferenceInput(kind: .image, path: path))
        }
        return library.activeModel.map(modelHasRef2VA) ?? true
    }
}
