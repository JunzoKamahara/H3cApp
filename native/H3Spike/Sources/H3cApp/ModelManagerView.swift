import AppKit
import SwiftUI

/// The dialog behind the toolbar's "モデル管理" button and the engine-status
/// popover's "モデル管理…" link. Manages ModelLibrary's two independent
/// lists - registered H3 checkpoint directories and registered LoRA files -
/// side by side in one sheet, since both are "pick a file/folder once,
/// reuse it by name later" in the same shape.
struct ModelManagerView: View {
    @ObservedObject var library: ModelLibrary
    /// Called after the active model changes (a direct selection, or the
    /// previously-active model being removed) so GenerationViewModel can
    /// reload the engine. Not called for LoRA changes - the stack only takes
    /// effect at the next generate() call, no engine reload needed.
    var onActiveModelChange: (UUID?) -> Void
    /// A generation is running (the model in use can't be removed then).
    var isGenerating = false

    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme
    @State private var section: Section = .models
    @State private var renamingModelID: UUID?
    @State private var renamingLoRAID: UUID?
    @State private var renameText: String = ""
    @State private var showingDownloadWizard = false
    @State private var detailModel: H3ModelEntry?
    @State private var removingModel: H3ModelEntry?
    @State private var addingRef2VA: H3ModelEntry?
    /// Bumped when a model's folder changed under the list (Ref2VA added):
    /// the rows read their folders while drawing, so they draw again.
    @State private var folderVersion = 0
    @State private var removalMessage: String?
    @State private var showingImport = false
    /// A LoRA picked from a folder macOS guards, waiting for the choice to
    /// move it into the LoRA folder.
    @State private var pendingProtectedLoRA: PendingLoRA?
    @State private var movingLoRA = false
    @State private var loraMoveError: String?

    private struct PendingLoRA: Identifiable {
        let id = UUID()
        let path: String
        let folderName: String
    }

    private var palette: H3Palette { H3Palette(colorScheme) }

    @State private var showingSwiftVRSetup = false

    private enum Section: String, CaseIterable, Identifiable {
        case models, loras, upscalers
        var id: String { rawValue }
        var label: String {
            switch self {
            case .models: return String(localized: "H3モデル")
            case .loras: return "LoRA"
            case .upscalers: return String(localized: "拡大モデル")
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("モデル管理").font(.system(size: 18, weight: .semibold))
                Spacer()
                Button("閉じる") { dismiss() }
            }
            .padding(H3Spacing.lg)

            Picker("", selection: $section) {
                ForEach(Section.allCases) { item in Text(item.label).tag(item) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, H3Spacing.lg)
            .padding(.bottom, H3Spacing.md)

            Divider()

            switch section {
            case .models: modelsSection
            case .loras: lorasSection
            case .upscalers: upscalersSection
            }
        }
        .frame(width: 560, height: 460)
        .background(palette.canvas)
        // A file put back, or the 書類-folder permission granted since,
        // shows up on the next look.
        .onAppear { library.retryFailedLoRAInfo() }
    }

    // MARK: H3 models

    private var modelsSection: some View {
        VStack(spacing: 0) {
            if library.models.isEmpty {
                emptyState(String(localized: "登録されたH3モデルはありません。"))
            } else {
                List {
                    ForEach(library.models) { model in
                        modelRow(model)
                    }
                }
                .listStyle(.inset)
                .id(folderVersion)
            }
            if let removalMessage {
                Text(removalMessage).font(.caption).foregroundStyle(palette.textSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, H3Spacing.md).padding(.top, H3Spacing.sm)
            }
            HStack {
                Text("FL2VA（と、あれば任意でRef2VA）を含むフォルダを登録してください。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                Spacer()
                Button("GGUFを取り込む…") { showingImport = true }
                Button("ダウンロードして追加…") { showingDownloadWizard = true }
                Button("フォルダを追加…") { addModel() }
            }
            .padding(H3Spacing.md)
        }
        .sheet(item: $removingModel) { model in
            ModelRemoveView(model: model, others: library.models.filter { $0.id != model.id },
                            onRemove: { plan in remove(model, deleting: plan) },
                            onCancel: { removingModel = nil })
        }
        .sheet(item: $detailModel) { model in
            ModelDetailView(model: model) { detailModel = nil }
        }
        .sheet(item: $addingRef2VA) { model in
            AddRef2VAView(model: model,
                          onFinished: { addingRef2VA = nil; folderVersion += 1 },
                          onCancelled: { addingRef2VA = nil; folderVersion += 1 })
        }
        .sheet(isPresented: $showingImport) {
            ModelImportView(
                library: library,
                onImported: { path, use in
                    let wasEmpty = library.models.isEmpty
                    library.addModel(path: path)
                    if use, let entry = library.models.first(where: { $0.path == path }) {
                        onActiveModelChange(entry.id)
                    } else if wasEmpty {
                        onActiveModelChange(library.activeModelID)
                    }
                    showingImport = false
                },
                onCancelled: { showingImport = false }
            )
        }
        .sheet(isPresented: $showingDownloadWizard) {
            ModelDownloadWizardView(
                suggestedDestination: defaultH3ModelDownloadPath,
                onCompleted: { path in
                    let wasEmpty = library.models.isEmpty
                    library.addModel(path: path)
                    if wasEmpty { onActiveModelChange(library.activeModelID) }
                    showingDownloadWizard = false
                },
                onCancelled: { showingDownloadWizard = false },
                library: library,
                onImported: { path, use in
                    let wasEmpty = library.models.isEmpty
                    library.addModel(path: path)
                    if use, let entry = library.models.first(where: { $0.path == path }) {
                        onActiveModelChange(entry.id)
                    } else if wasEmpty {
                        onActiveModelChange(library.activeModelID)
                    }
                    showingDownloadWizard = false
                }
            )
        }
    }

    private func modelRow(_ model: H3ModelEntry) -> some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: model.id == library.activeModelID ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(model.id == library.activeModelID ? palette.accent : palette.textSecondary)
            VStack(alignment: .leading, spacing: 2) {
                if renamingModelID == model.id {
                    TextField("名前", text: $renameText, onCommit: {
                        library.renameModel(id: model.id, name: renameText)
                        renamingModelID = nil
                    })
                    .textFieldStyle(.roundedBorder)
                } else {
                    Text(model.name).font(.body)
                }
                Text(model.path)
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(modelStatusText(model))
                    .font(.caption2)
                    .foregroundStyle(palette.textSecondary)
            }
            Spacer()
            Button {
                detailModel = model
            } label: {
                Image(systemName: "info.circle")
            }
            .buttonStyle(.borderless)
            .help("このモデルの中身（DiT・文章の部分・VAEの形式、キャッシュなど）を表示します")
            Menu {
                Button("詳細を表示…") { detailModel = model }
                if !modelHasRef2VA(model) {
                    Button("Ref2VAを追加…") { addingRef2VA = model }
                }
                Button("名前を変更") {
                    renameText = model.name
                    renamingModelID = model.id
                }
                Button("削除…", role: .destructive) { removingModel = model }
                    .disabled(isGenerating && model.id == library.activeModelID)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .contentShape(Rectangle())
        .onTapGesture {
            // switchModel(to:) always reloads (needed after a download
            // finishes into the already-active entry) - guard the redundant
            // case here instead, so re-tapping the current row is a no-op.
            if model.id != library.activeModelID { onActiveModelChange(model.id) }
        }
        .padding(.vertical, 4)
    }

    private func modelStatusText(_ model: H3ModelEntry) -> String {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        func has(_ subpath: String) -> Bool {
            fm.fileExists(atPath: model.path + subpath, isDirectory: &isDir) && isDir.boolValue
        }
        var parts = [
            has("/FL2VA") ? String(localized: "FL2VAあり") : String(localized: "FL2VAなし"),
            has("/Ref2VA") ? String(localized: "Ref2VAあり") : String(localized: "Ref2VAなし"),
        ]
        if let manifest = ModelImportManifest.load(modelPath: model.path) {
            parts.append(String(localized: "GGUFから取り込み（\(manifest.summary)）"))
        }
        return parts.joined(separator: summarySeparator)
    }

    /// Unregisters the model (moving off it first when it is in use, so the
    /// engine lets go of its files), then deletes the planned files.
    /// Unregisters the model (moving off it first when it is in use, so the
    /// engine lets go of its files), then deletes its caches and, if asked,
    /// moves its files to the Trash.
    private func remove(_ model: H3ModelEntry, deleting plan: ModelRemovalPlan) {
        removingModel = nil
        let wasActive = model.id == library.activeModelID
        library.removeModel(id: model.id)
        if wasActive { onActiveModelChange(library.activeModelID) }
        let name = model.name
        removalMessage = plan.includesFiles
            ? String(localized: "「\(name)」のファイルをゴミ箱に移しています…")
            : String(localized: "「\(name)」をモデル管理から外しました（ファイルは残っています）。")
        Task {
            let failed = await Task.detached { plan.execute() }.value
            let format = { (bytes: Int64) in ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file) }
            var message: String
            if !failed.isEmpty {
                message = String(localized: "「\(name)」の一部を削除できませんでした：\(failed.joined(separator: "、"))")
            } else if plan.includesFiles {
                message = plan.freedBytes < 1_000_000
                    ? String(localized: "「\(name)」のファイルをゴミ箱に移しました（ほかと共有しているため、ゴミ箱を空にしてもほとんど空きません）。")
                    : String(localized: "「\(name)」のファイルをゴミ箱に移しました（ゴミ箱を空にすると約\(format(plan.freedBytes))空きます）。")
            } else {
                message = String(localized: "「\(name)」をモデル管理から外しました（ファイルは残っています）。")
            }
            if failed.isEmpty && plan.cacheBytes > 0 {
                message += String(localized: "キャッシュを削除して約\(format(plan.cacheBytes))空きました。")
            }
            removalMessage = message
        }
    }

    private func addModel() {
        guard let path = chooseDirectory() else { return }
        library.addModel(path: path)
    }

    // MARK: LoRA

    private var lorasSection: some View {
        VStack(spacing: 0) {
            if library.loras.isEmpty {
                emptyState(String(localized: "登録されたLoRAはありません。"))
            } else {
                List {
                    ForEach(library.loras) { entry in
                        loraRow(entry)
                    }
                    .onMove { library.moveLoRAs(fromOffsets: $0, toOffset: $1) }
                }
                .listStyle(.inset)
            }
            HStack {
                if movingLoRA {
                    ProgressView().controlSize(.small)
                    Text("LoRAフォルダへ移しています…")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                } else if let loraMoveError {
                    Text(loraMoveError)
                        .font(.caption)
                        .foregroundStyle(palette.errorColor)
                        .lineLimit(2)
                } else {
                    Text("チェックしたLoRAを重ねて適用します（最大\(maxStackedLoRAs)個、ドラッグで並べ替え）。")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                }
                Spacer()
                Button("フォルダを開く") { openLoRAFolder() }
                    .help("LoRAフォルダ（\(defaultLoRADirectory)）をFinderで開きます。ここにファイルを入れてから「追加…」で選べます。")
                Button("追加…") { addLoRA() }
                    .disabled(movingLoRA)
            }
            .padding(H3Spacing.md)
        }
        .confirmationDialog("LoRAフォルダへ移しますか？", isPresented: Binding(
            get: { pendingProtectedLoRA != nil }, set: { if !$0 { pendingProtectedLoRA = nil } }
        ), presenting: pendingProtectedLoRA) { pending in
            Button("移して登録") { addLoRA(pending.path, moving: true, copying: false) }
            Button("コピーして登録") { addLoRA(pending.path, moving: true, copying: true) }
            Button("このまま登録") { addLoRA(pending.path, moving: false, copying: false) }
            Button("キャンセル", role: .cancel) {}
        } message: { pending in
            Text("選んだファイルは「\(pending.folderName)」にあります。macOSが保護しているフォルダなので、このまま登録すると、アプリを起動するたびにアクセスの許可を求められることがあります。LoRAフォルダ（\(defaultLoRADirectory)）へ移すと、求められなくなります。")
        }
    }

    private func loraRow(_ entry: LoRAEntry) -> some View {
        VStack(alignment: .leading, spacing: H3Spacing.xs) {
            loraMainRow(entry)
            // Its own line under the name column: inside that column it
            // widened the row past the sheet and pushed the menu to the edge.
            loraSamplingControls(entry)
                .padding(.leading, loraCheckboxIndent)
        }
        .padding(.vertical, 4)
    }

    /// Checkbox width plus the row spacing, so the line below lines up
    /// with the name.
    private let loraCheckboxIndent: CGFloat = 16 + H3Spacing.sm

    private func loraMainRow(_ entry: LoRAEntry) -> some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Toggle("", isOn: Binding(
                get: { entry.enabled },
                set: { library.setLoRAEnabled(id: entry.id, $0) }
            ))
            .toggleStyle(.checkbox)
            .labelsHidden()
            .disabled(!entry.enabled && library.enabledLoRAs.count >= maxStackedLoRAs)
            VStack(alignment: .leading, spacing: 2) {
                if renamingLoRAID == entry.id {
                    TextField("名前", text: $renameText, onCommit: {
                        library.renameLoRA(id: entry.id, name: renameText)
                        renamingLoRAID = nil
                    })
                    .textFieldStyle(.roundedBorder)
                } else {
                    Text(entry.name).font(.body)
                }
                Text(entry.path)
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                loraInfoText(entry)
            }
            Spacer()
            HStack(spacing: 4) {
                Text("強さ").font(.caption2).foregroundStyle(palette.textSecondary)
                TextField("1", text: Binding(
                    get: { entry.scaleText },
                    set: { library.setLoRAScale(id: entry.id, scaleText: $0) }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(width: 56)
                .help("学習時の効き具合に対する倍率。空欄は1。")
                Text("ステップ").font(.caption2).foregroundStyle(palette.textSecondary)
                TextField("—", text: Binding(
                    get: { entry.recommendedSteps.map(String.init) ?? "" },
                    set: { library.setLoRARecommendedSteps(id: entry.id, text: $0) }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(width: 40)
                .help("Turbo（蒸留）LoRAの学習ステップ数。オンにすると生成ステップ数がこれに切り替わります。空欄は通常のLoRA。")
            }
            Menu {
                Button("名前を変更") {
                    renameText = entry.name
                    renamingLoRAID = entry.id
                }
                if protectedFolderName(for: entry.path) != nil {
                    Button("LoRAフォルダへ移す") { moveRegisteredLoRA(entry) }
                        .disabled(movingLoRA)
                }
                Button("削除", role: .destructive) { library.removeLoRA(id: entry.id) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
    }

    /// What h3_lora_inspect read from the file's header: layout, how much
    /// of the DiT it touches, rank - or why it can't be used.
    private func loraInfoText(_ entry: LoRAEntry) -> some View {
        let text: String
        var isError = false
        switch library.loraInfo(for: entry) {
        case nil:
            text = String(localized: "ファイルの情報を読み込んでいます…")
        case .success(let info)?:
            var parts = [info.format,
                         String(localized: "\(info.blocks)/50ブロック") + (info.refinerBlocks > 0 ? "+refiner" : ""),
                         info.rankMin == info.rankMax ? "rank \(info.rankMax)" : "rank \(info.rankMin)–\(info.rankMax)"]
            if info.unsupported > 0 { parts.append(String(localized: "未対応\(info.unsupported)件は無視")) }
            text = parts.joined(separator: summarySeparator)
        case .failure(let error)?:
            text = FileManager.default.fileExists(atPath: entry.path)
                ? String(localized: "MiniMax-H3用として読み込めません: \(error.localizedDescription)")
                : String(localized: "ファイルが見つかりません")
            isError = true
        }
        return Text(text)
            .font(.caption2)
            .foregroundStyle(isError ? palette.errorColor : palette.textSecondary)
            .lineLimit(2)
    }

    /// The step rule and audio shift this LoRA runs with, set by hand for
    /// files that don't declare them (see ModelLibrary.sampling(for:)).
    /// 自動 shows what the file's metadata says.
    private func loraSamplingControls(_ entry: LoRAEntry) -> some View {
        let info = try? library.loraInfo(for: entry)?.get()
        let declared = info.map { $0.recommendsRenoise ? "re-noise" : "Euler" } ?? "…"
        let declaredShift = info?.recommendedAudioShift.map { $0.formatted() } ?? "3"
        return HStack(spacing: 6) {
            Text("ステップの進め方").font(.caption2).foregroundStyle(palette.textSecondary)
                .fixedSize()
            Picker("", selection: Binding(
                get: { entry.sampler },
                set: { library.setLoRASampler(id: entry.id, $0) }
            )) {
                Text("自動（\(declared)）").tag(LoRASampler.auto)
                Text("Euler").tag(LoRASampler.euler)
                Text("re-noise（LCM方式）").tag(LoRASampler.renoise)
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .controlSize(.small)
            .fixedSize()
            .help("少ないステップ向けに蒸留したLoRAの多くは、re-noise（ComfyUIのlcmサンプラーと同じ計算）を推奨しています。自動は、ファイルの説明に書かれた方式を使います（書かれていなければEuler）。re-noiseでは、ノイズ除去の再利用は1になります。")
            Text("音声シフト").font(.caption2).foregroundStyle(palette.textSecondary)
                .fixedSize()
            TextField(declaredShift, text: Binding(
                get: { entry.audioShiftText },
                set: { library.setLoRAAudioShift(id: entry.id, text: $0) }
            ))
            .textFieldStyle(.roundedBorder)
            .controlSize(.small)
            .frame(width: 44)
            .help("音声のノイズの減らし方を決める値。空欄は、ファイルの説明に書かれた値（なければ標準の3）。")
        }
    }

    private func addLoRA() {
        try? FileManager.default.createDirectory(atPath: defaultLoRADirectory, withIntermediateDirectories: true)
        guard let path = chooseFile(allowedContentTypes: safetensorsTypes, startingIn: defaultLoRADirectory) else { return }
        loraMoveError = nil
        if let folder = protectedFolderName(for: path) {
            pendingProtectedLoRA = PendingLoRA(path: path, folderName: folder)
        } else {
            library.addLoRA(path: path)
        }
    }

    private func addLoRA(_ path: String, moving: Bool, copying: Bool) {
        guard moving else { library.addLoRA(path: path); return }
        relocate(path, copying: copying) { library.addLoRA(path: $0) }
    }

    /// 「LoRAフォルダへ移す」 for an entry already registered from a guarded
    /// folder: the entry keeps its old path so older records still match.
    private func moveRegisteredLoRA(_ entry: LoRAEntry) {
        relocate(entry.path, copying: false) { library.relocateLoRA(id: entry.id, to: $0) }
    }

    private func relocate(_ path: String, copying: Bool, then done: @escaping (String) -> Void) {
        movingLoRA = true
        loraMoveError = nil
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Result { try ModelLibrary.moveLoRAFileToFolder(path, copy: copying) }
            }.value
            movingLoRA = false
            switch result {
            case .success(let newPath): done(newPath)
            case .failure(let error):
                loraMoveError = String(localized: "LoRAフォルダへ移せませんでした: \(error.localizedDescription)")
            }
        }
    }

    private func openLoRAFolder() {
        try? FileManager.default.createDirectory(atPath: defaultLoRADirectory, withIntermediateDirectories: true)
        NSWorkspace.shared.open(URL(fileURLWithPath: defaultLoRADirectory, isDirectory: true))
    }

    // MARK: Upscalers (the models the 拡大 sheet's AI methods need)

    private var upscalersSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            List {
                HStack(alignment: .top, spacing: H3Spacing.md) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(VideoUpscaler.swiftVR.label).font(.system(size: 13, weight: .medium))
                        Text(swiftVRStatusText).font(.caption).foregroundStyle(palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(verbatim: SwiftVRModel.directory).font(.caption2).foregroundStyle(palette.textSecondary)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    Button(VideoUpscaler.swiftVR.isAvailable ? "管理…" : "準備…") { showingSwiftVRSetup = true }
                }
                .padding(.vertical, 4)
                HStack(alignment: .top, spacing: H3Spacing.md) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Real-ESRGAN 汎用・アニメ").font(.system(size: 13, weight: .medium))
                        Text("アプリに含まれています（BSD-3-Clause）。準備は要りません。")
                            .font(.caption).foregroundStyle(palette.textSecondary)
                    }
                    Spacer()
                }
                .padding(.vertical, 4)
            }
            .listStyle(.inset)
            Text("「動画を拡大」のAI超解像が使うモデルです。SwiftVRの重みは研究・非商用のライセンスのため同梱せず、ここでダウンロードして変換します。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(H3Spacing.md)
        }
        .sheet(isPresented: $showingSwiftVRSetup) {
            SwiftVRSetupView(busy: isGenerating) { showingSwiftVRSetup = false }
        }
    }

    private var swiftVRStatusText: String {
        if SwiftVRSetup.shared.isBusy {
            return String(localized: "準備中です（\(SwiftVRSetup.shared.phase.apiText)）")
        }
        if VideoUpscaler.swiftVR.isAvailable {
            let size = ByteCountFormatter.string(fromByteCount: SwiftVRModel.bytesOnDisk(), countStyle: .file)
            if SwiftVRModel.originalRemoved {
                return String(localized: "使えます（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")で変換済み、元の重みは削除済み、\(size)）")
            }
            return String(localized: "使えます（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")で変換済み、\(size)）")
        }
        if SwiftVRModel.originalRemoved {
            return String(localized: "元の重みを削除したあとで、キャッシュが使えなくなりました。もう一度ダウンロードと変換が必要です。")
        }
        if SwiftVRModel.weightsPresent { return String(localized: "重みはありますが、このMac用に変換されていません。") }
        return String(localized: "まだありません。約20GBのダウンロードと変換が必要です。")
    }

    private func emptyState(_ text: String) -> some View {
        VStack {
            Spacer()
            Text(text).font(.callout).foregroundStyle(palette.textSecondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}
