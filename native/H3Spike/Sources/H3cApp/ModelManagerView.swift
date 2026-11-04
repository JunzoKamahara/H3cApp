import AppKit
import H3Engine
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
    @State private var showingSeedVR2Setup = false
    @State private var showingOpenH3Setup = false

    private enum Section: String, CaseIterable, Identifiable {
        case models, loras, refmods, upscalers
        var id: String { rawValue }
        var label: String {
            switch self {
            case .models: return String(localized: "H3モデル")
            case .loras: return "LoRA"
            case .refmods: return "RefMod"
            case .upscalers: return String(localized: "拡大モデル")
            }
        }
    }

    @State private var renamingRefModID: UUID?
    @State private var refmodRenameText = ""
    @State private var refmodMessage: String?

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
            case .refmods: refmodsSection
            case .upscalers: upscalersSection
            }
        }
        .frame(width: 560, height: 460)
        .background(palette.canvas)
        // A file put back, or the 書類-folder permission granted since,
        // shows up on the next look.
        .onAppear {
            library.retryFailedLoRAInfo()
            library.retryFailedRefModInfo()
            library.syncRefModFolder()
        }
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
            if let provenance = library.provenance(for: entry) {
                loraProvenanceText(provenance)
                    .padding(.leading, loraCheckboxIndent)
            }
        }
        .padding(.vertical, 4)
    }

    /// Where a file whose SHA-256 the app knows came from, and what that
    /// revision declared (ModelLibrary.knownLoRAProvenance). A record, not
    /// a judgement: which conditions apply to a use is the user's call.
    private func loraProvenanceText(_ provenance: LoRAProvenance) -> some View {
        Text("取得元の記録: \(provenance.repo) revision \(provenance.revisionShort)（\(provenance.revisionDate)）のファイルとSHA-256が一致。その版のREADMEはライセンスを\(provenance.declaredLicense)と宣言しています。\(provenance.note)")
            .font(.caption2)
            .foregroundStyle(palette.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
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

    // MARK: RefMods (saved reference latents, H3RefModInfo)

    private var refmodsSection: some View {
        VStack(spacing: 0) {
            if library.refmods.isEmpty {
                emptyState(String(localized: "RefModはありません。参照画像・動画・音声の行の「RefModとして保存…」で作るか、「追加…」でファイルを入れます。"))
            } else {
                List {
                    ForEach(library.refmods) { entry in
                        refmodRow(entry)
                    }
                    .onMove { library.moveRefMods(fromOffsets: $0, toOffset: $1) }
                }
                .listStyle(.inset)
            }
            HStack {
                if let refmodMessage {
                    Text(refmodMessage)
                        .font(.caption)
                        .foregroundStyle(palette.errorColor)
                        .lineLimit(2)
                } else {
                    Text("参照の「RefModを追加…」から選べます。ComfyUI-MiniMaxH3Mod形式のファイルと互換です。")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                }
                Spacer()
                Button("フォルダを開く") { openRefModFolder() }
                    .help("RefModフォルダ（\(defaultRefModDirectory)）をFinderで開きます。ここに入れたファイルは次に開いたときに一覧に載ります。")
                Button("追加…") { addRefMod() }
                    .help("RefModファイル（.safetensors）をRefModフォルダにコピーして登録します。")
            }
            .padding(H3Spacing.md)
        }
    }

    private func refmodRow(_ entry: RefModEntry) -> some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: refmodIcon(entry))
                .foregroundStyle(palette.textSecondary)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 2) {
                if renamingRefModID == entry.id {
                    TextField("名前", text: $refmodRenameText, onCommit: {
                        library.renameRefMod(id: entry.id, name: refmodRenameText)
                        renamingRefModID = nil
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
                refmodInfoText(entry)
            }
            Spacer()
            Menu {
                Button("名前を変更") {
                    refmodRenameText = entry.name
                    renamingRefModID = entry.id
                }
                Button("Finderで表示") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: entry.path)])
                }
                Divider()
                if entry.isInRefModFolder {
                    Button("ファイルをゴミ箱に入れて削除", role: .destructive) { trashRefMod(entry) }
                } else {
                    Button("一覧から外す", role: .destructive) { library.removeRefMod(id: entry.id) }
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .padding(.vertical, 4)
    }

    private func refmodIcon(_ entry: RefModEntry) -> String {
        guard case .success(let info)? = library.refmodInfo(for: entry) else { return "cube" }
        if info.isBundle { return "cube.box" }
        switch info.members.first?.kind {
        case .video?: return "video"
        case .audio?: return "waveform"
        default: return "photo"
        }
    }

    /// What the file holds: kind, latent size, DiT tokens and mode per
    /// member (a bundle lists each).
    @ViewBuilder
    private func refmodInfoText(_ entry: RefModEntry) -> some View {
        switch library.refmodInfo(for: entry) {
        case nil:
            Text("ファイルを読んでいます…").font(.caption2).foregroundStyle(palette.textSecondary)
        case .failure(let error)?:
            Text(error.localizedDescription).font(.caption2).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        case .success(let info)?:
            VStack(alignment: .leading, spacing: 1) {
                if info.isBundle {
                    Text("バンドル（\(info.members.count)件、形式v\(info.formatVersion)）")
                        .font(.caption2).foregroundStyle(palette.textSecondary)
                }
                ForEach(info.members) { member in
                    Text(refmodMemberSummary(member, inBundle: info.isBundle))
                        .font(.caption2).foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func addRefMod() {
        try? FileManager.default.createDirectory(atPath: defaultRefModDirectory, withIntermediateDirectories: true)
        guard let path = chooseFile(allowedContentTypes: safetensorsTypes) else { return }
        refmodMessage = nil
        do {
            let info = try H3RefModInfo.inspect(path: path)
            let target = (path as NSString).deletingLastPathComponent == defaultRefModDirectory
                ? path : try ModelLibrary.moveRefModFileToFolder(path, copy: true)
            library.addRefMod(path: target, name: info.name.isEmpty ? nil : info.name)
        } catch {
            refmodMessage = String(localized: "RefModを追加できませんでした: \(error.localizedDescription)")
        }
    }

    private func trashRefMod(_ entry: RefModEntry) {
        refmodMessage = nil
        do {
            try FileManager.default.trashItem(at: URL(fileURLWithPath: entry.path), resultingItemURL: nil)
            library.removeRefMod(id: entry.id)
        } catch {
            refmodMessage = String(localized: "RefModを削除できませんでした: \(error.localizedDescription)")
        }
    }

    private func openRefModFolder() {
        try? FileManager.default.createDirectory(atPath: defaultRefModDirectory, withIntermediateDirectories: true)
        NSWorkspace.shared.open(URL(fileURLWithPath: defaultRefModDirectory, isDirectory: true))
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
                        Text(VideoUpscaler.seedVR2.label).font(.system(size: 13, weight: .medium))
                        Text(seedVR2StatusText).font(.caption).foregroundStyle(palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(verbatim: SeedVR2Model.directory).font(.caption2).foregroundStyle(palette.textSecondary)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    Button(VideoUpscaler.seedVR2.isAvailable ? "管理…" : "準備…") { showingSeedVR2Setup = true }
                }
                .padding(.vertical, 4)
                HStack(alignment: .top, spacing: H3Spacing.md) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(VideoUpscaler.openH3.label).font(.system(size: 13, weight: .medium))
                        Text(openH3StatusText).font(.caption).foregroundStyle(palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(verbatim: OpenH3Model.directory).font(.caption2).foregroundStyle(palette.textSecondary)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    Button(VideoUpscaler.openH3.isAvailable ? "管理…" : "準備…") { showingOpenH3Setup = true }
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
            Text("「動画を拡大」のAI超解像が使うモデルです。SwiftVRとSeedVR2の重みは同梱せず、ここで配布元のApache License 2.0と宣言されていた版をダウンロードします（SHA-256で照合。SwiftVRはこのMac用に変換もします）。Open-H3 再生成は、NVIDIAの結合LoRA（MiniMax H3 Community License）から精緻化の部分をこのMacで取り出して使います。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(H3Spacing.md)
        }
        .sheet(isPresented: $showingSwiftVRSetup) {
            SwiftVRSetupView(busy: isGenerating) { showingSwiftVRSetup = false }
        }
        .sheet(isPresented: $showingSeedVR2Setup) {
            SeedVR2SetupView(busy: isGenerating) { showingSeedVR2Setup = false }
        }
        .sheet(isPresented: $showingOpenH3Setup) {
            OpenH3SetupView(busy: isGenerating) { showingOpenH3Setup = false }
        }
    }

    private var openH3StatusText: String {
        if OpenH3Setup.shared.isBusy {
            return String(localized: "準備中です（\(OpenH3Setup.shared.phase.apiText)）")
        }
        if VideoUpscaler.openH3.isAvailable {
            let size = ByteCountFormatter.string(fromByteCount: OpenH3Model.bytesOnDisk(), countStyle: .file)
            let check = OpenH3Model.verified ? "" : String(localized: "。取得元との照合はまだです（「管理…」から）")
            return String(localized: "使えます（精緻化LoRAは取り出し済み、\(size)）") + check
        }
        if OpenH3Model.upsamplerPresent && OpenH3Model.refinerPresent {
            return String(localized: "LightX2V Turbo v1.2のファイルがありません（「準備…」でダウンロードします）。")
        }
        return String(localized: "まだありません。約13GBのダウンロードと、精緻化LoRAの取り出しが必要です（終わると約5GB）。")
    }

    private var seedVR2StatusText: String {
        if SeedVR2Setup.shared.isBusy {
            return String(localized: "準備中です（\(SeedVR2Setup.shared.phase.apiText)）")
        }
        if VideoUpscaler.seedVR2.isAvailable {
            let size = ByteCountFormatter.string(fromByteCount: SeedVR2Model.bytesOnDisk(), countStyle: .file)
            let check = SeedVR2Model.verified ? "" : String(localized: "。取得元との照合はまだです（「管理…」から）")
            return String(localized: "使えます（\(size)）") + check
        }
        return String(localized: "まだありません。約4.7GBのダウンロードが必要です（変換は不要）。")
    }

    private var swiftVRStatusText: String {
        if SwiftVRSetup.shared.isBusy {
            return String(localized: "準備中です（\(SwiftVRSetup.shared.phase.apiText)）")
        }
        if VideoUpscaler.swiftVR.isAvailable {
            let size = ByteCountFormatter.string(fromByteCount: SwiftVRModel.bytesOnDisk(), countStyle: .file)
            let check = SwiftVRModel.verified ? "" : String(localized: "。取得元との照合はまだです（「管理…」から）")
            if SwiftVRModel.originalRemoved {
                return String(localized: "使えます（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")で変換済み、元の重みは削除済み、\(size)）") + check
            }
            return String(localized: "使えます（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")で変換済み、\(size)）") + check
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
