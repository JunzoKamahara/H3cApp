import Darwin
import Foundation
import SwiftUI

// Removing a model from Model Management, optionally with its files. A
// GGUF import clones files on the same disk (APFS), so a model's files can
// share their data with other models, the downloads, or the files they
// were imported from. APFS tracks those clones: each file has a clone id
// shared by every file holding the same data, and a count of them. A
// family whose every member is being deleted frees its data; one with a
// member left elsewhere frees only what the deleted members hold alone.

/// APFS clone facts for one file (getattrlist's extended common attributes).
struct CloneStat {
    var size: Int64
    /// Bytes only this file holds (not shared with a clone).
    var privateSize: Int64
    /// Shared by every file holding the same data; 0 when not tracked.
    var cloneID: UInt64
    /// How many files hold that data.
    var references: UInt32

    static func of(_ path: String) -> CloneStat? {
        let size = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.int64Value ?? 0
        var list = attrlist()
        list.bitmapcount = u_short(ATTR_BIT_MAP_COUNT)
        list.forkattr = attrgroup_t(ATTR_CMNEXT_PRIVATESIZE | ATTR_CMNEXT_CLONEID | ATTR_CMNEXT_CLONE_REFCNT)
        var buffer = [UInt8](repeating: 0, count: 64)
        let result = buffer.withUnsafeMutableBytes { raw in
            getattrlist(path, &list, raw.baseAddress, raw.count, UInt32(FSOPT_ATTR_CMN_EXTENDED | FSOPT_NOFOLLOW))
        }
        guard result == 0 else { return CloneStat(size: size, privateSize: size, cloneID: 0, references: 1) }
        // A u_int32 length, then the attributes in bit order.
        return buffer.withUnsafeBytes { raw in
            CloneStat(size: size,
                      privateSize: raw.loadUnaligned(fromByteOffset: 4, as: Int64.self),
                      cloneID: raw.loadUnaligned(fromByteOffset: 12, as: UInt64.self),
                      references: raw.loadUnaligned(fromByteOffset: 20, as: UInt32.self))
        }
    }
}

/// What deleting a model's files would remove and free (once the Trash
/// they go to is emptied).
struct ModelRemovalPlan: Sendable {
    struct Target: Sendable, Identifiable {
        var id: String { path }
        let path: String
        let label: String
        let bytes: Int64
    }

    /// Moved to the Trash (only when the files are deleted too).
    var targets: [Target] = []
    /// Deleted outright, even when only unregistering: the int8 caches and
    /// AdaLN curves built for generating (they can be rebuilt).
    var caches: [Target] = []
    var includesFiles = false
    /// Space emptying the Trash gives back.
    var freedBytes: Int64 = 0
    /// Space deleting the caches gives back right away.
    var cacheBytes: Int64 { caches.reduce(0) { $0 + $1.bytes } }
    /// Data the trashed files share with files that stay (kept).
    var sharedBytes: Int64 = 0
    var sharedFiles = 0
    /// Folders that refuse deletion (not a model folder, too high up).
    var refused: [String] = []

    /// `others` are the models staying registered; when there are none and
    /// the files go too, so do the app's shared download folders and the
    /// caches no model uses. `requireWeights` (the API) also refuses a
    /// model folder without weights (see holdsModelWeights).
    static func make(modelPath: String, id: UUID, others: [(path: String, id: UUID)],
                     includeFiles: Bool, requireWeights: Bool = false) -> ModelRemovalPlan {
        var plan = ModelRemovalPlan()
        plan.includesFiles = includeFiles
        let fm = FileManager.default
        func folder(_ path: String, _ label: String) -> Target? {
            fm.fileExists(atPath: path) ? Target(path: path, label: label, bytes: folderBytes(path)) : nil
        }

        let cacheRoot = h3AppSupportDirectory + "/cache"
        let cache = modelCacheDirectory(path: modelPath, id: id)
        if cache.hasPrefix(cacheRoot + "/"), let target = folder(cache, String(localized: "このモデルのキャッシュ")) {
            plan.caches.append(target)
        }
        guard includeFiles else { return plan }

        func add(_ path: String, _ label: String) {
            if fm.fileExists(atPath: path) { plan.targets.append(Target(path: path, label: label, bytes: 0)) }
        }
        if isDeletableModelFolder(modelPath) && (!requireWeights || holdsModelWeights(modelPath)) {
            add(modelPath, String(localized: "モデルのフォルダ"))
        } else {
            plan.refused.append(modelPath)
        }
        if others.isEmpty {
            add(importBaseDirectory, String(localized: "ダウンロードした共通部品"))
            add(importDownloadsDirectory, String(localized: "ダウンロードしたGGUF"))
            for name in ((try? fm.contentsOfDirectory(atPath: cacheRoot)) ?? []).sorted()
            where UUID(uuidString: name) != nil && cacheRoot + "/" + name != cache {
                if let target = folder(cacheRoot + "/" + name, String(localized: "使われていないキャッシュ")) {
                    plan.caches.append(target)
                }
            }
        }

        // Every trashed file, by clone family.
        var families: [UInt64: [CloneStat]] = [:]
        for (index, target) in plan.targets.enumerated() {
            var bytes: Int64 = 0
            for file in regularFiles(under: target.path) {
                guard let stat = CloneStat.of(file) else { continue }
                bytes += stat.size
                if stat.cloneID == 0 || stat.references <= 1 {
                    plan.freedBytes += stat.size
                } else {
                    families[stat.cloneID, default: []].append(stat)
                }
            }
            plan.targets[index] = Target(path: target.path, label: target.label, bytes: bytes)
        }
        for members in families.values {
            let shared = members.map { max($0.size - $0.privateSize, 0) }.max() ?? 0
            plan.freedBytes += members.reduce(0) { $0 + $1.privateSize }
            if members.count >= Int(members[0].references) {
                plan.freedBytes += shared
            } else {
                plan.sharedBytes += shared
                plan.sharedFiles += members.count
            }
        }
        return plan
    }

    private static func folderBytes(_ path: String) -> Int64 {
        regularFiles(under: path).reduce(0) { total, file in
            total + (((try? FileManager.default.attributesOfItem(atPath: file))?[.size] as? NSNumber)?.int64Value ?? 0)
        }
    }

    /// Only a folder that looks like a model (FL2VA/Ref2VA or an import
    /// record) and sits well below the root and the home folder.
    static func isDeletableModelFolder(_ path: String) -> Bool {
        let standard = (path as NSString).standardizingPath
        let fm = FileManager.default
        guard standard.split(separator: "/").count >= 3, standard != NSHomeDirectory(),
              !NSHomeDirectory().hasPrefix(standard + "/") else { return false }
        return fm.fileExists(atPath: standard + "/FL2VA") || fm.fileExists(atPath: standard + "/Ref2VA") ||
            fm.fileExists(atPath: standard + "/" + ModelImportManifest.fileName)
    }

    /// Whether FL2VA/transformer holds a weights file (.safetensors or
    /// .gguf), as every downloaded or imported model does. The API asks for
    /// this too: it registers any folder holding an FL2VA entry and can
    /// make one (a project named FL2VA), so the names above would let a
    /// request send any folder to the Trash. Only files count, since the
    /// API can also make folders named like weights.
    static func holdsModelWeights(_ path: String) -> Bool {
        let transformer = (path as NSString).standardizingPath + "/FL2VA/transformer"
        let fm = FileManager.default
        return ((try? fm.contentsOfDirectory(atPath: transformer)) ?? []).contains { name in
            var isDirectory: ObjCBool = false
            return (name.hasSuffix(".safetensors") || name.hasSuffix(".gguf")) &&
                fm.fileExists(atPath: transformer + "/" + name, isDirectory: &isDirectory) && !isDirectory.boolValue
        }
    }

    private static func regularFiles(under path: String) -> [String] {
        guard let enumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: path), includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
            options: []) else { return [] }
        var files: [String] = []
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if values?.isRegularFile == true && values?.isSymbolicLink != true { files.append(url.path) }
        }
        return files
    }

    /// Deletes the caches and moves the other targets to the Trash (their
    /// space comes back when it is emptied; until then they can be put
    /// back). Returns the paths that could not be removed.
    func execute() -> [String] {
        let fm = FileManager.default
        let failed = caches.compactMap { (try? fm.removeItem(atPath: $0.path)) == nil ? $0.path : nil } +
            targets.compactMap { target in
                (try? fm.trashItem(at: URL(fileURLWithPath: target.path), resultingItemURL: nil)) == nil
                    ? target.path : nil
            }
        // The shared parts live one level down (ImportBase/MiniMax-H3); an
        // emptied ImportBase goes too.
        let importBaseParent = (importBaseDirectory as NSString).deletingLastPathComponent
        if ((try? fm.contentsOfDirectory(atPath: importBaseParent)) ?? ["?"]).allSatisfy({ $0 == ".DS_Store" }) {
            try? fm.removeItem(atPath: importBaseParent)
        }
        return failed
    }
}

// MARK: - The confirmation sheet

struct ModelRemoveView: View {
    let model: H3ModelEntry
    let others: [H3ModelEntry]
    /// Called with the plan; the caller unregisters the model first (and
    /// moves off it if in use), then runs the plan.
    var onRemove: (_ plan: ModelRemovalPlan) -> Void
    var onCancel: () -> Void

    @Environment(\.colorScheme) private var colorScheme
    @State private var deleteFiles = false
    @State private var plan: ModelRemovalPlan?
    @State private var confirmingFiles = false

    private var palette: H3Palette { H3Palette(colorScheme) }
    private func size(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("「\(model.name)」を削除").font(.system(size: 16, weight: .semibold))
            Text("モデル管理から外します。ファイルを残しておけば、あとで「フォルダを追加…」で登録し直せます。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if let plan, plan.cacheBytes > 0 {
                Text("生成に使うキャッシュ（int8キャッシュなど、約\(size(plan.cacheBytes))）は削除します。ゴミ箱には入れず、すぐに空きます（もう一度使うときは作り直せます）。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Toggle("モデルファイルも削除する", isOn: $deleteFiles)
            if deleteFiles {
                filesSection
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("キャンセル") { onCancel() }.keyboardShortcut(.cancelAction)
                Button(deleteFiles ? String(localized: "外してゴミ箱に移す") : String(localized: "外す"),
                       role: .destructive) {
                    // Deleting the files asks once more.
                    if deleteFiles { confirmingFiles = true } else if let plan { onRemove(plan) }
                }
                // A sheet's destructive button is not red by itself.
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .disabled(plan == nil)
            }
        }
        .alert("モデルファイルを本当に削除してもいいですか？", isPresented: $confirmingFiles) {
            Button("削除する", role: .destructive) { if let plan { onRemove(plan) } }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text(confirmationText)
        }
        .padding(H3Spacing.lg)
        .frame(width: 560, height: deleteFiles ? 560 : 260)
        .background(palette.canvas)
        .task(id: deleteFiles) {
            plan = nil
            let path = model.path, id = model.id, files = deleteFiles
            let others = self.others.map { (path: $0.path, id: $0.id) }
            plan = await Task.detached {
                ModelRemovalPlan.make(modelPath: path, id: id, others: others, includeFiles: files)
            }.value
        }
    }

    private var confirmationText: String {
        let folders = plan?.targets.map(\.path).joined(separator: "、") ?? ""
        return String(localized: "次のフォルダをゴミ箱に移します：\(folders)")
    }

    @ViewBuilder private var filesSection: some View {
        if let plan {
            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    Text("ゴミ箱に移すもの").font(.callout.weight(.semibold))
                    ForEach(plan.targets) { target in targetRow(target) }
                    if !plan.caches.isEmpty {
                        Text("すぐに削除するキャッシュ").font(.callout.weight(.semibold))
                        ForEach(plan.caches) { target in targetRow(target) }
                    }
                    if others.isEmpty {
                        Text("最後のモデルなので、ダウンロードした共通部品・GGUFと、使われていないキャッシュも削除します。")
                            .font(.caption).foregroundStyle(palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    ForEach(plan.refused, id: \.self) { path in
                        Text("このフォルダはモデルのフォルダに見えないので削除しません：\(path)")
                            .font(.caption).foregroundStyle(palette.errorColor)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Divider()
                    // Under a megabyte is just configs and the like: say so
                    // rather than show "約529 バイト".
                    Text(plan.freedBytes < 1_000_000
                         ? String(localized: "ゴミ箱を空にしても、ほかと共有しているためほとんど空きません")
                         : String(localized: "ゴミ箱を空にすると解放される容量：約\(size(plan.freedBytes))"))
                        .font(.callout.weight(.semibold))
                    if plan.cacheBytes > 0 {
                        Text("キャッシュの削除ですぐに空く容量：約\(size(plan.cacheBytes))")
                            .font(.caption)
                    }
                    if plan.sharedBytes > 0 {
                        Text("ほかのモデルや、取り込みのもとにしたファイルと共有している部分（\(plan.sharedFiles)個のファイル、約\(size(plan.sharedBytes))）は、そちらで使われているので残ります。GGUFの取り込みは、同じディスク上のファイルを複製して容量を共有するためです。")
                            .font(.caption).foregroundStyle(palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Text("ファイルはゴミ箱に移します。ゴミ箱を空にするまで容量は空きませんが、それまではゴミ箱から戻せます。")
                        .font(.caption).foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            HStack {
                ProgressView().controlSize(.small)
                Text("削除するファイルと、解放される容量を調べています…")
                    .font(.caption).foregroundStyle(palette.textSecondary)
            }
        }
    }

    private func targetRow(_ target: ModelRemovalPlan.Target) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(verbatim: "\(target.label)\(itemSeparator)\(size(target.bytes))").font(.caption)
            Text(target.path).font(.caption2).foregroundStyle(palette.textSecondary)
                .lineLimit(1).truncationMode(.middle)
        }
    }
}
