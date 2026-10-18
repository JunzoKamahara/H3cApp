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

/// What deleting a model's files would remove and free.
struct ModelRemovalPlan: Sendable {
    struct Target: Sendable, Identifiable {
        var id: String { path }
        let path: String
        let label: String
        let bytes: Int64
    }

    var targets: [Target] = []
    /// Space the deletion gives back.
    var freedBytes: Int64 = 0
    /// Data the deleted files share with files that stay (kept).
    var sharedBytes: Int64 = 0
    var sharedFiles = 0
    /// Folders that refuse deletion (not a model folder, too high up).
    var refused: [String] = []

    /// `others` are the models staying registered; when there are none,
    /// the app's shared download folders and caches no model uses go too.
    static func make(modelPath: String, id: UUID, others: [(path: String, id: UUID)]) -> ModelRemovalPlan {
        var plan = ModelRemovalPlan()
        let fm = FileManager.default
        func add(_ path: String, _ label: String) {
            guard fm.fileExists(atPath: path) else { return }
            plan.targets.append(Target(path: path, label: label, bytes: 0))
        }
        if isDeletableModelFolder(modelPath) {
            add(modelPath, String(localized: "モデルのフォルダ"))
        } else {
            plan.refused.append(modelPath)
        }
        let cache = modelCacheDirectory(path: modelPath, id: id)
        if cache.hasPrefix(h3AppSupportDirectory + "/cache/") {
            add(cache, String(localized: "このモデルのキャッシュ"))
        }
        if others.isEmpty {
            add(importBaseDirectory, String(localized: "ダウンロードした共通部品"))
            add(importDownloadsDirectory, String(localized: "ダウンロードしたGGUF"))
            let kept = Set(others.map { modelCacheDirectory(path: $0.path, id: $0.id) })
            let cacheRoot = h3AppSupportDirectory + "/cache"
            for name in ((try? fm.contentsOfDirectory(atPath: cacheRoot)) ?? []).sorted()
            where UUID(uuidString: name) != nil && cacheRoot + "/" + name != cache && !kept.contains(cacheRoot + "/" + name) {
                add(cacheRoot + "/" + name, String(localized: "使われていないキャッシュ"))
            }
        }

        // Every file, by clone family.
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

    /// Deletes the targets for good (not to the Trash: the point is the
    /// space). Returns the paths that could not be removed.
    func execute() -> [String] {
        targets.compactMap { target in
            (try? FileManager.default.removeItem(atPath: target.path)) == nil ? target.path : nil
        }
    }
}

// MARK: - The confirmation sheet

struct ModelRemoveView: View {
    let model: H3ModelEntry
    let others: [H3ModelEntry]
    /// Called with whether to delete the files; the caller unregisters
    /// the model first, then runs the plan.
    var onRemove: (_ plan: ModelRemovalPlan?) -> Void
    var onCancel: () -> Void

    @Environment(\.colorScheme) private var colorScheme
    @State private var deleteFiles = false
    @State private var plan: ModelRemovalPlan?

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
            Toggle("モデルファイルも削除する", isOn: $deleteFiles)
            if deleteFiles {
                filesSection
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("キャンセル") { onCancel() }.keyboardShortcut(.cancelAction)
                Button(deleteFiles ? String(localized: "外してファイルを削除") : String(localized: "外す"),
                       role: .destructive) {
                    onRemove(deleteFiles ? plan : nil)
                }
                .disabled(deleteFiles && plan == nil)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 560, height: deleteFiles ? 520 : 220)
        .background(palette.canvas)
        .task(id: deleteFiles) {
            guard deleteFiles, plan == nil else { return }
            let path = model.path, id = model.id
            let others = self.others.map { (path: $0.path, id: $0.id) }
            plan = await Task.detached { ModelRemovalPlan.make(modelPath: path, id: id, others: others) }.value
        }
    }

    @ViewBuilder private var filesSection: some View {
        if let plan {
            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    Text("削除するもの").font(.callout.weight(.semibold))
                    ForEach(plan.targets) { target in
                        VStack(alignment: .leading, spacing: 1) {
                            Text(verbatim: "\(target.label)・\(size(target.bytes))").font(.caption)
                            Text(target.path).font(.caption2).foregroundStyle(palette.textSecondary)
                                .lineLimit(1).truncationMode(.middle)
                        }
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
                    Text("解放されるディスク容量：約\(size(plan.freedBytes))")
                        .font(.callout.weight(.semibold))
                    if plan.sharedBytes > 0 {
                        Text("ほかのモデルや、取り込みのもとにしたファイルと共有している部分（\(plan.sharedFiles)個のファイル、約\(size(plan.sharedBytes))）は、そちらで使われているので残ります。GGUFの取り込みは、同じディスク上のファイルを複製して容量を共有するためです。")
                            .font(.caption).foregroundStyle(palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Text("ゴミ箱には入らず、すぐに削除されます。元に戻せません。")
                        .font(.caption).foregroundStyle(palette.errorColor)
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
}
