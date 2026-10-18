import Foundation
import H3Engine
import SwiftUI

// Model Management's "詳細": what a registered model folder holds, read
// from the files themselves - each part of FL2VA and Ref2VA (format, size,
// layers, whether the vision tower is there), the caches built for it, how
// it was imported, and so what it can be used for.

/// What a model folder holds. Built off the main thread (it reads the
/// safetensors and GGUF headers).
struct ModelDetail: Sendable {
    enum Status: Sendable { case ok, info, warning, missing }

    struct Row: Identifiable, Sendable {
        let id = UUID()
        let title: String
        let status: Status
        let lines: [String]
    }

    struct Section: Identifiable, Sendable {
        let id = UUID()
        let title: String
        let note: String?
        let rows: [Row]
    }

    var sections: [Section] = []
    var totalBytes: Int64 = 0

    static func inspect(path: String, id: UUID) -> ModelDetail {
        var detail = ModelDetail()
        var features: [Row] = []
        let fl2va = inspectPartition(path + "/FL2VA", required: true)
        let ref2va = inspectPartition(path + "/Ref2VA", required: false)
        features.append(Row(title: String(localized: "文章から・最初と最後の画像から"),
                            status: fl2va.usable ? .ok : .missing,
                            lines: [fl2va.usable ? String(localized: "使えます（FL2VA）")
                                                 : String(localized: "使えません（FL2VAの部品が足りません）")]))
        if fl2va.usable && !fl2va.vision {
            features.append(Row(title: String(localized: "画像を使う生成"), status: .warning,
                                lines: [String(localized: "画像の部分がないので、最初の画像・最後の画像は使えません")]))
        }
        features.append(Row(title: String(localized: "参照画像・動画・音声から"),
                            status: ref2va.usable && ref2va.vision ? .ok : .info,
                            lines: [ref2va.usable && ref2va.vision ? String(localized: "使えます（Ref2VA）")
                                    : ref2va.present ? String(localized: "使えません（Ref2VAの部品が足りません）")
                                    : String(localized: "使えません（Ref2VAがありません）")]))

        let cache = modelCacheDirectory(path: path, id: id)
        let cacheRows = [
            ("dit_int8_v2.cache", String(localized: "int8キャッシュ（FL2VA）")),
            ("dit_int8_v2_ref2va.cache", String(localized: "int8キャッシュ（Ref2VA）")),
            ("adaln_fl2va.curve", String(localized: "条件の計算の近似（FL2VA）")),
            ("adaln_ref2va.curve", String(localized: "条件の計算の近似（Ref2VA）")),
        ].map { name, title -> Row in
            let size = fileSize(cache + "/" + name)
            let inFile = (name == "adaln_fl2va.curve" && fl2va.curveInFile) ||
                (name == "adaln_ref2va.curve" && ref2va.curveInFile)
            if inFile && size == 0 {
                return Row(title: title, status: .ok, lines: [String(localized: "不要（DiTのファイルに含まれています）")])
            }
            return Row(title: title, status: size > 0 ? .ok : .info,
                       lines: [size > 0 ? bytes(size) : String(localized: "まだ作られていません")])
        }
        let int8Ready = cacheRows.first?.status == .ok
        features.append(Row(title: String(localized: "高速モード（M5のint8キャッシュ）"),
                            status: int8Ready ? .ok : .info,
                            lines: [int8Ready ? String(localized: "キャッシュがあるので使えます")
                                              : String(localized: "詳細設定の「キャッシュを作成…」で作ると使えます（M5のみ）")]))

        detail.sections.append(Section(title: String(localized: "このモデルでできること"), note: nil, rows: features))
        detail.sections.append(Section(title: "FL2VA", note: String(localized: "文章から・最初と最後の画像からの生成に使います"),
                                       rows: fl2va.rows))
        detail.sections.append(Section(title: "Ref2VA", note: String(localized: "参照画像・動画・音声を使う生成に使います"),
                                       rows: ref2va.rows))
        detail.sections.append(Section(title: String(localized: "キャッシュ"), note: cache, rows: cacheRows))

        if let manifest = ModelImportManifest.load(modelPath: path) {
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            var rows = [Row(title: String(localized: "取り込み"), status: .info,
                            lines: [formatter.string(from: manifest.importedAt),
                                    String(localized: "トークナイザー・VAE・設定の複製元：\(baseName(manifest.baseModel))")])]
            for part in manifest.parts {
                let role: String
                switch part.role {
                case "dit": role = String(localized: "DiT（FL2VA）")
                case "ref2va": role = String(localized: "DiT（Ref2VA）")
                case "textEncoder": role = String(localized: "文章の部分")
                default: role = String(localized: "画像の部分")
                }
                rows.append(Row(title: role, status: .info,
                                lines: ["\(part.source)・\(part.type)・\(bytes(Int64(part.bytes)))"]))
            }
            detail.sections.append(Section(title: String(localized: "GGUFからの取り込み"), note: nil, rows: rows))
        }

        detail.totalBytes = folderSize(path)
        return detail
    }

    /// The import records the base's name as shown then, in the language
    /// of that moment; the names the app made itself are shown in today's.
    private static func baseName(_ recorded: String) -> String {
        let shared = String(localized: "ダウンロードした共通部品")
        let official = String(localized: "公式モデルの共通部品（ダウンロード済み）")
        let known = ["ダウンロードした共通部品": shared, "Downloaded shared parts": shared,
                     "公式モデルの共通部品（ダウンロード済み）": official,
                     "Official model's shared parts (downloaded)": official]
        return known[recorded] ?? recorded
    }

    // MARK: One partition (FL2VA or Ref2VA)

    private struct Partition {
        var rows: [Row] = []
        var present = false
        var usable = false
        var vision = false
        /// The DiT carries its own curve-form AdaLN (no curve file needed).
        var curveInFile = false
    }

    private static func inspectPartition(_ dir: String, required: Bool) -> Partition {
        var part = Partition()
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir) else {
            part.rows = [Row(title: String(localized: "フォルダ"), status: required ? .missing : .info,
                             lines: [String(localized: "ありません")])]
            return part
        }
        part.present = true
        var dit = false, text = false

        // DiT
        let transformer = dir + "/transformer"
        if let gguf = files(transformer, ".gguf").first, let info = try? H3GGUFInfo.inspect(path: gguf) {
            dit = true
            part.curveInFile = info.curveAdaLN
            part.rows.append(Row(title: String(localized: "DiT"), status: .ok, lines: [
                "GGUF・\(info.mainType)・\(bytes(Int64(info.fileBytes)))",
                info.curveAdaLN ? String(localized: "AdaLNは曲線形（近似を含む）") : String(localized: "AdaLNは密な形"),
            ]))
        } else if let st = SafetensorsSummary(dir: transformer) {
            dit = true
            part.curveInFile = st.names.contains("adaln_t_table")
            part.rows.append(Row(title: String(localized: "DiT"), status: .ok, lines: [
                st.formatLine,
                st.names.contains("adaln_t_table") ? String(localized: "AdaLNは曲線形（近似を含む）")
                                                   : String(localized: "AdaLNは密な形（公式）"),
            ]))
        } else {
            part.rows.append(Row(title: String(localized: "DiT"), status: .missing, lines: [String(localized: "ありません")]))
        }

        // Text encoder and vision tower
        let encoder = dir + "/text_encoder"
        var visionLine: String?
        if let gguf = files(encoder, ".gguf").first(where: { !$0.hasSuffix("/mmproj.gguf") }),
           let info = try? H3GGUFInfo.inspect(path: gguf) {
            text = info.textLayers >= 50
            part.rows.append(Row(title: String(localized: "文章の部分"), status: text ? .ok : .warning, lines: [
                "GGUF・\(info.mainType)・\(bytes(Int64(info.fileBytes)))",
                layerLine(info.textLayers),
            ]))
            if info.hasVision { visionLine = String(localized: "文章の部分のファイルに含まれています") }
        } else if let st = SafetensorsSummary(dir: encoder) {
            let layers = st.layerCount(prefix: "model.language_model.layers.")
            text = layers >= 50
            part.rows.append(Row(title: String(localized: "文章の部分"), status: text ? .ok : .warning,
                                 lines: [st.formatLine, layerLine(layers)]))
            if st.names.contains(where: { $0.hasPrefix("model.visual.") }) {
                visionLine = String(localized: "文章の部分のファイルに含まれています")
            }
        } else {
            part.rows.append(Row(title: String(localized: "文章の部分"), status: .missing,
                                 lines: [String(localized: "ありません")]))
        }
        if visionLine == nil, let mmproj = files(encoder, ".gguf").first(where: { $0.hasSuffix("/mmproj.gguf") }),
           let info = try? H3GGUFInfo.inspect(path: mmproj) {
            visionLine = "mmproj.gguf・GGUF・\(info.mainType)・\(bytes(Int64(info.fileBytes)))"
        }
        if visionLine == nil, fm.fileExists(atPath: encoder + "/vision.safetensors") {
            visionLine = "vision.safetensors・\(bytes(fileSize(encoder + "/vision.safetensors")))"
        }
        part.vision = visionLine != nil
        part.rows.append(Row(title: String(localized: "画像の部分"), status: part.vision ? .ok : .warning,
                             lines: [visionLine ?? String(localized: "ありません（画像を使う生成はできません）")]))

        // VAEs
        if let st = SafetensorsSummary(dir: dir + "/video_vae/source") {
            let kind = st.mainType == "F32" ? String(localized: "公式と同じ形") : String(localized: "公式より小さい形（unsloth・ComfyUI系）")
            part.rows.append(Row(title: String(localized: "映像のVAE"), status: .ok, lines: [st.formatLine, kind]))
        } else {
            part.rows.append(Row(title: String(localized: "映像のVAE"), status: .missing, lines: [String(localized: "ありません")]))
        }
        if let st = SafetensorsSummary(dir: dir + "/audio_vae") {
            let kind = st.names.contains { $0.hasSuffix(".weight_g") }
                ? String(localized: "公式と同じ形") : String(localized: "重み正規化を適用済みの形（unsloth・ComfyUI系）")
            part.rows.append(Row(title: String(localized: "音声のVAE"), status: .ok, lines: [st.formatLine, kind]))
        } else {
            part.rows.append(Row(title: String(localized: "音声のVAE"), status: .missing, lines: [String(localized: "ありません")]))
        }

        // Tokenizer and configs
        let needed = ["tokenizer/tokenizer.json", "transformer/config.json", "video_vae/config.json",
                      "audio_vae/config.json"]
        let missing = needed.filter { !fm.fileExists(atPath: dir + "/" + $0) }
        part.rows.append(Row(title: String(localized: "トークナイザー・設定"), status: missing.isEmpty ? .ok : .missing,
                             lines: [missing.isEmpty ? String(localized: "そろっています")
                                                     : String(localized: "足りません：\(missing.joined(separator: "、"))")]))

        part.usable = dit && text && missing.isEmpty &&
            !part.rows.contains { $0.status == .missing }
        return part
    }

    private static func layerLine(_ layers: Int) -> String {
        if layers == 50 { return String(localized: "50層（エンジンが使う分だけ）") }
        if layers > 50 { return String(localized: "\(layers)層（エンジンが使うのは50層）") }
        return String(localized: "\(layers)層（50層必要なので足りません）")
    }

    // MARK: Files

    static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }

    private static func files(_ dir: String, _ suffix: String) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.hasSuffix(suffix) }.sorted().map { dir + "/" + $0 }
    }

    static func fileSize(_ path: String) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.int64Value ?? 0
    }

    /// The files' sizes added up (clones on the same disk share their
    /// space, so this can be more than the folder really takes).
    private static func folderSize(_ path: String) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: path), includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true { total += Int64(values?.fileSize ?? 0) }
        }
        return total
    }
}

/// The safetensors files in a folder: how many, how big, which dtype holds
/// most of the bytes, and the tensor names.
private struct SafetensorsSummary {
    var files = 0
    var bytes: Int64 = 0
    var dtypeBytes: [String: Int64] = [:]
    var names: Set<String> = []

    init?(dir: String) {
        let paths = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }.map { dir + "/" + $0 }
        for path in paths {
            files += 1
            bytes += ModelDetail.fileSize(path)
            guard let header = Self.header(path) else { continue }
            for (name, value) in header where name != "__metadata__" {
                names.insert(name)
                guard let tensor = value as? [String: Any], let dtype = tensor["dtype"] as? String,
                      let offsets = tensor["data_offsets"] as? [NSNumber], offsets.count == 2 else { continue }
                dtypeBytes[dtype, default: 0] += offsets[1].int64Value - offsets[0].int64Value
            }
        }
        if files == 0 { return nil }
    }

    var mainType: String { dtypeBytes.max { $0.value < $1.value }?.key ?? "?" }

    var formatLine: String {
        files == 1 ? "safetensors・\(mainType)・\(ModelDetail.bytes(bytes))"
                   : String(localized: "safetensors・\(mainType)・\(files)ファイル・\(ModelDetail.bytes(bytes))")
    }

    /// Layers under a name prefix ("…layers.N.…"), from the highest index.
    func layerCount(prefix: String) -> Int {
        var highest = -1
        for name in names where name.hasPrefix(prefix) {
            let rest = name.dropFirst(prefix.count)
            if let index = Int(rest.prefix { $0.isNumber }) { highest = max(highest, index) }
        }
        return highest + 1
    }

    private static func header(_ path: String) -> [String: Any]? {
        guard let handle = FileHandle(forReadingAtPath: path) else { return nil }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else { return nil }
        let length = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard length > 0, length < 256 << 20, let json = try? handle.read(upToCount: Int(length)) else { return nil }
        return (try? JSONSerialization.jsonObject(with: json)) as? [String: Any]
    }
}

// MARK: - The sheet

struct ModelDetailView: View {
    let model: H3ModelEntry
    var onClose: () -> Void

    @Environment(\.colorScheme) private var colorScheme
    @State private var detail: ModelDetail?

    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.name).font(.system(size: 16, weight: .semibold))
                    Text(model.path).font(.caption).foregroundStyle(palette.textSecondary)
                        .lineLimit(2).truncationMode(.middle).textSelection(.enabled)
                    if let detail {
                        Text("ファイルの合計：\(ModelDetail.bytes(detail.totalBytes))（同じディスク上の複製は容量を共有しています）")
                            .font(.caption).foregroundStyle(palette.textSecondary)
                    }
                }
                Spacer()
                Button("Finderで表示") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)])
                }
            }
            Divider()
            if let detail {
                ScrollView {
                    VStack(alignment: .leading, spacing: H3Spacing.md) {
                        ForEach(detail.sections) { section in sectionView(section) }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("ファイルを調べています…").font(.caption).foregroundStyle(palette.textSecondary)
                }
                Spacer(minLength: 0)
            }
            HStack {
                Spacer()
                Button("閉じる") { onClose() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 620, height: 640)
        .background(palette.canvas)
        .task {
            let path = model.path, id = model.id
            detail = await Task.detached { ModelDetail.inspect(path: path, id: id) }.value
        }
    }

    private func sectionView(_ section: ModelDetail.Section) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(section.title).font(.callout.weight(.semibold))
            if let note = section.note {
                Text(note).font(.caption2).foregroundStyle(palette.textSecondary)
                    .lineLimit(1).truncationMode(.middle)
            }
            ForEach(section.rows) { row in
                HStack(alignment: .top, spacing: H3Spacing.sm) {
                    Image(systemName: icon(row.status)).foregroundStyle(color(row.status)).frame(width: 18)
                    Text(row.title).font(.caption).frame(width: 150, alignment: .leading)
                    VStack(alignment: .leading, spacing: 1) {
                        ForEach(row.lines, id: \.self) { line in
                            Text(line).font(.caption).foregroundStyle(palette.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .textSelection(.enabled)
                        }
                    }
                }
            }
        }
    }

    private func icon(_ status: ModelDetail.Status) -> String {
        switch status {
        case .ok: return "checkmark.circle.fill"
        case .info: return "info.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .missing: return "xmark.circle.fill"
        }
    }

    private func color(_ status: ModelDetail.Status) -> Color {
        switch status {
        case .ok: return .green
        case .info: return palette.accent
        case .warning: return .orange
        case .missing: return palette.errorColor
        }
    }
}
