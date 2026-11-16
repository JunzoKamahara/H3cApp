import SwiftUI

/// 高解像度化…: how to enlarge a project image (ImageUpscale.swift) - the
/// method, the factor with the size it makes, and for a redraw how far it
/// may depart from the picture.
struct ImageUpscaleSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let image: ProjectImage
    let close: () -> Void
    @State private var method: ImageUpscaleMethod = .redraw
    @State private var factor: UpscaleFactor = .two
    @State private var strength: RedrawStrength = .standard

    private var size: (width: Int, height: Int) {
        image.record.map { ($0.width, $0.height) } ?? (0, 0)
    }

    private var choices: [(factor: UpscaleFactor, width: Int, height: Int)] {
        ImageUpscale.choices(method, width: size.width, height: size.height)
    }

    private var chosen: (factor: UpscaleFactor, width: Int, height: Int)? {
        choices.first { $0.factor == factor }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("画像を高解像度化").font(.headline)
            Text("\(image.url.lastPathComponent)（\("\(size.width)×\(size.height)")）を大きくして、新しい画像としてプロジェクトに加えます。元の画像はそのまま残ります。")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)

            Picker("方法", selection: $method) {
                ForEach(ImageUpscaleMethod.available) { method in
                    Text(method == .redraw ? String(localized: "画像モデルで描き直す（おすすめ）") : method.label)
                        .tag(method)
                }
            }
            Text(method.detail)
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if !ImageUpscaleMethod.frameUpscalers.allSatisfy({ $0.method.isAvailable }) {
                Text("SwiftVR・SeedVR2は、モデル管理の「拡大モデル」でモデルを入れると選べます。")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if choices.isEmpty {
                Text("この画像は、この方法ではこれ以上大きくできません。")
                    .font(.callout).foregroundStyle(.orange)
            } else {
                Picker("大きさ", selection: $factor) {
                    ForEach(choices, id: \.factor) { choice in
                        Text("\(choice.factor.label)（\("\(choice.width)×\(choice.height)")）").tag(choice.factor)
                    }
                }
                if method == .redraw, let chosen, chosen.width * chosen.height < Int(Double(size.width * size.height) * chosen.factor.rawValue * chosen.factor.rawValue * 0.95) {
                    Text("描き直しは約8メガピクセルまでなので、この大きさになります。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            if method == .redraw {
                Picker("描き直しの強さ", selection: $strength) {
                    ForEach(RedrawStrength.allCases) { strength in
                        Text(strength.label).tag(strength)
                    }
                }
                .pickerStyle(.segmented)
                Text(strength.detail).font(.caption).foregroundStyle(.secondary)
                let prompt = viewModel.redrawPrompt(for: image)
                Text(prompt.isEmpty
                     ? String(localized: "プロンプト: なし（パネルの画像のプロンプトに内容を書いてください）")
                     : String(localized: "プロンプト: \(prompt)"))
                    .font(.caption)
                    .foregroundStyle(prompt.isEmpty ? .orange : .secondary)
                    .lineLimit(3)
                if let model = viewModel.library.activeImageModel {
                    Text("画像モデル: \(model.name)").font(.caption).foregroundStyle(.secondary)
                }
            }

            if let reason = viewModel.imageUpscaleBlockedReason(method) {
                Text(reason).font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Spacer()
                Button("キャンセル", role: .cancel, action: close)
                    .keyboardShortcut(.cancelAction)
                Button("高解像度化") {
                    if viewModel.upscaleImage(image, method: method, factor: factor, strength: strength) { close() }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(chosen == nil || viewModel.imageUpscaleBlockedReason(method) != nil
                          || (method == .redraw && viewModel.redrawPrompt(for: image).isEmpty))
            }
        }
        .padding(H3Spacing.xl)
        .frame(width: 480)
        .onChange(of: method) { _, _ in pickFactor() }
        .onAppear(perform: pickFactor)
    }

    /// 2倍 when the method offers it, else the largest it offers.
    private func pickFactor() {
        if choices.contains(where: { $0.factor == factor }) { return }
        factor = choices.first(where: { $0.factor == .two })?.factor ?? choices.last?.factor ?? .two
    }
}
