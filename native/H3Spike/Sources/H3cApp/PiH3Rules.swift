import Foundation

/// What a pi-H3 adapter among the enabled LoRAs rules out, said before the
/// engine is asked - it refuses too ("a pi-H3 adapter samples in 2 or 4
/// steps"), but only once the generation has begun.
enum PiH3Rules {
    /// The step counts its schedule exists for (h3_pi_h3_schedule_build).
    static let steps = [2, 4]

    /// The reason the form's next generation cannot run with the adapter,
    /// for the line above the generate button and POST /api/generate's 409.
    /// adapterSteps is the count the adapter was distilled for - the entry's
    /// 推奨ステップ数, which its file name gives ("4step") and モデル管理 can
    /// change; the file itself does not record it. The engine would run a
    /// 2-step adapter on the 4-step schedule without a word.
    static func problem(adapterName: String, steps: Int, adapterSteps: Int? = nil) -> String? {
        if let adapterSteps, Self.steps.contains(adapterSteps), steps != adapterSteps {
            return String(localized: "追加モデル「\(adapterName)」（pi-H3）は\(adapterSteps)ステップ用です。詳細設定の「生成ステップ数」を\(adapterSteps)にしてください")
        }
        if !Self.steps.contains(steps) {
            return String(localized: "追加モデル「\(adapterName)」（pi-H3）は2または4ステップで生成します。詳細設定の「生成ステップ数」を2か4にしてください")
        }
        return nil
    }
}
