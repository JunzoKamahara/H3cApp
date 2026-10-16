# H3cApp site: facts and sources

The numbers and UI names the GitHub Pages site (`docs/`) relies on, with
where each came from. Update this file first when a new release changes any
of them, then the pages (see `website-maintenance.md`).

- Checked: 2026-10-14
- Version described: **v0.5.0** (latest release; published 2026-10-16).
  v0.5.0 added つなげて生成 (length menu: 1-15 s in one generation, つないで15秒,
  20/30/45/60/90 s, カスタム 1-90 s; 詳細設定 つなげて生成の1本の長さ 約7秒
  (175 frames, default) / 約5秒 (124); 22-frame overlap; joined video listed
  as 「つないだ動画（n本・x秒）」 with clips in chains/<name>/), 動画の続きを作る,
  音声に合わせる（リップシンク）, whole-run progress (全n本中k本目, 全体の残り/完了予定),
  and the LoRA folder (Application Support/h3c-app/LoRA). The site documents
  つなげて生成 (guide #chain, settings #chain) but not yet lip sync or 続きを作る
  as their own sections. Timing: ~105 s per output second at landscape medium
  on the 24 GB M5 (README). Earlier:
  v0.4.6 (2026-10-14) added the re-noise step rule for few-step LoRAs such as
  DMAD and, in モデル管理, a per-LoRA ステップの進め方 (自動 / Euler /
  re-noise（LCM方式）) and 音声シフト; 自動 follows the file's metadata only.
  It also reads LoRA headers in the background (no freeze while macOS asks
  for 書類-folder access). v0.4.7 moved those two controls onto their own line
  (0.4.6 pushed the row's menu to the sheet edge); generation is unchanged.
  Other screens are unchanged, so captions saying "v0.4.4の実画面" stay true.
- Tag commit: v0.5.0 `920203c` (v0.4.7 `3b5a145`, v0.4.6 `5b91b98`)
- Repository: renamed from `h3c-app` to `H3cApp` for the site URL
  `https://junzokamahara.github.io/H3cApp/` (decided 2026-10-10); the site
  links to `github.com/JunzoKamahara/H3cApp`. Old URLs redirect on GitHub.
- `main` at check time: the v0.5.0 tag plus site-only commits; nothing
  main-only is described.

## Languages

Japanese at `/H3cApp/` (main, x-default) and English at `/H3cApp/en/`
(added 2026-10-11). English UI names come from the v0.4.4
`en.lproj/Localizable.strings` (e.g. 動画を書き出す… = Export Video…,
高速（int8キャッシュ）= Fast (int8 cache), 詳細設定 = Advanced Settings,
モデル管理 = Model Manager, サイズを確認… = Check Size…). The English
pages use English-UI screenshots (`app-window-en.jpg`,
`app-composer-en.jpg`): the v0.4.4 app from the DMG launched with
`-AppleLanguages (en)`, the same scratch project and example 3 regenerated
(seed 7, same settings; 5:58 this run), taken 2026-10-11 while macOS was in
light appearance (the Japanese ones are dark). In English the count chip
reads “×1”. `og-image-en.jpg` is the English share image.

## Distribution

| item | value | source |
|---|---|---|
| DMG | `H3cApp-0.5.0.dmg`, 19,144,266 bytes (about 19.1 MB), sha256 c6d30216…3b022e7cad | local build, uploaded to v0.5.0 |
| Direct URL | https://github.com/JunzoKamahara/H3cApp/releases/download/v0.5.0/H3cApp-0.5.0.dmg | same |
| Site link target | https://github.com/JunzoKamahara/H3cApp/releases/latest (no version-specific link on the buttons) | |
| Signing | Developer ID, notarized and stapled; `spctl` reports "Notarized Developer ID" for the DMG and for the app inside it | checked 2026-10-16 (v0.5.0: `stapler validate`, `spctl` on the DMG and the app, CFBundleShortVersionString 0.5.0 / build 12, minos 15.0) |
| Install | open the DMG, drag H3cApp to Applications | |
| Minimum macOS | 15.0 (`LSMinimumSystemVersion` in the v0.4.4 Info.plist). Releases up to 0.4.3 said 13, which was wrong (engine uses macOS 15 MPSGraph SDPA, macOS 14 BF16) | `native/H3Spike/Packaging/Info.plist` at v0.4.4, release notes |
| CPU | Apple Silicon only (arm64 build) | |
| No Python / PyTorch / FFmpeg / Xcode CLT needed to use the DMG | README requirements (CLT only for building from source) | `README.ja.md` |

## Hardware and memory

- M5-class GPU (Metal 4 hardware tensor units): enables the
  "高速（int8キャッシュ）" compute mode and "高速モード（試験的）"
  (`h3_fast_attention_available()`: ccv linked and the GPU has neural
  matrix accelerators). The DMG links ccv.
- Other Apple Silicon: the app selects "省メモリ（SSDストリーミング）" by
  default (`defaultComputeMode`); "常駐" is available. Speed not measured
  here; the site says it differs and is not M5-equivalent.
- Memory: only the author's M5 / 24 GB was measured. 512x512 15 s fit in
  24 GB (swap 559-715 MiB across the 36-run grid, README). No minimum is
  claimed.
- The app warns about 常駐 when physical memory is below 32 GiB (int8-capable
  GPU) or 64 GiB (others) (`isLowMemoryForResident`).

## Models and disk

| item | value | source |
|---|---|---|
| Model | MiniMaxAI/MiniMax-H3 on Hugging Face | |
| FL2VA | 81 files, 144,051,182,625 bytes = 144.1 GB = 134.16 GiB | HF API tree, 2026-10-10, no pagination |
| Ref2VA | 81 files, 144,051,182,613 bytes | same |
| Both | about 288.1 GB = 268.32 GiB | |
| App's display | `formatGB` divides by 1,073,741,824 but prints "GB", so FL2VA shows about "134.2GB"; the idle text says "FL2VAだけで約130GB、Ref2VAも含めると約260GB" | `ModelDownloadWizardView.swift` |
| int8 cache | 19,279,769,664 bytes (about 19.3 GB, 17.96 GiB) per transformer; FL2VA and Ref2VA each need their own | `~/models/cache/*.cache` on the dev Mac |
| Cache location | `~/Library/Application Support/h3c-app/cache/<model id>/` (always internal; legacy path for the first-ever model) | `attentionCacheDirectory` |
| Default download folder | `~/Library/Application Support/h3c-app/MiniMax-H3`; "変更…" picks a folder and appends `/MiniMax-H3` | `defaultH3ModelDownloadPath`, wizard |
| Resume | a file whose size already matches is skipped; partial files continue with a Range request | `ModelDownloader.swift` |

Not used: the older note's "FL2VA 164GB".

## UI names (v0.4.4, Japanese)

- Wizard title "MiniMax-H3 モデルのダウンロード"; buttons "フォルダを追加…",
  "変更…", toggle "Ref2VA も含める（参照画像・動画機能。おおよそ倍のサイズになります）",
  "サイズを確認…" → "合計 …GB（…ファイル）。「ダウンロード開始」を押してください。" →
  "ダウンロード開始" → "中止" / "再開" → "ダウンロードが完了しました。" → "完了";
  "あとで" closes it. Shown at first launch when no model loads.
- Toolbar: project menu (folder icon, left), "モデル管理" (stack icon),
  status icon: "モデルを準備しています" (hourglass) / "準備完了" (checkmark) /
  "モデルが見つかりません" (warning). Model manager: "フォルダを追加…",
  "ダウンロードして追加…", LoRA "追加…".
- Composer: "文章から" / "画像から"; image modes "最初・最後の画像" /
  "参照画像・動画・音声"; "最初の画像", "最後の画像も指定…", "ファイルを選ぶ…",
  "取り除く"; chips 画面の形 (横長/正方形/縦長), 大きさ (小/中/大 + "2倍に拡大"),
  長さ (1-15 秒), 本数 (1,2,3,4,5,6,8,10,15,20); generate button ↑, Enter,
  ⌘Return; Shift+Enter new line. Default: 文章から, 正方形・中 512, 5 秒, 1 本,
  20 steps.
- 詳細設定 (⌘,): サイズ (カスタムサイズを使う, 半分のサイズで生成して2倍に拡大,
  estimate "このMacでの目安"), 生成ステップ数, 計算方式 (高速（int8キャッシュ）/
  常駐（大容量メモリ向け、キャッシュ不要）/ 省メモリ（SSDストリーミング）,
  "キャッシュを作成…"), 速度 (標準（高品質）/ 高速 / 最速), 高速モード（試験的）
  (shown only when available), ノイズ除去の再利用（reuse）, 使用する層数,
  ランダムさ (毎回変える / 固定する), 追加モデル（LoRA）.
- Missing cache message: "int8キャッシュが見つかりません。詳細設定の「計算方式」から
  作成するか、SSDストリーミングに切り替えてください".
- Result: "動画を書き出す…" (⌘S, starts in Downloads), "使用した設定…"
  ("フォームに戻す", "プリセットとして保存…", "設定をテキストでコピー",
  "同じシードを使う"), "参照に使う", "Finderで表示".
- Projects: "新規プロジェクト…", "プロジェクトとして保存…", "プロジェクトを開く…",
  "最近のプロジェクト", "プロジェクトの動画…" (rows: "表示", "設定を戻す",
  "参照に使う", "削除…" → "ゴミ箱に移動"), "プロジェクトを閉じる". Default
  folder `~/Movies/H3cApp/<name>`. Batch without a project: sheet
  "プロジェクトを選んで生成" with "新しいプロジェクト" / "既存のプロジェクト",
  "作成して生成" / "開いて生成", overwrite alert "上書きして生成".
- Without a project the result is a temp file deleted by the next
  generation (`deleteTemporaryPreview`).
- Sizes: square 256/512/768; landscape 448x256/672x384/960x544; portrait
  swapped; 2x = host high-quality resampling (vImage), e.g. landscape 大 x2
  = 1920x1088. Custom finished size, generated canvas <= 600,000 pixels.
- Lengths snap to 5 + 17k frames at 24 fps (5 s → 124 frames ≈ 5.17 s).
- Ref2VA: up to 12 references (engine); audio cannot be the only reference.
- LoRA (guide `#lora`): モデル管理 → tab "LoRA" → "追加…" (.safetensors);
  row shows format / "N/50ブロック" / rank or "MiniMax-H3用として読み込めません";
  per-row "強さ" (blank = 1) and "ステップ" (Turbo step count, auto-read from
  `\d+step` in the file name); up to 8 on (`maxStackedLoRAs`); "削除" only
  drops the list entry. 詳細設定 "追加モデル（LoRA）" toggles + "管理…".
  Turning a Turbo LoRA on sets steps to its count; off restores 20 only if
  untouched (`followTurboLoRASteps`); note turns red when steps differ; warns
  when more than one Turbo LoRA is on. LoRAs are recorded in a video's
  settings and in projects; unregistered ones are reported on open. The app
  can't tell FL2V vs Ref2V files apart; lightx2v/Minimax-h3-Turbo names them
  `fl2v`/`ref2v` and `4step`/`8step` (HF file list, 2026-10-11, Apache-2.0).
  Works with all three compute modes; formats per README "LoRA".

## Performance shown on the site

From the README grid (M5, 24 GB, square 512x512, T2V, 20 steps, int8
cache, prompt "A cat playing with a ball of yarn.", seed 7, one run each,
request to finished file, measured 2026-10-02/03 through the app API; app
version at the time not recorded, so not presented as a v0.4.4 measurement):

| length | 標準 (reuse 2) | 最速 + 高速モード (reuse 1, core reuse 4) |
|---|---|---|
| 5 s | 5:59 | 2:38 |
| 10 s | 15:56 | 5:39 |
| 15 s | 30:08 | 9:25 |

"3.2x" (15 s) compares those two settings on the same M5 only.

## Media

| file | what | made with |
|---|---|---|
| `docs/assets/video/ex1.mp4` | 256x256, 5 s, sound; getting-started prompt; 104 s (1:44) on M5 24 GB | H3cApp v0.4.4 (notarized app run from the DMG), API, seed 7, 20 steps, 標準, int8 cache, no project; re-encoded for the web |
| `docs/assets/video/ex2.mp4` | 512x512, 5 s, same prompt; 359 s (5:59) | same |
| `docs/assets/video/ex3.mp4` | landscape 中 x2 (672x384 → 1344x768), 5 s, guitar prompt; 369 s (6:09) | same |
| `docs/assets/images/app-window.jpg` | real v0.4.4 window (760x652 pt) right after ex3 finished via the API (panel collapsed, summary line shown) | `screencapture -l` of the v0.4.4 app run from the DMG |
| `docs/assets/images/app-composer.jpg` | real v0.4.4 window relaunched with a scratch project (panel expanded, empty preview), resized to 1600 px | same; the window frame default was enlarged for the shot and restored afterwards |
| `docs/assets/images/app-lora.jpg`, `app-lora-en.jpg` | モデル管理 → LoRA tab (sheet only, 1100x900 px, light), the lightx2v Turbo fl2v 4step v1.2 LoRA (unchecked, ステップの進め方 自動（Euler）) and DMAD full_critic (checked, 自動（re-noise）, 音声シフト placeholder 2) | `screencapture -l` of the signed v0.4.7 build, ja and `-AppleLanguages (en)`, taken 2026-10-14 while macOS was in light appearance (the old v0.4.4 shots were dark), cropped to the sheet and scaled from 1119 px to 1100 px wide |
| `docs/assets/images/app-advanced.jpg`, `app-advanced-en.jpg` | 詳細設定 sheet right after 既定に戻す (scrolled to the top, 1064x1268 px, light; the dark appearance could not be forced), no project open | `screencapture -l` of the dev build (AdvancedSettingsView unchanged since v0.4.4); the ja shot's estimate reflects a 1 s length, the en shot's 5 s |
| `docs/assets/images/og-image.jpg` | 1200x630 share image: app icon, title, and a frame of ex2 | composed with Pillow (Hiragino) |
| icons | from `native/H3Spike/Packaging/AppIcon.icns` | `iconutil`, `sips` |

Web encoding: H.264 High, CRF 24 (preset slow), yuv420p, AAC 128 kbps,
`+faststart`; PSNR against the app's output 43.2 dB (ex2) and 45.9 dB (ex3);
posters are the frame at 2.5 s. The three files total about 1.1 MB.

External: author's YouTube video https://youtu.be/PZrm470rGo0 ("走る少女２",
channel OSU-DS-kamahara; public per oEmbed 2026-10-10; content not viewed,
so it is only linked, not described as an H3cApp output). The note
screenshot (assets.st-note.com, older UI) is not used.

## Licenses

- App: MIT (`LICENSE`, copyright Salvatore Sanfilippo); based on antirez/h3.c.
  Third-party notices in `THIRD_PARTY_NOTICES.md` (ccv, BSD-3-Clause).
- Model: MiniMax H3 Community License
  (https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE, Q&A at
  docs/QA-about-License.md). The site does not judge commercial use. It
  calls the license "somewhat unusual" (per the author) and summarizes,
  from the license text read 2026-10-11: Applicable Territory excludes the
  EU, UK, Republic of Korea and USA (I.3, I.5, V.4); Acceptable Use Policy
  (Exhibit A); MiniMax H3 Works and Outputs may not improve other AI models
  (V.3); commercial terms - written authorization above USD 20M yearly
  revenue, "MiniMax H3" shown in the UI of commercial products (IV). Shown
  on index (#model-license warning), getting-started #models, faq #license,
  and every footer, in both languages.

## Open points for the editor

- Published 2026-10-10 at `https://junzokamahara.github.io/H3cApp/` (Pages
  from main `/docs`, repository renamed to H3cApp, About → Website set).
  The path is case-sensitive: `/h3capp/` and the old `/h3c-app/` return
  404, so always link `/H3cApp/`. canonical,
  og:url, og:image, robots.txt and sitemap.xml use it; recheck after Pages
  is enabled.
- Speed on non-M5 Apple Silicon and with less than 24 GB is not measured.
- Japanese prompts are not evaluated; the site says so.
- Time to build the int8 cache is not stated (the app's own text says
  "数十秒〜1分程度", not re-measured here).
- Adding Ref2VA later: the model manager's "ダウンロードして追加…" opens the
  wizard at the default download path and, on completion, *adds* a model
  entry (`addModel` does not dedupe paths), so downloading Ref2VA into the
  existing folder leaves two entries for the same folder. Both point at the
  same files (Ref2VA is read from `<model>/Ref2VA` at generation time), but
  each entry has its own int8 cache directory (by id). Described from the
  code; not exercised in the UI.
- Screenshots of the download wizard, model manager and advanced settings
  were not taken (the agent could not operate the UI); see
  `website-maintenance.md` for the list.
