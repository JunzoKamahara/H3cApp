# 静止画生成（MiniMax H3 Image）とギャラリー — 実装計画

作成 2026-11-14（基準コミット main 04ceaec）。設計書「H3cApp 静止画生成とギャラリーの実装計画」（設計版 1.0、同日）を、H3cApp の現行コードと上流の配布物に照らして具体化したもの。設計書の方針（動画が初期モード、用途ごとにモデル名を明示、画像と潜在をセットで保存、製品版は Python 不要）は変えない。変えるのは、推論側の作業範囲、VAE の扱い、画像サイズ、潜在の直接利用の位置づけ。

- 公式コード: https://github.com/MiniMax-AI/MiniMax-H3-Image （sha `40850901`。モデルコードは無く、README と HTTP クライアント例だけ）
- 公式重み: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image （sha `5f904730`、計 204 GB）
- ComfyUI 形式の重み: https://huggingface.co/Comfy-Org/MiniMax-H3-Image （sha `0bb149f5`）
- 基準実装: https://github.com/Anionex/h3-image-mlx （`eccdf321`、Apache-2.0。Metal カーネル 3 本は MIT）
- P-H3-Image: https://huggingface.co/PrunaAI/P-H3-Image-Comfy （sha `44a0fb73`）、diffusers 版 https://huggingface.co/PrunaAI/P-H3-Image （sha `7f98acef`）
- 算法の確認に読んだ実装: diffusers main `43e30028`、ComfyUI master `eee59185`

この文書の「確定」は、H3cApp のコードは直接読んだもの、上流は該当コミットのソース・設定・safetensors ヘッダを読んだもの。コードは実行しておらず、重み本体は取得していない。未検証の点は 9 章にまとめた。

## 1. 結論

1. **画像 DiT は移植ではなく既存 `h3_dit` の拡張で済む。** 公式 H3 Image の DiT は動画版と同形（50 層、hidden 5376、56 ヘッド × 128、テンソル名も同じ）で、既存エンジンは曲線形式 AdaLN と ComfyUI INT8 + ConvRot を既に読める。足すのは「音声なしの経路」と「ステップ別 PDD ヘッド」。
2. **画像 VAE デコーダは畳み込みではなく ViT 型**で、既存の動画デコーダと同系統。定数を可変にすれば足りる。
3. **テキストエンコーダは動画版と同一ファイル**（sha256 一致）。共用の検証は済んでいる。
4. **公式モデルは 4MP 以上で生成する設計**で、動画キャンバスの上限（600,000 px）はその対応域の下にある。設計書の「1024 級の初期サイズ」と「サイズ一致での潜在直接利用」は、どちらもフェーズ 0 の実測で成否を決める。
5. **潜在の直接利用は最後に回す。** 省けるのは 1 フレームの VAE エンコードだけで、画素は Qwen 視覚エンコーダのために引き続き要る。潜在の保存だけは最初から入れる。

## 2. 確定した事実

### 2.1 公式 H3 Image の構成

**DiT**（`transformer/config.json`、diffusers `transformer_minimax_h3_image.py`）

- 50 層、hidden 5376、56 ヘッド × 128（内部 7168）、SwiGLU FFN 14336、token refiner 2 層、パッチ (1,2,2)、入出力 24ch（行 96 次元）、テキスト 5120 次元、33.1B。`audio_enabled: false` で、音声の射影と出力は無い。
- 単一ストリームの自己注意。列は `[text | 参照画像 | 生成画像]`。
- 時刻は t = 1 − σ。guidance 埋め込みは無い。AdaLN は 6 パート × 5376 × 3 モダリティ = 96768 行で、モダリティは画像 0、テキスト 1。テキスト行は生成行と同じ t を使う。参照画像の行は σ = 0.001 に固定。
- RoPE は 3 軸。t 軸はテキストが 0..N−1、参照が N, N+1, …、生成画像がその次。h/w は面積で正規化した `(i/√(h·w) + (1 − n/√(h·w))/2)·32`。
- 出力は共有の `final_layer.norm` の後、ステップ i 専用のヘッド i。

**サンプラー（PDD）**

- 6 ステップ固定、σ = 1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0。shift は掛けない。CFG と negative prompt は無い。
- 更新は `x += (σ − σ_next)·v`（v はデータ向き、F32）。初期ノイズは F32 の [1,24,1,H/16,W/16]。
- ヘッドの格納は配布物で違う。公式 `pdd_fused_heads.safetensors` は `video_out.weight` F32 [6,96,5376] と `bias` [6,96] で、完全なヘッドが 6 個。Comfy-Org 版は `final_layer.video_out` [576,5376] で、ComfyUI のコメントによれば先頭が完全なヘッド、残り 5 個は先頭との差分（step k ≥ 1 は head[0] + head[k]）。メタデータに `pdd_steps` と `pdd_sigmas` を持つ。

**テキストエンコーダ**

- Qwen3-VL 32B の `hidden_states[50]`（最終 norm なし）、hidden 5120。チャットテンプレートも特殊トークンも付けず、プロンプトをそのままトークン化し、4096 トークンで切り詰める。
- 編集時は参照ごとに `"Picture i: "`（山括弧なし）と視覚ブロックを前置する。P-H3 と既存の H3cApp は `"<Picture i>: "` で、書式が違う。

**VAE**

- エンコーダは動画と同じ因果 3D CNN（[128,256,256,512,512,1024]、2 層/段）を 1 フレームで使う。潜在 24ch、空間 1/16、`latents_mean/std`（各 24 値）で正規化。
- デコーダは 2D ViT。24 層、16 ヘッド × 64 = 1024 次元、register 4 トークン + ゼロトークン 1、2 軸 RoPE（theta 100、rope_dim_ratio 0.5）、LayerScale、`proj_out` 768 = 3·16·16。出力は ImageNet の mean/std で逆正規化する。
- タイル復号（256 px、重なり 64 px 以上）が既定で有効。無効にすると出力が変わる。

**解像度**

- 両辺 32 の倍数。diffusers の既定は 2048×2048。4MP 未満の指定は、同じ縦横比の 4MP キャンバスで生成してから縮小する。有効域は約 4〜20MP、推奨 16MP。API の下限は 655,360 px、縦横比 1:3〜3:1。
- h3-image-mlx は 256〜4096、既定 1024²。

### 2.2 配布物

| 配布元 | DiT | AdaLN | PDD ヘッド | VAE | テキスト |
| - | - | - | - | - | - |
| MiniMaxAI/MiniMax-H3-Image | BF16 12 分割 66.28 GB、原形式キー | 密（`time_embedder` + 50 個の射影） | 別ファイル、完全 6 個 | F32 2.34 GB | BF16 14 分割 66.73 GB |
| Comfy-Org/MiniMax-H3-Image | 単一ファイル。bf16 40.2 GB / int8_convrot 21.0 GB / w4a8 12.5 GB | 曲線形式 `adaln_t_table` [1025,8] | DiT 内、先頭 + 差分 | fp16 1.17 GB | `Comfy-Org/MiniMax-H3` と同一ファイル |

- 公式の画像版と動画版（`MiniMaxAI/MiniMax-H3`、sha `0919fb68`）で、`text_encoder/` 23 個、`tokenizer/` 4 個、`processor/` 7 個のサイズと sha256 がすべて一致する。
- VAE は `latent_channels`、`latents_mean/std`、エンコーダ構成が一致。エンコーダ重みは 4 テンソル（`encoder.conv_in.bias`、`conv_out.bias`、`norm_out.weight`、`quant_conv.weight`）のバイト一致まで確認した。
- INT8 の対象は 50 ブロック × 4 線形（`qkv_proj`、`out_proj`、`fc1`、`fc2`）。形式は `X.weight`（I8）、`X.weight_scale`（F32）、`X.comfy_quant`（`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`）。token refiner は BF16 のまま。
- h3-image-mlx が既定で使うのは Comfy-Org の int8_convrot 版と fp16 VAE で、sha256 と revision を `weights.json` に固定している。

### 2.3 H3cApp の現状（04ceaec）

**そのまま使えるもの**

- DiT の定数とテンソル名が同じ: `H3_DIT_BLOCKS 50`、`H3_DIT_HIDDEN 5376`（`src/model/h3_dit_schedule.h`）、`condition_proj`、`token_refiner.blocks.{0,1}`、`rope.inv_freq`、`video_patch_proj`、`final_layer.{norm,video_out}`（`src/model/h3_dit_setup.c:48-63, 124, 584-596`）。
- RoPE の空間位置の式が同じ: `h3_frame_grid`（`src/core/h3_host.c:298`）。
- 曲線形式 AdaLN: `adaln_t_table` と `blocks.N.adaln_proj.linear` を読む（`src/model/h3_adaln_curve.c:31, 314`）。ComfyUI 由来のチェックポイントが曲線を内蔵する場合の判定もある（`h3_adaln_curve_in_dir`）。
- ComfyUI INT8 + ConvRot: `src/io/h3_weights.c:571` 以降、`src/io/h3_quant.c:417` 以降。SSD ストリーミングでも読める（`src/model/h3_dit_weights.c:255`）。
- テキストのみの経路: `h3_tokenizer_encode` → `h3_text_encode_bf16`（Qwen3-VL の先頭 50 層、`src/core/h3.c:1185-1193`）。
- 独自 sigma 列: `h3_params.video_sigmas`（`include/h3.h:277`）と Euler 更新。
- 参照条件への `0.999·x + 0.001·noise`: `h3_augment_span`（`src/core/h3_conditioning.c:596`）。上流の参照処理と同じ式。
- 動画 VAE デコーダは ViT 型で、タイル 256 px・重なり 64 px（`src/model/h3_video_vae.c:14-32`）。画像デコーダと同じ系統。
- モデルファミリー表（`src/model/h3_model_family.c:14`）。ステップ数の固定や能力の検証をエンジン側で行う既存の仕組み。
- 参照素材の複製: `ProjectFiles.importReference`（`Projects.swift:494`）。フォームの入力は保存・生成時に自動で `references/` へ取り込まれ、失敗は `failedReferenceImports` に残る（`Projects.swift:1250-1274`）。
- 生成物を入力に回す前例: `useVideoAsReference`（`Projects.swift:1404`）と API `/api/project/use-as-reference`。
- 共有ファイルを容量を増やさずに持つ前例: Ref2VA 追加時に FL2VA から複製する（`ModelAddRef2VA.swift:76`、APFS クローン）。

**画像生成を妨げる前提**

- `h3_load_dir` は `FL2VA/transformer`、`FL2VA/video_vae/source`、`FL2VA/audio_vae` を必須とする（`src/core/h3.c:104-109`）。
- `h3_generate` は 22 フレーム未満を拒否する（`src/core/h3.c:466`）。`h3_layout_build` は `frame_count < 5` を拒否する（`src/core/h3_host.c`）。設計書の「フレーム数 1 を渡す方法は採らない」は、そもそも通らない。
- DiT は音声を前提にしている。`final_layer.audio_out` を無条件に読み（`src/model/h3_dit_setup.c:598`）、音声行ゼロを拒否する（`src/model/h3_dit.c:628`）。`h3_dit_forward` と各 denoise 関数は音声潜在を引数に取る。
- 動画 VAE デコーダの形は `enum` の定数（`HIDDEN 2048`、`LAYERS 36`、`HEADS 32`、`ROPE_HALF 24`、`OUTPUT_PATCH 3·4·16·16`）で固定。
- C 側に画像の書き出しが無い（`src/io` にあるのは `h3_av_read_image_f32` と MP4 の書き出し）。
- アプリのモデル選択は 1 つ: `activeModelID`（`ModelLibrary.swift:228, 234`）。ウィンドウごとに `H3Engine` を 1 つ持つ（`GenerationViewModel.swift:791`）。
- `ProjectDraft` は秒数・ステップ・計算方式などが必須キーの動画専用構造（`Projects.swift:67-83`）。
- プロジェクトの生成物一覧はルート直下の `*.mp4` と同名の `.json`（`ProjectFiles.videos`、`Projects.swift:635`）。空の無題プロジェクトの判定は `containsVideo`（`Projects.swift:620`）で、呼び出し元は起動時の掃除（`Projects.swift:397`）と閉じるとき（`Projects.swift:1020`）。
- 動画キャンバスの上限は 600,000 px、辺は 256〜1344（`GenerationModels.swift:54-56`）。プリセットの最大は 960×544 と 768×768。
- プロジェクトの登録簿は「最近の項目」10 件だけ（`ProjectStore.maxRecents`、`Projects.swift:256`）。

## 3. 設計書から変える点

| 設計書 | 変更 | 理由 |
| - | - | - |
| §3 画像固有処理を h3-image-mlx から移植 | 既存 `h3_dit` に音声なし経路と PDD ヘッドを足す。h3-image-mlx は数値比較の基準としてだけ使う | DiT が同形で、ローダも対応済み（2.3） |
| §3 公式画像用 2D デコーダは専用実装 | 動画デコーダ（`h3_video_vae.c`）の形を可変にして共用する。時間方向のチャンク処理は通さない | どちらも ViT 型。違いは層数・次元・RoPE 比・出力パッチ |
| §8 `src/model` に画像 DiT を追加 | 新規ファイルは PDD とレイアウト、画像用の入口に限る | 同上 |
| §11 エンコーダのファイル共用を検証 | 公式配布は sha256 一致で確定。導入時に一致を確かめて複製する | 2.2 |
| §4 初期画像サイズは 1024 級 | フェーズ 0 で 1024² と 2048² を比べて決める。既定の候補は 2048²（公式の既定）と 1024²（h3-image-mlx の既定） | 公式は 4MP 未満を 4MP で生成して縮小する |
| §6、§10.1 サイズ一致で潜在を直接利用（512×512、672×384） | 「動画サイズで生成した画像が使える品質か」をフェーズ 0 で確かめてから着手を決める。フェーズの順は統合の後 | 動画キャンバスは公式の対応域（655,360 px 以上）の下 |
| §9 段階 3（潜在受け渡し）→ 段階 4（統合） | 統合を先、潜在受け渡しを後 | 効果が小さく、前提が未確定 |
| §2 P-H3-Image は H3 派生の画像モデル | 動画 FL2VA の派生として扱う。DiT は音声枝ありの既存経路を使う | 5.6 |
| §7 旧アプリからの互換は別問題 | 追加キーは省略可能にするので旧版でも開ける。ただし旧版の起動時の掃除でデータが消える件は対策を入れる | 7 章 |

## 4. 最初に対応する配布物

**Comfy-Org の int8_convrot 版 DiT（21.0 GB）+ fp16 VAE（1.17 GB）+ 既存の動画モデルのテキストエンコーダ** を最初の対象にする。

- 既存ローダの形式（単一 safetensors、INT8 + ConvRot、曲線形式 AdaLN）にそのまま当たる。
- 基準実装の h3-image-mlx が同じファイルを使うので、段階出力を直接比べられる。
- 公式 BF16（66 GB、密な AdaLN、ヘッド別ファイル）はその後。既存の密 AdaLN 経路と int8 注意キャッシュ生成に載るはずだが、PDD ヘッドの読み方が違う。
- w4a8（`asym_w4a8_int8`、コードブック 16 値）は既存ローダに無いので対象外。

モデルフォルダは動画モデルと別のレイアウトにする（例: `image/transformer/`、`image/vae/`、`text_encoder/`、`tokenizer/`）。`FL2VA/` を偽装しない。`ModelDetailView.swift` と `ModelManagerView.swift` は `FL2VA`/`Ref2VA` の有無で表示を決めているので、用途ごとの検査に分ける。

## 5. 実装フェーズ

### フェーズ 0: 基準の固定と早期確認

着手の可否と範囲を決める確認を先に済ませる。

1. **ローダ**: Comfy-Org int8_convrot 版 DiT のヘッダを `h3_gguf_inspect`（`include/h3.h:637`）と重みストアで開く。`adaln_t_table` [1025,8] を既存の曲線読み込みが受けるか、`final_layer.video_out` [576,5376] と `audio_patch_proj` の欠落でどこが止まるかを記録する。
2. **サイズと品質**: h3-image-mlx で同じプロンプト・seed の 768×768、960×544、1024²、2048² を出して比べる。決めること: 画像モードの既定サイズ、プリセット、動画サイズでの生成を「動画の開始画像用」として出すかどうか。
3. **基準出力**: 比較用に、トークン列、初期ノイズ（ファイルで共有。numpy PCG64 の F32 正規乱数は `h3_rng` と一致しない）、テキスト埋め込み、refiner 出力、各ステップの潜在、最終潜在、復号画像を保存する。h3-image-mlx は段階ごとに `embedding.safetensors`、`latent.safetensors` を書くので、それを使う。
4. **ライセンス**: 画像版の LICENSE と NOTICE を読み、対象地域、表示義務、再配布条件を確定する（8 章）。

完了条件: 1〜4 の結果がこの文書の付録に入り、既定サイズと潜在直接利用の扱いが決まっている。

### フェーズ 1: アプリの土台

推論を待たずに着手できる。既存の画像ファイルを「生成済み画像」として取り込んで一巡させる。

**モードと下書き**

- `GenerationModels.swift` に `GenerationKind { video, image }` を追加する。`CreationMethod.image`（`GenerationModels.swift:275`、「画像を入力して動画を作る」）は名前も意味も変えない。
- 画像用のサイズ型を `SizeProfile` と別に作る（`ImageSizeProfile`）。`AspectShape` は共用する。2 倍拡大や `customMaxPixels` は持ち込まない。
- `ProjectFile`（`Projects.swift:126`）に省略可能な `generationKind: String?` と `imageDraft: ImageDraft?` を足す。`ImageDraft` はプロンプト、モデル ID、サイズ、seed、枚数。`ProjectDraft` には触らない。`version` は 1 のまま。
- `GenerationViewModel` に `generationKind` と画像用の公開状態（`imagePrompt` など）を足し、`currentDraft`/`applyDraft`（`Projects.swift:677, 714`）の画像版を書く。`requestGenerate`（`Projects.swift:824`）と Enter・生成ボタン（`CreationFormView.swift:473, 767`）を現在のモードへ振り分ける。

**モデル選択**

- `H3ModelEntry`（`ModelLibrary.swift:67`）に省略可能な `kind: String?`（nil は動画）を足す。保存済みの一覧はそのまま読める。
- `ModelLibrary` に `activeImageModelID` を足す（UserDefaults の別キー）。`activeModel` は動画用のまま。
- 動画用と画像用でエンジンを別に持つ。`loadModel`（`GenerationViewModel.swift:791`）は動画用のまま、画像用エンジンは画像モードで最初に必要になったときに作る。

**プロンプト欄**

- `CreationFormView.swift` の `promptSection`（465 行）のツールバー先頭に、`video`/`photo` のメニューとモデル名のチップを置く。`composerChip`（684 行）を使う。
- 画像モードでは `topRow` の「作り方」、`imageInputSection`、`continuationSection`、`lipSyncSection`、`durationMenu`、`sizeMenu` の 2 倍拡大、`continueButtons` を出さない。`batchMenu` は枚数に読み替える（初期は 1）。
- `statusText` の見積もりと `validationMessage`（`GenerationViewModel.swift:994`）は画像用の分岐を持つ。モデル未導入時は「画像モデルを追加」を出す。
- プリセット（`SettingsPresets.swift`）は動画設定だけを対象のままにする。

**保存**

- `images/<yyyyMMdd-HHmmss_seedN>.png`、同名の `.json`（`ProjectImageRecord`）、`.latent.safetensors` を置く。名前の規則は `videoBaseName`（`Projects.swift:604`）に合わせる。
- 書き込みは一時名で行い、PNG と記録が揃ってから最終名へ移す。潜在の保存だけ失敗した場合は画像を残し、記録に `latent: nil` と理由を書く。
- `ProjectImageRecord` の項目: assetID、PNG のハッシュ、モデル（表示名、配布元、revision、重み指紋）、プロンプト、seed、ステップ、sigma 列、生成時の画素サイズ、潜在サイズ、latentSpaceID、正規化の識別、dtype、レイアウト、生成時間、アプリ版。
- `containsVideo` を「生成物があるか」（mp4、または `images/` の PNG）に置き換え、2 つの呼び出し元を直す。`references/` を除外する扱いはそのまま。
- AI 生成表示: 動画は `GenerationMetadata.tag` で付けている（`GenerationViewModel.swift:1467`）。PNG 用の埋め込みを `GenerationMetadata.swift` と `AIDisclosure.swift` に足す。

**ギャラリーと受け渡し**

- `ProjectVideosSheet`（`ProjectViews.swift:333`）、`VideoThumbnail`（682 行）、`VideoNode`（727 行）を、画像と動画を並べる `GalleryAsset` に広げる。動画の親子関係（連結クリップ）は `VideoNode` の扱いを保つ。
- サムネイルは縮小画像を遅延で作り、`images/.thumbnails/` に再生成可能なキャッシュとして置く。
- `PreviewPane.swift` に画像表示を足す。`resultURL` は動画前提なので、表示対象を画像と動画で分ける。
- 受け渡しは `useVideoAsReference` と同じ形で 3 つ。
  - この画像から動画を作る: `generationKind = .video`、`creationMethod = .image`、`imageInputMode = .firstLastFrame`、`firstFramePath = 画像`。動画プロンプトが空なら画像プロンプトを入れる。生成は始めない。
  - 最後の画像に使う: `lastFramePath = 画像`。
  - 参照画像に追加: `imageInputMode = .referenceImage`、`referenceImages` に追加。Ref2VA が無ければ `ModelAddRef2VA` へ案内する。
- 別プロジェクトの画像でも、`references/` への複製は既存の自動取り込みが行う。潜在を使う場合に備え、`importReference` が同名の `.latent.safetensors` と記録も一緒に複製するようにする。
- 動画の記録（`ProjectVideoRecord`、`Projects.swift:139`）に省略可能な `sourceImage`（元画像の assetID）を足す。
- 「すべてのプロジェクト」は、初期版では最近の項目と `~/Movies/H3cApp` 直下に限る。登録簿の新設は 9 章の未確定事項。

**API とローカライズ**

- `/api/generate` に `kind` を足すか `/api/generate-image` を新設し、`/api/result/image`、`/api/project` の一覧に画像を足す（`GenerationViewModel+API.swift`）。`docs/API.md` と `docs/API.ja.md` を更新する。
- 文言は `Packaging/ja.lproj` と `Packaging/en.lproj` の `Localizable.strings` に足す。

完了条件: 既存画像で、モード往復・保存・ギャラリー・動画への受け渡しが一巡する。旧プロジェクトが動画として開く。画像だけの無題プロジェクトが再起動後も残る。

### フェーズ 2: 公式画像推論

小さく確かめられる順に進める。各段は h3-image-mlx の段階出力と比べる。

**2a. 画像 VAE デコーダ**

- `h3_video_vae.c` の `enum` 定数（`HIDDEN`、`LAYERS`、`HEADS`、`FFN`、`ROPE_HALF`、`OUTPUT_PATCH`）を、読み込んだ設定から決まる構造体の値にする。動画は 36 層・2048・32 ヘッド・出力 3·4·16·16、画像は 24 層・1024・16 ヘッド・出力 3·16·16、RoPE 比 0.5。
- 画像用の入口（`h3_image_vae_decode`）は潜在 [24,H/16,W/16] を受け、時間方向のチャンク（`CHUNK_LATENT_TIME`、`FIRST_CHUNK_FRAMES`）を通さずにタイルを回す。
- 確認: 上流の `latent.safetensors` を入れて PNG を出し、上流の復号と比べる。動画側は `tests/test_real_video_vae.c` と `tests/test_semantic_vae.c` で回帰を見る。
- fp16 の VAE ファイルを F32 に広げて読めるかを確かめる（既存は F32 の `video_vae/source`）。

**2b. テキスト**

- 既存のテキストのみ経路をそのまま使う。確認するのは 2 点: `h3_tokenizer_encode` の第 3 引数が特殊トークンを付けない指定になっているか、4096 トークンの切り詰め。
- 上流の `embedding.safetensors`（BF16、[tokens,5120]）と突き合わせる。

**2c. 画像 DiT**

- **レイアウト**: `h3_layout_spec` に画像用の作り方を足す（text、参照画像、生成画像 1 枚、音声なし）。位置は `h3_frame_grid` を使い、t 軸は text_len から。`tests/test_plan.c` と同じ形でホスト側の単体試験を書く。
- **音声なし経路**: `audio_patch_proj` と `final_layer.audio_out` の読み込み、音声行の確保、`h3_dit_forward` の音声入出力を、プロファイルで切る。`h3_dit.c` に 227 箇所、`h3_dit_setup.c` に 85 箇所の `audio` があり、ここがフェーズ 2 の最大の作業。先に「音声行ゼロを許す」変更だけを入れ、動画の回帰（`tests/test_real_dit.c`、`tests/test_determinism.c`、`tests/repro_check.c`）が変わらないことを確かめてから画像を載せる。
- **AdaLN**: `h3_dit_schedule_precompute_curve` を画像条件（`visual_condition` は参照画像があるときだけ、`audio_condition` は 0）で呼ぶ。テキスト行が生成行と同じ t を使うこと、参照行が σ = 0.001 の行を使うことを `h3_dit_schedule_row_map` で確かめる。
- **PDD ヘッド**: `final_layer.video_out` を [steps,96,5376] として読み、ステップごとに選ぶ。Comfy-Org 形式は head[0] + head[k] に展開して持つ。公式の `pdd_fused_heads.safetensors` は完全なヘッドとして読む。新規ファイル `src/model/h3_pdd.c`。
- **sigma 列**: チェックポイントの `pdd_sigmas` メタデータを使う。無ければ 2.1 の固定値。
- **ファミリー**: `h3_model_family.c` に `H3_FAMILY_H3_IMAGE`（ステップは 6 だけ、能力は `H3_CAN_ATTENTION_CACHE | H3_CAN_ADALN_CURVE`）を足し、reuse・層スキップ・re-noise・音声 shift・独自 sigma をエンジン側で拒否する。LoRA は後続。

**2d. 入口と Swift 側**

- `include/h3_image.h` に `h3_image_ctx`、`h3_image_load_dir`、`h3_image_params`（幅、高さ、seed、進捗、中止フラグ、潜在の保存先、初期ノイズの差し替え）、`h3_image_generate` を置く。結果は RGB8 のバッファと正規化済み潜在。`h3_params` には足さない。
- メモリの事前判定（`h3_plan_generation` 相当）を画像用に書く。トークン数は 2048² で画像 4096 + テキスト。
- `H3Engine` モジュールに `H3ImageEngine.swift` を足す。`H3Engine.workQueue`（`H3Engine.swift:470`、いまは `private static`）をモジュール内で共有し、動画・キャッシュ生成・RefMod 抽出と重ならないようにする。App Nap の抑止、autorelease pool、「GPU の後片付けが済んでからストリームを閉じる」の順序（`H3Engine.swift:667-792`）は同じにする。
- 進捗の段階名を決め、`ProgressEstimator.swift:50` の表と `TimingCalibration` のキーに画像用を足す。
- PNG は Swift 側で ImageIO を使って書く。

完了条件: 基準入力で各段が基準と合う（BF16 の丸めと INT8 の量子化の差を切り分けたうえで）。M5 24GB で生成・中止・再生成が通り、中止後に GPU の割り当てが残らない。動画の回帰が変わらない。

### フェーズ 3: 統合

- 複数ウィンドウ、API、動画生成、画像生成、アップスケールが同時に走らないこと。
- モデル管理: 画像用の導入ウィザード（`ModelDownloadWizardView.swift`、`ModelDownloader.swift`）。動画モデルが導入済みなら、テキストエンコーダとトークナイザは sha256 を確かめて複製し、容量見積もりから除く。削除（`ModelRemoval.swift`）は他プロファイルからの参照を確かめる。
- 生成中の画像モデルの重みが、動画へ移るときに残らないこと。
- プロジェクトの移動（`ProjectMove.swift`）と削除で、`images/` と潜在、サムネイルが一緒に動くこと。
- `THIRD_PARTY_NOTICES.md` とモデル導入画面に、モデル・重み・実装の出典とライセンスを足す。
- 設計書 §10.3 の回帰項目を通す。

### フェーズ 4: 潜在の受け渡し（条件付き）

フェーズ 0 の 2 で「動画サイズで生成した画像が使える」と分かった場合に着手する。

- 省けるのは開始・終了画像の VAE エンコードだけ（`h3_pooled_video_encode`、`src/core/h3.c:1068`）。目安は 768×768 で約 8 秒（参照動画での実測係数 14.2 µs/pixel-frame、`GenerationViewModel.swift:583`。単一フレームは別に測る）。Qwen 視覚エンコーダは同じ画素を必要とする（`src/core/h3.c:1104-1113`）ので、画像の読み込みは残る。同じ入力での 2 回目以降は既存の条件キャッシュ（`H3_CACHE_CONDITIONING`）が効く。
- 入口: `h3_params` に `first_frame_latent`/`last_frame_latent`（safetensors のパス）を足し、`h3_pooled_video_encode` の代わりに読み込んで `h3_dit_patchify_video` へ渡す。潜在グリッドが `h3_latent_canvas(render_width, render_height)` と一致しなければ画素経路へ落とす。条件キーに潜在ファイルを入れる（`h3_conditioning_key`、`src/core/h3_conditioning.c:376`）。
- 読み込みは RefMods の safetensors 検査（`src/model/h3_refmod.c`）を参考にするが、経路は分ける。`.h3lt` には入れない。
- Ref2VA の参照は、参照用のキャンバス（`h3_reference_image_canvas`）が動画サイズと別なので対象外から始める。
- 上流の参照エンコードは posterior からのサンプル（seed 42 固定）で、H3cApp のエンコーダと同じ値にはならない。生成潜在と再エンコード潜在の差を、設計書 §10.1 の手順で記録する。

採用しない場合も、フェーズ 1 の潜在保存は残す（画像編集や再復号で使える）。

### フェーズ 5: P-H3-Image

- **由来**: H3 Image（2026-11-10 公開）ではなく動画 FL2VA の派生（10-28 公開）。README は、付属の LoRA（rank 128、qkv は 384、2.0 GB）を動画 FL2VA の DiT に強度 1.0 で当てると P-H3-Image になると書いている。本体が FL2VA と LoRA のマージそのものかは未確認。
- **DiT**: 音声枝あり（`audio_patch_proj`、`final_layer.audio_out`）、ヘッドは単一 [96,5376]、曲線形式 AdaLN `adaln_t_table` [4097,16]。列は `[text | refs | audio | image]` で、音声潜在 2 個 × 2ch を同時にデノイズして捨てる。既存の動画経路に近い。
- **サンプリング**: 8 ステップ、CFG なし、σ = 1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0。
- **テキスト**: Qwen3-VL-4B（36 層、hidden 2560）。`hidden_states[13,19,25,31,36]` を連結（12800）、標準化、Linear で 5120、逆標準化し、先頭トークンを `sink_embedding` に置き換える。ここが新規実装。
- **VAE**: エンコーダは動画と同じ。デコーダは 36 層・2048 次元の 2D（`proj_out` [768,2048]）で、次元は既存の動画デコーダと同じ。
- **配布**: 単一 safetensors。bf16 / int8_convrot / w4a8。int8_convrot は DiT 21.2 GB、テキスト 5.1 GB、VAE 3.1 GB。
- **ライセンス**: 動画版の MiniMax H3 Community License。
- **解像度**: 既定 2912²、推奨 8MP。

公式モデルの完成後に、4B エンコーダによる負荷の差と品質を比べて優先度を決める。

## 6. 検証

設計書 §10 に従う。コードに即して足す点:

- 単体試験は既存の並びに合わせて `tests/` に置く: `test_image_layout.c`（位置とセグメント）、`test_pdd.c`（ヘッドの展開と sigma）、`test_real_image_vae.c`、`test_real_image_dit.c`（基準潜在との比較）。`Makefile` に足す。
- 初期ノイズはファイルで共有する。アプリの seed は `h3_rng` を使い、上流と同じ画像になることは求めない。
- 動画の回帰は、音声なし経路と VAE の可変化を入れた各時点で `tests/repro_check.c` と `tests/test_determinism.c` を通す。
- 速度とメモリは M5 24GB で、モデル読み込み、テキスト、DiT、VAE 復号、保存に分けて測る。VAE 復号は 2048² でタイル数が増えるので別に見る。

## 7. リスク

- **旧版との併用でデータが消える。** 旧版は起動時に、mp4 を含まない無題プロジェクトを削除する（`sweepEmptyUntitledProjects`、`Projects.swift:397`）。新版で作った画像だけの無題プロジェクトは、旧版を一度起動すると消える。対策: 新版は無題プロジェクトの置き場所を別のフォルダにする。
- **旧版で新形式を開いて保存すると画像の下書きが落ちる。** 追加キーは省略可能なので旧版でも開けるが、旧版が `project.json` を書き直すと `imageDraft` は消える。`images/` のファイルは残る。
- **音声なし経路の変更が動画に波及する。** 変更は「音声行ゼロを許す」に限り、動画のコード経路は分岐させない。
- **低解像度での品質。** 公式の対応域の外で使うことになる。フェーズ 0 で決める。
- **ライセンスの違い**（8 章）。

## 8. ライセンス

- **コード**: h3-image-mlx は Apache-2.0。Metal カーネル 3 本は MIT（Copyright (c) 2026 David Dalcu、ddalcu/mlx-serve 由来）。数値比較の基準として使う範囲では持ち込みは無い。式や定数を移す場合は該当ファイルの表示を `THIRD_PARTY_NOTICES.md` に足す。
- **公式画像モデル**: MiniMax-H3-Image Community License。動画版とは別の契約で、年商 100 万 USD 超またはホスティングサービスでは事前許諾が要る（動画版は 2,000 万 USD）。再配布には LICENSE と NOTICE の同梱、商用の UI には「MiniMax-H3-Image」の表示が要る。上流の調査では EU、英国、韓国、米国が対象外地域と読めたが、LICENSE 本文での確認はフェーズ 0 で行う。
- **P-H3-Image**: 動画版の MiniMax H3 Community License。Qwen 由来の部分は Apache-2.0。
- アプリのリンクは動画モデルのものだけ（`AppMenus.swift:10-11`）。画像モデルのライセンスへのリンクと、導入時の表示を足す。

## 9. 未検証・未確定

| 項目 | 確かめ方 |
| - | - |
| DiT の音声依存の深さ（`h3_dit.c`、`h3_dit_block.c` の全体は未読） | フェーズ 2c の最初に「音声行ゼロを許す」変更の差分を見積もる |
| Comfy-Org 版 PDD ヘッドが「先頭 + 差分」であること（ComfyUI のコメントと h3-image-mlx の実装からの読み。重みでは未照合） | 公式 `pdd_fused_heads` と数値で突き合わせる |
| VAE エンコーダ重みの完全一致（4 テンソルまで確認） | 全テンソルのハッシュを比べる |
| `h3_tokenizer_encode` の特殊トークンの扱いと 4096 トークンの切り詰め | フェーズ 2b |
| 既存の曲線読み込みが [1025,8] の表を受けるか | フェーズ 0 の 1 |
| 既定の画像サイズ、上限、プリセット | フェーズ 0 の 2 と M5 24GB での実測 |
| 潜在直接利用を有効にするか | フェーズ 0 の 2、フェーズ 4 の品質比較 |
| 潜在の F16 保存 | F32 を基準に誤差と容量を比べる。2048² の F32 で約 1.5 MB なので急がない |
| 全プロジェクト索引の持ち方 | 初期版は最近の項目と既定フォルダ。登録簿の新設は、外部ディスクの切断と移動後の復旧を含めて別に設計する |
| 活性量子化の丸め（上流は `mx.round`、half-even かは未確認） | INT8 経路の数値比較で差が出たら確かめる |
| 画像版ライセンスの対象地域と表示義務 | フェーズ 0 の 4 |
| P-H3 本体が FL2VA + LoRA のマージと一致するか、最大トークン長 | フェーズ 5 の最初 |
