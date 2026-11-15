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
- ヘッドの格納は配布物で違う。公式 `pdd_fused_heads.safetensors` は `video_out.weight` F32 [6,96,5376] と `bias` [6,96] で、完全なヘッドが 6 個。Comfy-Org 版は `final_layer.video_out` [576,5376] で、先頭の 96 行が完全なヘッド、残り 5 個は先頭との差分（step k ≥ 1 は head[0] + head[k]）。公式の 6 個と数値で照合済み（付録 A.1）。メタデータに `pdd_steps` と `pdd_sigmas` を持つ。

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

1. **ローダ**（済、付録 A.1）: Comfy-Org int8_convrot 版 DiT のヘッダを `h3_gguf_inspect`（`include/h3.h:637`）と重みストアで開く。`adaln_t_table` [1025,8] を既存の曲線読み込みが受けるか、`final_layer.video_out` [576,5376] と `audio_patch_proj` の欠落でどこが止まるかを記録する。
2. **サイズと品質**（済、付録 A.2）: h3-image-mlx で同じプロンプト・seed の 768×768、960×544、1024²、2048² を出して比べる。決めること: 画像モードの既定サイズ、プリセット、動画サイズでの生成を「動画の開始画像用」として出すかどうか。
3. **基準出力**（済、付録 A.3）: 比較用に、トークン列、初期ノイズ（ファイルで共有。numpy PCG64 の F32 正規乱数は `h3_rng` と一致しない）、テキスト埋め込み、refiner 出力、各ステップの潜在、最終潜在、復号画像を保存する。h3-image-mlx は段階ごとに `embedding.safetensors`、`latent.safetensors` を書くので、それを使う。
4. **ライセンス**（済、付録 A.4）: 画像版の LICENSE と NOTICE を読み、対象地域、表示義務、再配布条件を確定する（8 章）。

完了条件: 1〜4 の結果がこの文書の付録に入り、既定サイズと潜在直接利用の扱いが決まっている。

### フェーズ 1: アプリの土台

推論を待たずに着手できる。既存の画像ファイルを「生成済み画像」として取り込んで一巡させる。

状態: 実装済み（ブランチ `image-phase1`、2026-11-15）。結果と、下の記述から変えた点は付録 B。

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

完了条件: 既存画像で、モード往復・保存・ギャラリー・動画への受け渡しが一巡する。旧プロジェクトが動画として開く。画像だけの無題プロジェクトが再起動後も残る。（満たした。付録 B.2）

### フェーズ 2: 公式画像推論

小さく確かめられる順に進める。各段は h3-image-mlx の段階出力と比べる。

状態: テキストから画像までを実装済み（ブランチ `image-phase2`、2026-11-15）。結果、下の記述から変えた点、残したものは付録 C。

**2a. 画像 VAE デコーダ**

- `h3_video_vae.c` の `enum` 定数（`HIDDEN`、`LAYERS`、`HEADS`、`FFN`、`ROPE_HALF`、`OUTPUT_PATCH`）を、読み込んだ設定から決まる構造体の値にする。動画は 36 層・2048・32 ヘッド・出力 3·4·16·16、画像は 24 層・1024・16 ヘッド・出力 3·16·16、RoPE 比 0.5。
- 画像用の入口（`h3_image_vae_decode`）は潜在 [24,H/16,W/16] を受け、時間方向のチャンク（`CHUNK_LATENT_TIME`、`FIRST_CHUNK_FRAMES`）を通さずにタイルを回す。
- 確認: 上流の `latent.safetensors` を入れて PNG を出し、上流の復号と比べる。動画側は `tests/test_real_video_vae.c` と `tests/test_semantic_vae.c` で回帰を見る。
- fp16 の VAE ファイルを F32 に広げて読めるかを確かめる（既存は F32 の `video_vae/source`）。

**2b. テキスト**

- 既存のテキストのみ経路をそのまま使う。特殊トークンが付かないことと、この経路の埋め込みで画像 DiT が正しく動くことは確認済み（付録 A.3）。`h3_tokenizer_encode` は長さを切り詰めないので、画像用の入口で 4096 トークンの上限を入れる。
- 上流の `embedding.safetensors`（BF16、[tokens,5120]）とは突き合わせ済み（付録 A.3）。トークンごとの相対誤差は中央値 5% 弱、コサイン類似度は中央値 0.999 で、差の大半は重みの量子化形式（Q4_K_M と int8）による。DiT 以降の比較には、基準の埋め込みをそのまま入力に使う。

**2c. 画像 DiT**

- **レイアウト**: `h3_layout_spec` に画像用の作り方を足す（text、参照画像、生成画像 1 枚、音声なし）。位置は `h3_frame_grid` を使い、t 軸は text_len から。`tests/test_plan.c` と同じ形でホスト側の単体試験を書く。
- **音声なし経路**: `audio_patch_proj` と `final_layer.audio_out` の読み込み、音声行の確保、`h3_dit_forward` の音声入出力を、プロファイルで切る。`h3_dit.c` に 227 箇所、`h3_dit_setup.c` に 85 箇所の `audio` があり、ここがフェーズ 2 の最大の作業。先に「音声行ゼロを許す」変更だけを入れ、動画の回帰（`tests/test_real_dit.c`、`tests/test_determinism.c`、`tests/repro_check.c`）が変わらないことを確かめてから画像を載せる。
- **AdaLN**: `h3_dit_schedule_precompute_curve` を画像条件（`visual_condition` は参照画像があるときだけ、`audio_condition` は 0）で呼ぶ。テキスト行が生成行と同じ t を使うこと、参照行が σ = 0.001 の行を使うことを `h3_dit_schedule_row_map` で確かめる。
- **取り込み時の判別**: `h3_gguf_inspect` は画像 DiT を動画の H3 DiT と同じに報告する（付録 A.1）。`audio_patch_proj.weight` が無いこと、`final_layer.video_out.weight` の行数が 96 の倍数（96 × ステップ数）であること、`config` メタデータの `pdd_steps` を見て画像用と判別し、`h3_gguf_info` に載せる。動画モデルの導入口は画像 DiT を断る。
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

状態: 実装済み（ブランチ `image-phase3`、2026-11-15）。結果と残したものは付録 D。

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
- 最終画像の一致は合否に使わない。テキスト埋め込みの 2% 弱の差で、6 ステップ後の潜在は 25〜37% ずれて別の構図になる（付録 A.3）。DiT は基準の埋め込みと初期ノイズを入れて 1 ステップ目の出力で比べ、2 ステップ目以降は基準の「前ステップの潜在」を入れて 1 ステップずつ比べる。
- 動画の回帰は、音声なし経路と VAE の可変化を入れた各時点で `tests/repro_check.c` と `tests/test_determinism.c` を通す。
- 速度とメモリは M5 24GB で、モデル読み込み、テキスト、DiT、VAE 復号、保存に分けて測る。VAE 復号は 2048² でタイル数が増えるので別に見る。

## 7. リスク

- **旧版との併用でデータが消える。** 旧版は起動時に、mp4 を含まない無題プロジェクトを削除する（`sweepEmptyUntitledProjects`、`Projects.swift:397`）。新版で作った画像だけの無題プロジェクトは、旧版を一度起動すると消える。対策: 新版は無題プロジェクトの置き場所を別のフォルダにする。フォルダだけでは足りず、`project.json` の印も変えた（付録 B.1）。
- **旧版で新形式を開いて保存すると画像の下書きが落ちる。** 追加キーは省略可能なので旧版でも開けるが、旧版が `project.json` を書き直すと `imageDraft` は消える。`images/` のファイルは残る。
- **音声なし経路の変更が動画に波及する。** 変更は「音声行ゼロを許す」に限り、動画のコード経路は分岐させない。
- **低解像度での品質。** 公式の対応域の外で使うことになる。フェーズ 0 で決める。
- **ライセンスの違い**（8 章）。

## 8. ライセンス

- **コード**: h3-image-mlx は Apache-2.0。Metal カーネル 3 本は MIT（Copyright (c) 2026 David Dalcu、ddalcu/mlx-serve 由来）。数値比較の基準として使う範囲では持ち込みは無い。式や定数を移す場合は該当ファイルの表示を `THIRD_PARTY_NOTICES.md` に足す。
- **公式画像モデル**: MiniMax-H3-Image Community License。動画版とは別の契約。本文を読んで確定した内容は付録 A.4。要点は、対象外地域（EU、英国、韓国、米国）では重みも出力も使えないこと、年商 100 万 USD 超またはホスティングサービスでは事前許諾が要ること（動画版は 2,000 万 USD）、重みを配る場合は契約書の写しと NOTICE を付けること、商用の UI に「MiniMax-H3-Image」を表示すること、生成機能を第三者に提供する製品には利用制限の通知と不正利用への対策が要ること。
- **P-H3-Image**: 動画版の MiniMax H3 Community License。Qwen 由来の部分は Apache-2.0。
- アプリのリンクは動画モデルのものだけ（`AppMenus.swift:10-11`）。画像モデルのライセンスへのリンクと、導入時の表示を足す。

## 9. 未検証・未確定

| 項目 | 確かめ方 |
| - | - |
| DiT の音声依存の深さ（`h3_dit.c`、`h3_dit_block.c` の全体は未読） | フェーズ 2c の最初に「音声行ゼロを許す」変更の差分を見積もる |
| 4096 トークンの切り詰め。`h3_tokenizer_encode` は切り詰めない（特殊トークンを付けないことは付録 A.3 で確認済み） | フェーズ 2b で画像用の入口に上限を入れる |
| 同じ重みでの H3cApp と上流のテキスト経路の比較。付録 A.3 の比較は重みの形式が違う（H3cApp は Q4_K_M の GGUF、上流は int8_convrot）ので、差が量子化形式だけによるとは言い切れていない。エンジンのテキストエンコーダは int8_convrot を読まない | 必要になったら、Q4_K_M の重みを浮動小数点で計算する参照を Python で作るか、エンジンに int8_convrot のテキストエンコーダの読み込みを足して比べる |
| 画像モデルで使うテキストエンコーダの配布物。動画モデルの GGUF を共用する前提（フェーズ 3）だと、埋め込みは上流と 5% ほど違い、同じ seed でも上流とは別の絵になる。品質の差は目視では見えていない（付録 A.3） | 導入ウィザードの設計時に決める。動画モデルを持たない利用者向けに何を取得させるかも合わせて決める |
| 既定の画像サイズ、上限、プリセット。付録 A.2 の提案（既定 1024²、上限 2048 級）は 2 プロンプト × 1 seed の目視による | 縦長、極端な縦横比、全身の人物、細かい文字を足して見る。4096 級は M5 24GB で実測する |
| 潜在直接利用を有効にするか。前提（動画サイズの画像が使える）は付録 A.2 で満たした | フェーズ 4 の品質比較 |
| 潜在の F16 保存 | F32 を基準に誤差と容量を比べる。2048² の F32 で約 1.5 MB なので急がない |
| 全プロジェクト索引の持ち方 | 初期版は最近の項目と既定フォルダ。登録簿の新設は、外部ディスクの切断と移動後の復旧を含めて別に設計する |
| 活性量子化の丸め（上流は `mx.round`、half-even かは未確認） | INT8 経路の数値比較で差が出たら確かめる |
| P-H3 本体が FL2VA + LoRA のマージと一致するか、最大トークン長 | フェーズ 5 の最初 |

## 付録 A. フェーズ 0 の結果

### A.1 ローダ（2026-11-14、ブランチ `image-phase0`）

**対象と方法**

- `Comfy-Org/MiniMax-H3-Image` `0bb149f5` の `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors`（20,979,314,802 バイト、978 テンソル）。
- 本体 21.0 GB は取得していない。ヘッダ（102,800 バイト）と、ブロックと refiner の大きい行列を除く全テンソル（170 MB、205 区間）だけを取り寄せ、残りを穴にした同じ大きさの疎なファイルで試した。**INT8 のブロック重みと refiner の行列の中身は読んでいない。**
- 検査プログラムは `tests/image_dit_probe.c`（`make h3_image_dit_probe`、`make test` には入れていない）。`build/bin/h3_image_dit_probe <transformer dir> build/bin/h3_shaders.metal [load]`。
- 比較用に、動画版 `Comfy-Org/MiniMax-H3` `54472221` の `minimax_h3_fl2va_pruned_int8_convrot.safetensors` のヘッダと、公式 `MiniMaxAI/MiniMax-H3-Image` `5f904730` の `pdd_fused_heads.safetensors`（12.4 MB）を取得した。

**そのまま通ったもの**

| 段 | 結果 |
| - | - |
| `h3_weight_store_open` | 1 シャード、978 テンソル。`config` メタデータから `pdd_steps` 6 と `pdd_sigmas` を読める |
| `h3_gguf_inspect` | DiT、`INT8 ConvRot`、50 ブロック、曲線 AdaLN あり。エラーなし |
| `h3_hyperflow_inspect` | two_time 0、50 ブロック |
| ブロックと refiner のテンソル名・形 | `condition_proj`、`token_refiner.blocks.{0,1}`、`token_refiner.final_norm`、`blocks.{0,49}` の 8 種、`rope.inv_freq` [16]、`video_patch_proj` [5376,96]、`final_layer.norm` がすべて動画ローダの期待どおり |
| INT8 + ConvRot | `blocks.0` の 4 線形が `h3_weight_is_comfy_int8` で 1、ConvRot のグループ 256、`weight_scale` を読める。`qkv_proj` は ComfyUI の [Q;K;V] 行順として並べ替えの対象になる（refiner の BF16 も同じ） |
| 曲線 AdaLN | `h3_adaln_curve_in_store` 1、`h3_adaln_curve_from_store` が grid 1025・rank 8 を受ける。`blocks.{0,49}.adaln_proj.linear` [96768,8] と `final_layer.adaln_proj.linear` [10752,8] の重み（F16）と bias（BF16）を F32 に広げて読める。非有限値なし |
| PDD の sigma 列での AdaLN | `h3_dit_schedule_precompute_curve` が 6 ステップを受ける。条件なしで時刻行 6、参照条件ありで 7（条件行が 1 つ増える） |
| 動画ローダ `h3_dit_load_t2va`（SSD ストリーミング） | テキスト refine、曲線 AdaLN の事前計算（50/50）、50 ブロックのストリーム準備まで進む |

**止まるところ**

動画ローダは `load_core` の最後（`src/model/h3_dit_setup.c:584-601`）で止まる。報告されるエラーは最後に書かれた `required weight is absent: final_layer.audio_out.bias`。実際の不足は次の 6 個で、これ以外に差は無い。

| テンソル | 画像版 | 動画ローダの期待 |
| - | - | - |
| `audio_patch_proj.weight` / `.bias` | 無い | [5376,32] / [5376] |
| `final_layer.audio_out.weight` / `.bias` | 無い | [32,5376] / [32] |
| `final_layer.video_out.weight` / `.bias` | [576,5376] / [576] | [96,5376] / [96] |

動画版の pruned int8_convrot とヘッダを比べた差もこの 6 個と、次の 2 点だけだった。

- AdaLN の bias が BF16（動画版は F16）。読み込みは通る。
- `blocks.N.attn.comfy_attention.config`（U8 [37]、中身は `[{"attention": "comfy_kitchen_int8"}]`）が 50 個ある。エンジンは読まないので無視される。

**PDD ヘッドの並び**

公式の 6 個（[6,96,5376]）と Comfy-Org 版の 576 行を突き合わせた。

- 先頭 96 行は公式の head[0] とビット一致（相対誤差 0）。
- k = 1..5 の 96 行は head[k] − head[0] と一致（相対誤差 1e-8）。bias も同じ。
- 各ヘッドの L2 は先頭 58.85、差分 1.15〜1.51。

したがって step k のヘッドは「先頭 + k 番目」で復元できる。h3-image-mlx の読みと同じ。

**分かった問題**

- `h3_gguf_inspect` は画像 DiT を動画の H3 DiT と区別しない（role DiT、ファミリーは MiniMax H3）。いまの取り込み口に渡すと動画モデルとして受け付け、生成時に上のエラーで止まる。判別を 2c に足した。
- `tests/test_real_dit.c` は 04ceaec の時点でコンパイルできない（`h3_dit_load_t2va` の引数が 28 個に対して 25 個。LoRA と prep の引数が入る前の呼び方）。2c が回帰の確認に挙げているので、音声なし経路に手を入れる前に直す。

**計画への影響**

- 4 章の「既存ローダの形式にそのまま当たる」は、ヘッダと小さいテンソルの範囲で確認できた。重みの読み込みで新しく要るのは、音声の 4 テンソルを読まない経路と、6 ヘッドの読み方だけ。
- AdaLN、INT8 + ConvRot、refiner、RoPE、パッチ射影のローダには手を入れない。
- INT8 本体の復元値は下の追試で確認した。

**本体での追試（同日）**

本体（sha256 `83eb140f…6ee622`、h3-image-mlx の `weights.json` の値と一致）を取得し（置き場所は `~/H3cApp-models/MiniMax-H3-Image/`。`~/Documents` はクラウドへ同期されるので外に置く）、同じ検査を本体で繰り返した。結果は疎なファイルのときと同じ。

加えて、エンジンが復元した重み（`h3_weight_load_bf16`: INT8 × 行ごとの scale、ConvRot の逆回転、QKV の行の並べ替え）を、上流の式（h3-image-mlx `core.py`: `(int8 × scale)` を入力側 256 要素ごとに H256 と掛ける）で復元した値と比べた。`h3_image_dit_probe … dump <tensor> <out>` で書き出して numpy で比較。

| テンソル | 相対 L2 誤差 | BF16 に丸めた基準と一致する要素 |
| - | - | - |
| `blocks.0.attn.qkv_proj.weight` [21504,5376] | 1.66e-3 | 99.91% |
| `blocks.0.attn.out_proj.weight` [5376,7168] | 1.66e-3 | 99.86% |
| `blocks.0.mlp.fc1.weight` [28672,5376] | 1.66e-3 | 99.91% |
| `blocks.0.mlp.fc2.weight` [5376,14336] | 1.66e-3 | 99.88% |
| `blocks.27.attn.qkv_proj.weight` | 1.66e-3 | 99.91% |
| `blocks.49.mlp.fc2.weight` | 1.65e-3 | 99.89% |
| `token_refiner.blocks.1.attn.qkv_proj.weight`（BF16） | 0 | 100% |

相対 L2 の 1.66e-3 は F32 を BF16 に丸める誤差そのもので、残りの 0.1% の要素は 1 ulp の違い（丸める前の演算順の差）。QKV の並べ替え（ComfyUI の [全 Q; 全 K; 全 V] → ヘッドごとの [q; k; v]）も上流の読みと合っている。

**上流との違いとして残る点**: h3-image-mlx は重みを復元せず、活性も回転して INT8 に量子化する W8A8 で積を取る（`rotate_quantize`）。H3cApp は重みを BF16 に戻してから自前の経路（BF16 または自前の int8 量子化）で計算する。したがって出力はビット一致しない。フェーズ 2c の比較は、この差を見込んだ許容幅で行う。

### A.2 サイズと品質（2026-11-14）

**条件**

- M5（24 GB）。h3-image-mlx `eccdf321`（0.1.0a5）、mlx 0.32.4、`stream` モード、backend は自動選択（M5 なので W8A8 TensorOps）、prefetch と async は切。
- DiT は Comfy-Org int8_convrot、VAE は fp16。seed 42、6 ステップ。
- テキスト埋め込みは上流のテキスト段ではなく、H3cApp 自身の経路（`h3_image_text_embed`、手元の Qwen3-VL Q4_K_M、先頭 50 層）で作って DiT 段に渡した。翌日に上流の埋め込みで同じ 8 組を作り直した（付録 A.3）。時間、メモリ、下の所見は変わらない。
- プロンプトは 2 つ（`misc/image-phase0/prompts.json`）: 陶芸家の人物写真（`potter`）、文字入りの静物（`teapot`、手書きの「OPEN」の札）。

**時間とメモリ**（2 プロンプトでほぼ同じ。値は `teapot`）

| サイズ | 画素数 | 系列長 | DiT 段 | 1 ステップ | DiT 段のピーク | VAE 段 | VAE 段のピーク |
| - | - | - | - | - | - | - | - |
| 768×768 | 0.59 MP | 607 | 41.7 s | 6.8 s | 2.48 GB | 0.8 s | 1.02 GB |
| 960×544 | 0.52 MP | 541 | 40.8 s | 6.7 s | 2.48 GB | 0.9 s | 1.02 GB |
| 1024×1024 | 1.05 MP | 1055 | 50.5 s | 8.3 s | 2.48 GB | 1.2 s | 1.04 GB |
| 2048×2048 | 4.19 MP | 4127 | 133.0 s | 22.1 s | 3.35 GB | 4.0 s | 1.25 GB |

- 小さいサイズで時間が縮まらないのは、`stream` モードが毎ステップ全ブロックを読み直すため（6 ステップで 117 GB を読み、読み込み 18 秒 + 変換 7 秒）。重みを常駐させる構成なら下がる。
- これは h3-image-mlx の値で、H3cApp の速度ではない。メモリは各段が別プロセスで動いたときの値。

**見た目**（`misc/image-phase0/<prompt>/sheet_full.jpg`、`sheet_crop.jpg`）

- 4 サイズとも、構図・人物・手・文字が破綻なく出た。`teapot` は全サイズで「OPEN」が正しく書かれ、黄色いホーローのポット、白いカップ、青い木のテーブルが揃う。`potter` は顔、手、ろくろ、棚の器が自然。
- 768×768 と 960×544（公式の対応域 655,360 px の下）でも、破綻や崩れは見えない。細部はやや硬い（輪郭が強め）が、動画の開始画像としては使える。
- 2048² は肌や釉薬の質感が細かい。1024² は 2048² より細部が甘いが、縮小表示では差が小さい。
- 判断は 2 プロンプト × 1 seed を目で見たもので、指標による評価はしていない。縦長や極端な縦横比、人物の全身、細かい文字は試していない。

**決めたこと（提案）**

| 項目 | 提案 | 根拠 |
| - | - | - |
| 画像モードの既定サイズ | 1024×1024 | 2048² の 4 割弱の時間で、破綻のない画像が出る |
| プリセット | 1024 級と 2048 級を縦横比ごとに。上限は当面 2048 級 | 公式の既定（2048²）を選べるようにする。4096 級は M5 24 GB での実測が無い |
| 動画サイズでの生成 | 「動画の開始画像用」として、動画のキャンバスと同じサイズを選べるようにする | 768×768 と 960×544 で使える画像が出た |
| 潜在の直接利用（フェーズ 4） | 前提は満たした。着手の可否はフェーズ 4 の品質比較（潜在の直接利用と再エンコードの差）で決める | 同上 |

設計書の「1024 級の初期サイズ」は、そのまま採れる。

### A.3 基準出力とテキスト経路（2026-11-14、15）

**テキスト経路**

- トークン ID: 上流のトークナイザ（`tokenizers`、`add_special_tokens=False`）と `h3_tokenizer_encode` の結果が、2 プロンプト（39 トークンと 31 トークン）で完全に一致した。特殊トークンは付かない。手元の `tokenizer.json` の sha256 は h3-image-mlx が固定している値（`a5d85b6d…`）と同じ。
- 上流のテキストエンコーダ（`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`、27,141,342,152 バイト、sha256 `bc2ced0f…` は h3-image-mlx の固定値と一致）: 言語層は 50 層で、メタデータは `{"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}`。H3cApp の `h3_text_encode_bf16`（先頭 50 層、最終 norm なし）と同じ位置の出力である。埋め込み表と norm は BF16、射影 7 種 × 50 層が int8 + ConvRot。視覚タワー（`visual.*`、BF16）も同じファイルに入っている。
- 上流のテキスト段は 31〜39 トークンで 6.2〜6.7 秒、ピーク 1.1〜1.2 GB。

**埋め込みの数値比較**（[tokens,5120]、BF16。先頭トークンの行ノルムは約 15,100、ほかは 240〜370 なので、行ごとの値で見る）

3 つの埋め込みを比べた。上流そのまま（int8 重み、活性化も int8 に量子化する W8A8）、同じ int8 重みで活性化を量子化しない計算（`stage_trace.py` の `H3_FLOAT_ACTIVATIONS=1`。重みそのものの値を表す）、H3cApp（Q4_K_M の GGUF）。

| 基準 | 比較対象 | トークンごとの相対 L2（中央値 / 最大） | トークンごとのコサイン（中央値 / 最小） | 行ノルムの比 |
| - | - | - | - | - |
| int8 重み・活性化は量子化なし | 上流そのまま（W8A8） | 1.4% / 3.0〜4.6% | 0.99990 / 0.9990 | 0.987〜1.010 |
| int8 重み・活性化は量子化なし | H3cApp（Q4_K_M） | 4.5〜4.7% / 16〜21% | 0.9990 / 0.977〜0.987 | 0.966〜1.010 |
| 上流そのまま（W8A8） | H3cApp（Q4_K_M） | 4.6〜4.9% / 19〜21% | 0.9989 / 0.978〜0.983 | 0.956〜1.013 |

- H3cApp と上流の差は、上流の活性化量子化（1.4%）より、重みの量子化形式の違い（Q4_K_M と int8、4.5% 前後）が大きい。H3cApp との比較で誤差が最大の行は文末近くの語（`teapot` の 2 回目の「 te」、`potter` の「 of」）で、系列の後ろ 4 分の 1 がやや大きい。
- 同じ重みを両方の実装に通した比較ではない。エンジンのテキストエンコーダは int8_convrot を読まないので、「差は量子化形式だけによる」ことは示せていない（9 章）。位置・層数・形式が合っていること、差が量子化で説明できる大きさであることまでが確認できた範囲。

**埋め込みの差が画像に及ぼす影響**（同じ seed 42・同じ初期ノイズ、上流そのままの埋め込みで作った結果を基準にした相対 L2）

| 比較 | サイズ | refiner 出力 | 1 ステップ後 | 3 ステップ後 | 最終潜在 | 画像の PSNR |
| - | - | - | - | - | - | - |
| 同じ入力の再実行（`teapot`） | 768×768 | 0 | 0 | 0 | 0 | 一致 |
| 活性化を量子化しない埋め込み（`teapot` / `potter`） | 1024×1024 | 1.8% / 1.7% | 0.12% / 0.12% | 9.7% / 5.2% | 37% / 25% | 17.6 / 19.8 dB |
| H3cApp の埋め込み（`teapot`） | 4 サイズ | 6.7% | 0.19〜0.66% | 3.7〜12% | 15〜46% | 15.3〜24.6 dB |
| H3cApp の埋め込み（`potter`） | 4 サイズ | 8.8% | 0.23〜0.43% | 4.9〜12% | 24〜52% | 13.4〜20.4 dB |

- 上流の DiT 段は同じ入力で再実行するとビット一致する。基準出力は再現できる。
- 6 ステップのサンプラは埋め込みの小さな差を増幅する。refiner 出力で 1.7% の差でも、最終潜在は 25〜37% ずれて構図が変わる。H3cApp の埋め込みによるずれ（15〜52%）はこれと同じ桁で、特別に大きくはない。
- どの埋め込みでも画像はプロンプトどおりで、品質の優劣は目視では付かない（上流の埋め込みでも 4 サイズすべてで「OPEN」の札が正しく書かれ、人物の顔と手は自然）。
- フェーズ 2 への帰結: 最終画像の一致は合否に使えない。段ごとに基準の入力を入れて比べる（6 章）。1 ステップ目の出力は refiner 出力の 7〜9% の差に対しても 0.7% 以内に収まるので、ここが DiT の実装差を見る場所になる。

**VAE**

- 画像 VAE（Comfy-Org fp16）のエンコーダ 116 テンソルと `quant_conv` 2 テンソルが、手元の動画 VAE（`video_vae/source/model.safetensors`、F16）とすべてビット一致した。デコーダは 296 テンソルすべて形が違い、`post_quant_conv` も別物。
- 手元の動画 VAE は F16 で、既存の読み込みはこれを読んでいる。2a の「fp16 を読めるか」は、動画側では既に成り立っている。

**保存したもの**（`misc/image-phase0/<prompt>/`、gitignore 済み、計 約 290 MB）

埋め込みは 3 種類: `embedding.upstream.safetensors`（上流そのまま。基準）、`embedding.upstream-float.safetensors`（活性化を量子化しない計算）、`embedding.h3c.safetensors`（H3cApp。`ids` も入っている）。

実行結果は `<埋め込みの種類>/<WxH>/` に置いた。`upstream/` が基準で、2 プロンプト × 4 サイズの 8 組。`h3c/` は 11 月 14 日に作った同じ 8 組、`upstream-float/` は 1024×1024 の 2 組、`upstream-repeat/` は再現性の確認に使った `teapot` 768×768 の 1 組。

| ファイル | 中身 |
| - | - |
| `request.json` | プロンプト、サイズ、seed |
| `embedding.safetensors` | `embedding` BF16 [tokens,5120]、`tags` |
| `trace.safetensors` | `noise` F32 [24,H/16,W/16]、`sigmas`、`angles`（RoPE の角度 [系列長,48]）、`refined_text` BF16 [tokens,5376]、`block0_mod`・`block49_mod` [6,3,6,5376]、`final_mod` [6,1,2,5376]、`patches_after_step_1..6` と `head_out_step_1..6` F32 [行数,96] |
| `latent.safetensors` | 最終潜在 F32 [24,H/16,W/16]（正規化済み） |
| `image.png` | 復号結果 |
| `profile.dit.json`、`time.*.txt` | 段ごとの時間とピークメモリ |

作り方は `tools/image_phase0/README.md`。上流のコードは書き換えず、DiT 段のソースに保存の行を実行時に差し込んでいる（`stage_trace.py`）。

**まだ無いもの**

- 参照画像つき（編集）の基準出力。参照の VAE エンコードと視覚エンコーダが絡むので、フェーズ 2 の後半で作る。
- H3cApp 側の DiT・VAE 出力との数値比較。フェーズ 2 の各段で、この基準を入力にして行う。

### A.4 ライセンス（2026-11-14）

読んだもの: `MiniMaxAI/MiniMax-H3-Image` `5f904730` の `LICENSE`（18,208 バイト、2026-11-11 更新）。GitHub の `MiniMax-AI/MiniMax-H3-Image` `40850901` の `LICENSE` とバイト一致。h3-image-mlx 同梱の `MODEL_LICENSE.txt` は申請先 URL の 1 行だけ違う旧版。**公式リポジトリに `NOTICE` ファイルは無い。** 文面は契約書の III.4 が指定している。

法的な判断ではなく、条文の読み取りである。配布の前に条文そのものを確かめること。

| 項目 | 条文 | 内容 |
| - | - | - |
| 対象外地域 | I.5、V.4、別紙 A の 1 | EU、英国、韓国、米国。この地域では重みも**出力も**使用・複製・配布・表示できない。日本は対象地域に入る |
| 量子化版の扱い | I.11 | 改変や派生は Model Derivative で、同じ契約が及ぶ。Comfy-Org の int8 版もこれに当たる |
| 重みを配るとき | III.1、III.2、III.4 | 契約書の写しを渡す。改変したファイルには改変した旨を目立つように記す。`NOTICE` テキストに「MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.」を入れる |
| 推奨（義務ではない） | III.3 | 「Powered by MiniMax-H3-Image」の表示、生成ファイルへの AI 生成の識別子、利用体験の公開 |
| 事前許諾 | IV.1 | 自社と関連会社の年商が 100 万 USD を超える場合、または第三者に Hosted Service を提供する場合 |
| UI 表示 | IV.2 | 商用の製品・サービスは UI に「MiniMax-H3-Image」を目立つように表示する |
| 利用者への拘束 | V.2 | 重みや、それを組み込んだ製品を使わせる前に、V 章と別紙 A と同等以上の条件に利用者を拘束し、その制限が適用されることを知らせる |
| 他モデルの改善 | V.3 | 重みや出力を、他の AI モデルの改善に使えない |
| 生成機能を提供する製品 | V.5 | 第三者に生成機能を提供する製品は、違反利用を防ぐ合理的な対策を実装・維持・点検し、違反の通報窓口を持ち、通報を受けたら調べて止める |
| 出力の権利 | VI.4 | MiniMax は出力に権利を主張しない。責任は利用者にある |
| 公開時の表示 | 別紙 A の 12 | 出力を公開の場に出すときは、機械生成であることを明示する |
| 準拠法 | IX | 香港法、香港の裁判所の専属管轄 |

**H3cApp への影響**

- 動画版と同じく、重みは同梱せず利用者が取得する形にする。その場合も V.2（利用条件の通知）と V.5（生成機能を提供する製品の対策）は「重みを組み込んだ製品を使わせる」「生成機能を提供する」に当たるかどうかで読みが分かれる。導入ウィザードで契約書の全文へのリンクと対象外地域を示し、同意を取ってから取得を始める形を既定にする。
- 対象外地域は出力の利用まで及ぶ。導入ウィザードの最初に地域の制限を出す。
- 画像モードの UI にモデル名「MiniMax-H3-Image」を出す（モデル名のチップで満たせる）。
- PNG に AI 生成の識別子を入れるのは推奨。別紙 A の 12 に備えて、メタデータ（tEXt/iTXt か XMP）に入れる方針を 5 章のフェーズ 1 に足す価値がある。
- V.5 の「通報窓口」を配布アプリとしてどう満たすかは未決。動画版の扱い（`THIRD_PARTY_NOTICES.md`）と合わせて決める。

## 付録 B. フェーズ 1 の結果

### B.1 実装したものと、計画から変えた点（2026-11-15、ブランチ `image-phase1`）

新規ファイルは `ImageModels.swift`（種別、サイズ、下書き、記録、`images/` の読み書き）、`ImageMetadata.swift`（PNG の読み書きと AI 生成表示）、`GenerationViewModel+Images.swift`（取り込み、表示、受け渡し）。既存の変更は `Projects.swift`、`GenerationViewModel.swift`、`GenerationViewModel+API.swift`、`ModelLibrary.swift`、`CreationFormView.swift`、`PreviewPane.swift`、`ProjectViews.swift`、`SettingsPresets.swift`、両言語の `Localizable.strings`、`docs/API.md`、`docs/API.ja.md`。

計画どおりに入れたもの: `GenerationKind`、`ImageSizeProfile`（標準 1024 級・大 2048 級・動画用）、`ProjectFile` の `generationKind` と `imageDraft`、`ProjectImageRecord`、`images/` への保存（隠し名で書いてから改名）、動画と画像を並べる一覧、サムネイルのキャッシュ（`images/.thumbnails/`）、プレビューの画像表示、受け渡し 3 種、`ProjectVideoRecord.sourceImage`、`importReference` に続く記録と潜在の複製、「すべてのプロジェクト」（最近の項目、`~/Movies/H3cApp` 直下、無題プロジェクト）。

計画から変えた点:

| 計画 | 実際 | 理由 |
| - | - | - |
| 種別のメニューとモデル名のチップを、プロンプト欄のツールバー先頭に置く | 種別のメニューはパネル上段（`topRow`）の先頭。モデルのチップは画像モードのツールバーだけに置く | ツールバーに 2 つ足すと、動画側のチップと「続きを生成」がパネル幅（640 pt）に収まらない。実測で約 20 pt、続きのボタンが出るとさらに超えた。動画のモデルはウィンドウのツールバーにある |
| `H3ModelEntry` に `kind` を足し、同じ一覧に置く | `kind` は足した。画像モデルは別の一覧（`H3ModelLibrary.imageModels`、`activeImageModelID`）に置く | 旧版は `models` の全項目を動画モデルとして読み込む |
| 無題プロジェクトの置き場所を別のフォルダにする | フォルダを `UntitledProjects/` にし、`project.json` の印も `untitledProject` に変えた。旧フォルダ `Untitled/` の旧プロジェクトはそのまま扱う | 旧版は、最後に開いたプロジェクトを起動時に開き直し、閉じるときに「`untitled` で動画なし」なら消す。フォルダを変えただけでは、画像だけの無題プロジェクトがこの経路で消える |
| PNG の AI 生成表示を `GenerationMetadata.swift` と `AIDisclosure.swift` に足す | `ImageMetadata.swift` に書いた。XMP（iTXt）に IPTC の AI 項目を入れる | MP4 のボックス操作とは共有する部分が無い |
| `/api/generate` に `kind` を足すか `/api/generate-image` を新設 | `/api/generate` に `kind` を足した（`image` は 501）。ほかに `/api/kind`、`/api/project/import-image`、`show-image`、`use-image`、`delete-image`、`/api/result/image`、`GET /api/status` の `form` | 取り込みと受け渡しを API からも行えるようにした。確認の自動化にも使う |

取り込みの扱い: PNG はバイト単位でそのまま置く。ほかの形式は向きを正して PNG にし、元のメタデータを引き継ぐ（元画像が持つ出自の情報を落とさない）。取り込んだ画像に AI 生成の印は付けない。記録には `importedFrom` を書く。

### B.2 確認したこと

`tools/image_phase1/roundtrip_check.py`（モデル不要、API 経由、42 項目）がすべて通る。アプリは別名の複製を、ホームを作業フォルダへ向けて動かすので、導入済みアプリのプロジェクトと設定には触れない。確かめている内容:

- 新しいウィンドウは動画モードで、無題プロジェクトは `UntitledProjects/` にでき、`project.json` に旧版が反応する印が無い。
- 旧版の無題プロジェクトは、動画があれば残り、空なら掃除される。
- 画像ファイル（PNG と、横倒しで保存された JPEG）を取り込める。PNG はバイト一致、JPEG は正しい向きになる。記録にハッシュ、ID、画素サイズが入る。画像でないファイルは理由を返して何も足さない。
- プレビューへの表示と `GET /api/result/image`。
- 受け渡し 3 種で動画のフォームが切り替わり、画像が `references/` へ記録ごと複製され、`project.json` が相対パスで保存される。生成は始まらない。
- アプリを強制終了して起動し直しても、画像だけの無題プロジェクトが残り、開き直され、画像モードと受け渡し済みのフォームが戻る。
- 旧形式のプロジェクトは動画プロジェクトとして開き、`project.json` に画像用のキーが増えない。

別に手で確かめたこと:

- 取り込んだ画像を最初の画像にして、手元の動画モデル（GGUF、SSD ストリーミング、256×256、1 秒、4 ステップ）で動画を生成した。動画の記録の `sourceImage` に画像の ID が入り、`GET /api/project` の `source_image` に出る。プレビューは画像から動画へ切り替わる。
- `ImageMetadata` の AI 生成表示は、単体のプログラムで PNG に書いて読み戻した（IPTC の `DigitalSourceType`、`AISystemUsed`、`AISystemVersionUsed`、`AIPromptInformation`、`dc:description`、`xmp:CreatorTool`）。生成経路がまだ無いので、アプリからは呼ばれていない。
- 画面（画像モード、動画モード、一覧、すべてのプロジェクト）はスクリーンショットで確認した。
- `native/H3Spike/check_localizations.sh`: 1229 キー、不足 0。

### B.3 まだ無いもの、確かめていないこと

- 画像の生成（フェーズ 2）。画像モードの生成ボタンは常に無効で、「画像モデルを追加」から開くモデル管理には、まだ画像モデルの導入口が無い（フェーズ 3）。
- 画像モードの seed の入力欄。下書きと記録には項目がある。
- 画像の削除は、操作と API を書いたが実行していない（ファイルがゴミ箱に入るため）。
- 画像を別のプロジェクトへ移す操作。無題プロジェクトの保存（フォルダごとの移動）では一緒に動く。
- 「すべてのプロジェクト」にある別プロジェクトの動画は、参照に使うことと Finder での表示だけができる。
- リリースビルド（`package_app.sh`）での表示と英語表示。確認はデバッグビルドを直接起動して行ったので、文言は日本語のキーのまま出ている。
- 旧版（0.9.0）を実際に起動しての併用試験。旧版の動作はコードから読んだもの。

## 付録 C. フェーズ 2 の結果

### C.1 実装したもの（2026-11-15、ブランチ `image-phase2`）

| 部分 | ファイル | 内容 |
| - | - | - |
| 画像 VAE デコーダ | `src/model/h3_image_vae.{h,c}` | 2D ViT のタイル復号。層数・幅は重みの形から読む。16×16 潜在のタイルを 4 以上重ね、線形に混ぜる |
| 画像 DiT | `src/model/h3_dit_setup.c`、`h3_dit_block.c`、`h3_dit.c`、`h3_dit_private.h` | 音声行なしのレイアウトを許す。`final_layer.video_out` をステップごとのヘッドに展開する（head[0] + 差分）。最終層は F32。入口は `h3_dit_forward_image` |
| ファミリー | `src/model/h3_model_family.c`、`include/h3.h` | `H3_FAMILY_H3_IMAGE`（6 ステップだけ、能力なし） |
| 入口 | `include/h3_image.h`、`src/core/h3_image.c` | `h3_image_load_dir`、`h3_image_generate`。テキスト（4096 トークンで切る）→ DiT 6 ステップ → VAE。sigma 列はチェックポイントのメタデータから読む |
| 取り込み時の判別 | `src/io/h3_gguf.c`、`ModelImport.swift` | `h3_gguf_info.still_image`。動画モデルの取り込みは静止画用 DiT を理由を示して除外する |
| Swift | `H3ImageEngine.swift`、`GenerationViewModel+Images.swift` ほか | 生成、進捗、中止、PNG（AI 生成表示つき）・記録・潜在の保存、画像モデルのフォルダ登録 |
| API | `GenerationViewModel+API.swift`、`docs/API*.md` | `/api/generate` の `kind: "image"`、`/api/image-models`、`/api/image-models/add` |
| 検査 | `tests/image_vae_check.c`、`tests/image_dit_check.c`、`tools/h3_image_cli.c`、`tools/image_phase2/` | 基準出力との段ごとの比較と、通しの実行 |

モデルフォルダは `transformer/`、`vae/`（`config.json` を含む）、`text_encoder/`、`tokenizer/`。DiT は SSD ストリーミング（2 ブロック常駐）だけで動かす。

計画から変えた点:

| 計画 | 実際 | 理由 |
| - | - | - |
| `h3_video_vae.c` の定数を設定値にして共用する | 画像用の別ファイル（`h3_image_vae.c`）にした。GPU 演算は動画デコーダと同じものを使う | 動画デコーダに手を入れないので、動画の回帰が起きない。P-H3（フェーズ 5）用の可変化はこのファイルで足りる |
| PDD ヘッドを `src/model/h3_pdd.c` に置く | DiT の読み込み（`load_step_heads`）に入れた | 40 行ほどで、DiT の内部状態に書く処理だった |
| ファミリーの能力は `H3_CAN_ATTENTION_CACHE \| H3_CAN_ADALN_CURVE` | 能力なし | int8 キャッシュは未対応。AdaLN の曲線はチェックポイント自身が持ち、外部の曲線ファイルは使わない |
| 画像用の単体試験 4 本を `make test` に足す | 実重みを使う検査プログラム 3 本を足した（`make test` の外）。`make test` はファミリー表の試験を更新 | レイアウトと位置は既存の `h3_layout_build` がそのまま画像の配置を出すので、新しいホスト側ロジックがほとんど無い |

### C.2 見つけて直した不具合

M5 では、パスに `/transformer/` を含む重みをファイルから直接マップして GPU バッファにする（`h3_gpu.m`）。Comfy-Org の画像 DiT は、長さ 37 バイトのメタデータ用テンソル（`comfy_attention.config`）が間に入るため、BF16・F16・F32 のテンソル 307 個が要素サイズの境界に揃っていない位置にある。これをそのままマップすると値が壊れ、DiT の出力がすべて非有限になった。同じファイルを別名のフォルダ（`diffusion_models/`）から開くとマップされないので、検査プログラムでは出ず、モデルフォルダ経由の通し実行で初めて出た。

修正: 要素サイズの境界に揃った位置のテンソルだけをマップし、ほかは従来どおり読み込む。動画モデルのファイル（手元の INT8 ConvRot 版）には揃っていないテンソルが無く、動作は変わらない。

### C.3 基準との比較（M5 24 GB）

**VAE**: 上流の最終潜在 8 組（2 プロンプト × 4 サイズ）を復号し、上流の PNG と比べた。PSNR 61.1〜62.7 dB、8 ビット値の最大差 3、95〜96% が完全一致。差は上流が半精度で計算している分。時間は 768² で 1.9 秒、1024² で 2.8 秒、2048² で 12.1 秒（上流は 4.0 秒。タイルを 4 枚ずつまとめていない分が遅い）。

**DiT、1 ステップずつ**（基準の「そのステップの前の潜在」を入力にした速度の比較）:

| 基準 | サイズ | 相対 L2（6 ステップ） | コサイン（最小） |
| - | - | - | - |
| 上流そのまま（`teapot`） | 768×768 | 1.4〜4.3% | 0.9991 |
| 上流・活性化の量子化なし（`teapot`） | 768×768 | 1.0〜4.2% | 0.9991 |
| 上流そのまま（`potter`） | 1024×1024 | 1.2〜6.2% | 0.9981 |

- 活性化を量子化しない上流と比べても差は縮まらない。差の主因は上流の W8A8 ではなく、エンジン側の計算（int8 から戻した重みと活性化を BF16 で持つ）と読める。動画 DiT の既存の許容（`tests/test_semantic_dit.c` の相対 L2 10% 未満）の内側。
- 位置、AdaLN の行、ステップ別ヘッドのどれかが違えば、この一致にはならない。

**DiT、6 ステップ通し**（基準の初期ノイズから）: 最終潜在の相対 L2 は 20〜40%。A.3 で見た増幅と同じ桁で、同じ場面の別の絵になる。目視では上流と優劣が付かない。

**時間とメモリ**（`h3_image_cli`、テキストは Q4_K_M）:

| サイズ | テキスト | DiT 6 ステップ | VAE | ピークメモリ |
| - | - | - | - | - |
| 768×768 | 7.5 s | 34.9 s（上流 41.7 s） | 2.1 s | — |
| 1024×1024 | 9.8 s | 48.1 s（上流 50.5 s） | 2.9 s | 7.7 GB |
| 1344×768（アプリ内） | — | 全体で約 60 s | — | — |

ピークはテキストエンコーダの段。DiT の段は 2.3 GB。

### C.4 アプリでの確認

隔離したホームで動かしたテスト用の複製に、API から操作した。

- 画像モデルのフォルダの登録。動画モデルのフォルダや存在しないフォルダは理由を返して登録しない。
- 生成（768×768、44.5 秒）: PNG、記録（プロンプト、seed、sigma 列、モデル名、潜在の記述）、潜在ファイル（safetensors、F32 [24,48,48]）が `images/` に入り、プレビューに出る。PNG の XMP に AI 生成の表示とプロンプトが入る。
- 生成中の中止: 2 秒で止まり、画像は保存されず、状態が戻る。続けて 2 枚の連続生成（1 枚目は固定 seed、2 枚目は乱数）が通る。
- 生成後の GPU 割り当ては 0.63 → 0.65 → 0.67 → 0.67 GiB で、増え続けない（動画モデルを読み込んだ状態の値）。
- 動画の回帰: フェーズ 1 の確認で作った動画（GGUF、SSD ストリーミング、256×256、4 ステップ、最初の画像つき）を同じ seed で作り直し、潜在ファイル（`.h3lt`）がバイト一致した。`make test` も通る。

### C.5 残したもの、確かめていないこと

- **参照画像つきの生成（編集）**。レイアウトと AdaLN の行は既存のもので足りる見込みだが、参照の VAE エンコード（posterior からのサンプル）と、`"Picture i: "` 形式の視覚ブロックが未実装。
- **速度**。DiT は SSD ストリーミングだけ。int8 キャッシュや常駐は未対応で、能力表にも入れていない。VAE のタイルはまとめていない。
- **メモリの事前判定**。大きさの上限は辺 4096 までの検査だけで、このMacで動くかの判定は無い。2048² は検査プログラムの VAE だけ確かめ、DiT は通していない。
- **公式の BF16 配布**（密な AdaLN、`pdd_fused_heads.safetensors`）。Comfy-Org の int8_convrot 版だけを動かした。GGUF の画像 DiT の判別も入れていない。
- **画像モードの seed の入力欄**と、進捗の実績による見積もり（`ProgressEstimator`、`TimingCalibration`）。残り時間は経過からの単純な外挿。
- **モデルの導入ウィザード**（フェーズ 3）。いまはフォルダを選んで登録する。モデル管理の画面に画像モデルは出ない。
- **画像の削除の実行**（付録 B.3 と同じ理由）、**リリースビルドでの確認**、**英語表示の確認**。
- **重みの指紋**。記録の `model` は名前だけ。
- テキスト埋め込みは動画モデルの GGUF を使うので、同じ seed でも上流とは別の絵になる（A.3）。初期ノイズも `h3_rng` で、上流（numpy）とは違う。

## 付録 D. フェーズ 3 の結果

### D.1 実装したもの（2026-11-15、ブランチ `image-phase3`）

**画像モデルの導入**（`ImageModelSetup.swift`）

- 取得するものと固定した版:

| ファイル | 取得元と版 | SHA-256 |
| - | - | - |
| DiT（int8 + ConvRot、20,979,314,802 バイト） | Comfy-Org/MiniMax-H3-Image `892de7cb…`（2026-11-10） | `83eb140f…6ee622` |
| VAE（fp16、1,168,505,312 バイト） | 同上 | `d80e9c2e…8721af` |
| `vae/config.json`、`tokenizer/tokenizer.json` | MiniMaxAI/MiniMax-H3-Image `492fd8c4…` | `7fcc21be…e3e4bb`、`a5d85b6d…173c7` |
| `LICENSE` | MiniMaxAI/MiniMax-H3-Image `5f904730…` | `0ec3bb5e…6d423b` |

- ライセンス本文は 11 月 11 日に 1 か所変わっている（許諾申請先の URL が `platform.minimax.io/license` から `/console/license` に）。付録 A.4 で読んだ最新版に固定した。
- テキストエンコーダは、導入済みの動画モデルの `FL2VA/text_encoder` を複製する。同じボリュームならクローンになり、容量は増えない（確認: 18.2 GB の複製の前後で空き容量が変わらない）。動画モデルが 1 つも無ければ、ダウンロードを始める前に理由を示して止める。
- 置き場所は `Application Support/H3cApp/ImageModels/MiniMax-H3-Image/`。重みの隣に `LICENSE` と `H3IMAGE_PROVENANCE.json`（版、各ファイルのハッシュ、照合日時）を書く。
- 導入画面は、契約が動画モデルと別であること、対象外地域（EU・英国・韓国・米国）が画像にも及ぶこと、年商 100 万 USD 超とホスティングでの事前許諾、商用での「MiniMax-H3-Image」表示、公開時の機械生成の明示、他モデルの学習・改良の禁止を挙げ、本文へのリンクを付ける。確認のチェックなしではダウンロードできない。
- 完了すると全ウィンドウの一覧に登録される（`ModelLibrary.registerImageModel`）。記録の `model` に取得元と版が入る。

**モデル管理**: 「画像モデル」の区分を足した。一覧、使用するモデルの切り替え、フォルダの追加（検査つき）、ダウンロードしたモデルの管理と削除。削除は画像モデルのフォルダだけを消す。テキストエンコーダは複製なので、動画モデルとの間に参照関係は無い。

**プロジェクト**: 画像を別のプロジェクトへ移す操作（記録と潜在ごと。移動先で名前が重なれば番号を付ける）。一覧の画像の「その他」と `/api/project/move-image`。

**ライセンス表記**: `THIRD_PARTY_NOTICES.md` に MiniMax H3 Image の節。「H3cAppについて」とヘルプメニューに画像モデルのライセンスへのリンク。

**API**: `/api/image-models/setup`、`/api/image-models/setup/cancel`、`GET /api/image-models` の `setup`、`/api/project/move-image`。

### D.2 実行の排他

- 動画の生成、画像の生成、int8 キャッシュの作成、RefMod の抽出は、プロセスに 1 本の直列キュー（`H3Engine.workQueue`）で動く。画像エンジンも同じキューを使うので、別のウィンドウや API から重ねて始めても同時には走らない。
- 同じウィンドウでは `isGenerating` を共用する。画像の生成中は動画の生成と、GPU を占有する拡大（SwiftVR、SeedVR2 など）が始まらず、拡大・取り込み・キャッシュ作成の最中は画像の生成が始まらない。
- 画像の生成が読み込むもの（テキストエンコーダ、DiT、VAE）は生成のたびに解放する。動画へ移るときに画像モデルの重みは残らない（生成後の GPU 割り当てが増えないことは付録 C.4）。
- 既存の制約がそのまま残る点: 別のウィンドウで動いている拡大と、このウィンドウの生成は直列化されていない（動画でも同じ）。

### D.3 設計書 §10.3 の項目

| 項目 | 状況 |
| - | - |
| 新規ウィンドウは動画で始まり、旧プロジェクトも動画として開ける | 検査で確認 |
| モードを往復しても各下書きとモデル選択が保持される | 検査で確認（下書き）。モデル選択は動画・画像で別に保存する |
| 画像だけの無題プロジェクトが再起動時に消えない | 検査で確認（強制終了後の起動） |
| ギャラリーから開始・終了・参照へ渡した内容とモデル表示が一致する | 検査で確認（フォームの内容）。実際に動画を生成して `sourceImage` も確認 |
| 元画像や生成元プロジェクトを削除しても、受け取り先動画の入力が失われない | 渡した画像は `references/` に記録ごと複製される（検査で確認）。元画像を移動しても複製は残る |
| 潜在欠損、破損、画像編集、サイズ不一致、未対応 VAE でのフォールバック | 潜在の直接利用（フェーズ 4）が未着手なので、いまは常に画素から入力する |
| キャンセル、失敗、容量不足の後に再生成でき、途中ファイルが完成品として残らない | 中止後の再生成を確認。画像は隠し名で書いてから改名する。容量不足は試していない |
| 複数ウィンドウ、API、動画生成、画像生成、他の重い GPU 処理が実行制約を守る | D.2。複数ウィンドウでの実走はしていない（直列キューの共用はコードで確認） |
| 既存の動画、連結、リップシンク、RefMods、アップスケール、モデル管理を壊さない | 動画の生成は潜在がバイト一致（付録 C.4）。`make test` 通過。連結、リップシンク、RefMods、拡大は実走していない |
| 生成記録から、生成時のモデルと設定を識別して復元できる | 記録にプロンプト、seed、sigma 列、サイズ、モデル名・取得元・版が入る。「設定を戻す」で画像のフォームに戻る |

### D.4 確認したこと

- 導入の通し: DiT は取得済みのものを置き、VAE（1.17 GB）を実際にダウンロードさせた。一覧の確認、ダウンロード、小さいファイル 3 つの取得、テキストエンコーダの複製、22 GB の SHA-256 照合、記録の書き込み、登録までが通り、導入したモデルで画像を生成できた（768×768、45.6 秒）。DiT 21 GB のダウンロード自体は既存のダウンローダの経路で、この確認では通していない。
- 確認のチェックなしの API 呼び出しは 400 を返す。
- 画像の移動: 画像、記録、潜在が移動先の `images/` に移り、元のプロジェクトから消える。プロジェクトでないフォルダは断る。
- `tools/image_phase1/roundtrip_check.py`: 47 項目すべて通過。`make test`、ローカライズ検査も通過。

### D.5 残したもの、確かめていないこと

- **動画モデルを持たない利用者**は画像モデルを導入できない（テキストエンコーダを単独で取得する経路が無い）。
- **取得元は Hugging Face だけ**。上流が既定にしている ModelScope は使っていない。
- **ライセンス V.5**（生成機能を第三者に提供する製品に求められる、利用制限の通知と通報の仕組み）を配布アプリとしてどう満たすかは未決のまま。導入画面とヘルプで条件を示すところまで。
- 連結、リップシンク、RefMods、拡大の回帰は実走していない。複数ウィンドウでの同時操作も実走していない。
- 画像の削除の実行、リリースビルドと英語表示での確認は、付録 B.3・C.5 と同じく未確認。
- フェーズ 2 から残っているもの（参照画像つきの生成、メモリの事前判定、seed の入力欄、公式 BF16 配布、速度）はそのまま。
