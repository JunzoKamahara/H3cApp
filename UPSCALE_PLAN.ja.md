# H3cApp 高解像度化（AI 動画超解像）計画

[English](UPSCALE_PLAN.md) | [日本語](UPSCALE_PLAN.ja.md)

この文書は、生成済み動画を生成的ディテール付きで拡大する「AI 動画超解像」機能の設計・実装計画です。
実装者（人または AI）が判断に迷わずに着手できるよう、決定事項・確認済みの事実・推定・未確認を区別して書いています。
作成 2026-10-28、対象は h3c-app v0.7.1 時点の main。

## 背景（なぜこの文書を書くか）

- 次のタスクは「高解像度化」。LightX2V 精査メモ（2026-10-28）の結論は
  「H3 エンジンは置き換えない。画質改善は **SwiftVR の ReAE 省メモリ化を取り込んだ独立アップスケーラー**、
  H3 側は LoRA 補正の別計算・必要に応じたメモリ管理」。
- 既存の拡大機能（v0.6.8–0.6.9）は Lanczos / MetalFX / Real-ESRGAN（CNN）までで、
  「SeedVR2 に近い生成的な質感」は出ない。SeedVR2 自体は 24 GB Mac では非現実的（メモリ34 GB、539 s/149fr@320x240）。
- 本文書は、実装者（人または AI）が **判断に迷わず実装に入れる** ことを目的に、
  決定事項・検証済み事実・推定・未確認を区別して書く。

## 決定事項（2026-10-28）

| 論点 | 決定 |
|---|---|
| モデル | **両対応の設計**。共通の「一段 Wan 系 VSR エンジン」を作り、まず SwiftVR、後で FlashVSR-1.1 Tiny を追加できる構造にする |
| 文書 | **日英ペアで新規作成**（`UPSCALE_PLAN.ja.md` / `UPSCALE_PLAN.md`） |
| 代替案（H3 自身の再ノイズ hires-fix） | **実験フェーズとして含める** |

---

## §0 読み方・前提

- 対象リポジトリ: `~/Documents/work/h3c-app`（MIT）。作業は **main から 1 タスク 1 ブランチ**、
  マージ後削除（このリポジトリの運用ルール）。コミット前に `git branch --show-current`。
- 記法: **[確認]** = ソースまたは公式資料で確認済み、**[推定]** = 計算や類推、**[未確認]** = 実装時に測る／確認する。
- 対象機: Apple Silicon 24 GB を基準。**M5（MTLGPUFamilyApple10, Metal 4 TensorOps）** では int8 NAX 経路、
  それ以外（M1–M4）は BF16 MPSGraph 経路で **動作必須**（速度は劣る）。
- 外部プログラム禁止（ffmpeg 等なし）。全て app 内（既存の拡大機能と同じ制約）。
- 重みは **同梱しない**。初回に利用者が HF から取得（ライセンス表示）、端末内で変換キャッシュ化。

## §1 背景・目的・ゴール

- ゴール: 生成済み動画（例 960x544, 24 fps, 5–90 s）を **2x（将来 4x）** に、生成的ディテール付きで、
  24 GB Mac 上の実用時間で拡大する。ユーザーの関心は「肌・髪・質感が SeedVR2 級」。
- 非ゴール: H3 生成エンジンの置換、学習、SeedVR2 の移植。
- **品質目標（第一）: SeedVR2（3B）の出力にできるだけ近づける。** SeedVR2 自体は 24 GB Mac で動かないので「SeedVR2 を目標品質の基準（gold reference）とし、本実装の出力をそれと比較して差を縮める」と定義する（測り方は §9.5）。
  目標値: 同じ入力集合で、無参照画質指標（MUSIQ / MANIQA / CLIP-IQA、動画は DOVER）の平均が **SeedVR2-3B の 95% 以上**、フリッカ指標が SeedVR2 以下、ユーザーのブラインド A/B で「SeedVR2 と区別できない or 同等」が過半 [目標値]。
  本実装（Metal 移植）と公式 PyTorch SwiftVR の差は §9 の parity 許容内に収める（移植で品質を落とさない）。
- 成功基準（受け入れ条件、§8 で再掲）:
  - 上の品質目標を §9.5 の手順で測り、結果（達成／未達と差の内訳）を文書に記録する。
  - 960x544→1920x1088、124 フレーム（5.17 s）が **M5 24 GB で 2 分以内 [推定目標]**、ピーク ≤ 10 GiB。
  - M4 24 GB でも完走（目標 ≤ 6 分、ピーク ≤ 14 GiB）。
  - PyTorch リファレンスと数値一致（§9 の許容値）。
  - 既存 UI の「拡大」の選択肢として追加され、既存 4 方式は不変。
- 品質と速度の優先順位: **品質優先モード（全注意・BF16・overlap あり）を必ず残し、速度モード（MFSWA・int8）は品質差を測ってから既定にする。**

## §2 候補モデルの調査結果（2026-10-28）

| 項目 | SwiftVR（第一） | FlashVSR-1.1 Tiny（第二） | （参考）SeedVR2 |
|---|---|---|---|
| 公式 | github H-oliday/SwiftVR, HF H-oliday/SwiftVR, arXiv 2606.09516 | github OpenImagingLab/FlashVSR, HF JunhaoZhuang/FlashVSR-v1.1 | ByteDance-Seed |
| 骨格 | Wan2.2-TI2V-5B（30 層, dim 3072, 24 head×128, FFN 14336, in/out 48ch, patch (1,2,2)）**[確認: transformer/config.json]** | Wan2.1-T2V-1.3B 形（30 層, dim 1536, 12 head×128, FFN 8960, in/out 16ch）**[確認: Wan2.1-1.3B config, DMD重み 5.68 GB=F32 1.42B]** | 3B/7B DiT |
| 推論 | **1 ステップ**、t=1000 固定、z_HQ = z_LQ − v **[確認]** | 1 ステップ、t=1000、ストリーミング KV キャッシュ、ノイズ入力＋LQ 特徴加算 **[確認]** | 多ステップ |
| 注意 | MFSWA: 16x16 窓（潜在トークン格子）、時間方向は全結合、偶数層非シフト／奇数層半窓シフト、窓はクランプしてマスク無し dense SDPA **[確認: 公式 transformer.py / LightX2V attention.py]** | 3D 窓 (2,8,8)、因果ブロックマスク、**LCSA = Q/K 窓平均の top-k ブロック選択**（局所範囲 9/11）、ブロックスパース核が必要 **[確認]** | 全注意 |
| AE | **ReAE** 40.95M params、16x 空間・4x 時間、48ch、TAEHV 系（MemBlock 状態で因果ストリーム）**[確認: 公式 reae.py]** | TCDecoder（TAEHV 系 [512,256,128,128]、潜在 16+768 で LQ 画素を条件連結）+ LQ_proj_in（Causal_LQ4x_Proj 576 MB） | 3D VAE |
| 重み | F32 20.0 GB + reae 164 MB + prompt_emb 4.2 MB | 5.68 GB + 0.58 GB + 0.19 GB（Wan2.1_VAE 0.5 GB は Tiny では不要）| — |
| ライセンス | **SwiftVR Research License（非商用・研究／個人のみ。商用は要書面許諾）**。コードは Apache-2.0。HF の古い版は Apache-2.0 表記だったが現行 README は Research License **[確認]** | **Apache-2.0**（コード・HF タグ） | Apache-2.0 |
| 公称性能 | H100 bf16 1920x1080: Enc 26 ms / DiT 328 ms / Dec 85 ms、24 fr、ピーク 29 GB（ReAE 非バッチ）**[確認: 論文 Table 5]** | A100 768x1408 ≈17 fps **[確認]** | H100 1440p 24fr 17.3 s |
| 4x 以外 | 論文は 4x 例のみ。入力は目標サイズへ bilinear 拡大してから復元するので **任意倍率（2x 含む）は機構的に可能、画質は要測定 [未確認]** | 「4x 専用」、入力は 128 の倍数へ中央クロップ | — |
| Apple 実装 | なし（MPS 未検証） | WaifuFinder の Swift/MLX 変換（BF16 3.5 GB、16 GB Mac で ROI 検証） | mlx-gen（要 ffmpeg） |

追加候補 **[未調査]**: FlashVSR+（2026-10 公開、github OpenImagingLab/FlashVSR_Plus、arXiv 2610.17552、「Hybrid Attention + RY-DMD」で品質改善を主張、学習済みモデル公開）。ライセンス・骨格・注意方式を P1 で確認し、FlashVSR Tiny の代わりに第二バックエンド候補とするか判断する。

**判断**: SwiftVR は品質・速度・実装の単純さ（固定条件・マスク無し注意・小 AE）で最有力。
SeedVR2 との品質差について、SwiftVR 論文は無参照指標で「強い」と主張するが **具体的な数値比較はこの調査では取れていない [未確認]** → §9.5 で自分で測る。
唯一の障害はライセンス → **同梱せず、利用者ダウンロード＋非商用注記**（既存の MiniMax H3 Community License 表示と同じ流儀）。
FlashVSR Tiny は許諾面で安全だが、データ依存のブロックスパース注意と KV キャッシュのため実装量が多い → 第二フェーズ。

## §3 既存コードの地図（実装者が最初に読む場所）

| 役割 | 場所 | 要点 |
|---|---|---|
| 拡大の入口・フレーム I/O | `native/H3Spike/Sources/H3cApp/VideoUpscale.swift` | `VideoUpscaler` enum（rawValue が API 名・記録値）、`UpscaleFactor` 1.5/2/3/4、**private `FrameScaler`**（BGRA8 1 枚→1 枚、同期、時間状態なし）、AVAssetReader(32BGRA, Metal 互換) → scale → AVAssetWriter(HEVC, 品質値)、音声・メタデータはパススルー、隠しファイルに書いて完了時置換、進捗 4 フレームごと、キャンセルはフレーム間 |
| CNN 超解像 | `RealESRGAN.swift` | MPSGraph F16、`graph.run` 毎タイル（executable 化なし）、タイル=コア 1024+文脈、F16 safetensors ローダ |
| TAEHV 系デコーダの既存実装 | `TinyPreviewDecoder.swift` | **ReAE と同じ層構成**（Clamp, conv, MemBlock, nearest 2x, TGrow, pixel shuffle）を MPSGraph で実装。`MPSGraphExecutable` を形状キーでキャッシュ、GPU 限定配置（ANE が F32 を拒否） |
| UI | `UpscaleSheet.swift`, `H3cApp.swift`(動画メニュー), `ProjectViews.swift`, `ContentView.swift`（進捗バナー） | `VideoUpscaler.allCases` を列挙。`isAvailable` で無効化 |
| HTTP API | `GenerationViewModel+API.swift` | `/api/project/upscale-video`, `/api/upscale-file`, `/api/upscale/cancel`、`upscaler`/`scale` 検証 |
| 記録 | `Projects.swift`(`upscaledFrom`,`upscaler`,`upscaledSize`,`upscaleFactor`), `Continuation.swift`(enlargement) | 未知の rawValue は nil/standard に落ちる |
| オンデマンド DL の既存例 | `DenoisePreview.swift`（taeh3 22.7 MB, URLSession）, `ModelDownloader.swift`（再開・分割、巨大 ckpt 向け）, `ModelImportDownload.swift`(`ModelLicenseNotice`) | 大容量 DL は `ModelDownloader` を流用 |
| エンジン GPU 基盤 | `h3_gpu.h/.m`, `h3_shaders.metal`(実行時コンパイル, `h3_gpu_create` のカーネル名リストに追加が必要), `h3_metal.m` | BF16/int8 線形（NAX `matmul2d`）、MPSGraph SDPA（batch 次元あり）、LayerNorm/RMSNorm/GELU、conv3d（channels-last, パディング別）、`stream_file_*`(F_NOCACHE)、mmap ゼロコピー |
| Transformer 駆動の見本 | `h3_dit.c` | QKV→SDPA→out、int8 経路、SSD 二重バッファ先読み pthread、`h3_attention_cache.c` の int8 キャッシュ＋フィンガープリント |
| 動画書き出し | `h3_av_writer.m`（HEVC/H.264, BT.709） | エンジンから直接書く場合の手段 |
| 変換ツール | `tools/convert_realesrgan.py`（numpy のみ・制限付き unpickler）、`h3_gguf.c`/`h3_quant.c` | 変換は **端末内 C** で行う方針（Python 配布不可） |
| ビルド | `Makefile`(`LIB_C`/`LIB_M`), `native/H3Spike/Package.swift`, `package_app.sh`, `BUILD.md`（`make test`、fixture 無しは skip） | |
| 文書 | `TECHNICAL(.ja).md`, `GUIDE(.ja).md`, `API(.ja).md`, `THIRD_PARTY_NOTICES.md`, `SPEEDUP_ROADMAP.md`, `docs/`(サイト) | 日英ペア、見出し数一致、計測は機種・設定・seed を明記 |

## §4 モデル仕様（実装に必要な全て）

### §4.1 SwiftVR DiT（Wan2.2-TI2V-5B 形）[確認: diffusers transformer_wan.py, LightX2V swiftvr/model.py, 公式 streaming/dit.py]

- 入力潜在 z: [48, T, H/16, W/16]（ReAE 出力、スケーリングなし）。パッチ (1,2,2) → トークン数 N = T·(H/32)·(W/32)、
  トークン = 48×2×2 = 192 次元の線形（Conv3d(48→3072, k=s=(1,2,2)) は 192→3072 の Linear）。
- 条件（**全て定数**）: timestep = 1000.0。
  `temb = Linear(SiLU(Linear(sinusoidal_256(1000))))`、`block_mod = Linear(SiLU(temb))` → 6×3072。
  テキスト: `prompt_emb[1,512,4096]` → Linear(4096→3072) → GELU(tanh) → Linear(3072→3072) = context[512,3072]。
- ブロック（30 層、各 `scale_shift_table[1,6,3072]`）:
  1. `shift,scale,gate,c_shift,c_scale,c_gate = scale_shift_table + block_mod`（F32）
  2. `h = LN_noaffine(x)·(1+scale)+shift` → 自己注意: q=RMSNorm_h(Wq h), k=RMSNorm_h(Wk h), v=Wv h（RMSNorm は head 全体 3072 次元で 1 本 = "rms_norm_across_heads"）、3D RoPE（t:44, h:42, w:42 次元、θ=10000、実数 interleave 形式）、MFSWA、Wo → `x += attn·gate`
  3. `h = LN_affine(x)`（norm2, cross_attn_norm=true）→ 交差注意: q=RMSNorm(Wq h), k=RMSNorm(Wk ctx), v=Wv ctx（**K,V は定数 → 事前計算**）、Wo → `x += attn`
  4. `h = LN_noaffine(x)·(1+c_scale)+c_shift` → FFN Linear(3072→14336) GELU(tanh) Linear(14336→3072) → `x += ff·c_gate`
- 出力: `LN_noaffine(x)·(1+scale_out)+shift_out`（`scale_shift_table[1,2,3072]+temb`）→ `proj_out` Linear(3072→192) → unpatchify → v [48,T,H/16,W/16]。
- 一段推論: **z_HQ = z_LQ − v**（論文 Eq.5; LightX2V `latents = model_input - predict(...)`）。乱数なし・seed なし（決定的）。
- 入力前処理: LQ 動画を **目標サイズへ bilinear（align_corners=False, antialias なし）で拡大** → [0,1] → 右・下を 0 で **32 の倍数にパディング** → ReAE。出力は目標サイズに切り出し（偶数丸め）。
- MFSWA（窓 (16,16) 潜在トークン、時間は全結合）:
  - 軸ごとの窓開始位置（LightX2V `axis_window_starts`）: `size ≤ window` なら {0}。それ以外 `starts = clamp(k·window − shift, 0, size−window)` for k=0..ceil(size/window)+1、重複除去、
    さらに中間要素は `starts[i+1] > starts[i−1] + window` のときのみ保持。shift = 0（偶数層）/ window/2（奇数層）。
  - 例 1920x1088 → 格子 34x60: 非シフト H:{0,16,18} W:{0,16,32,44} → 12 窓、シフト H:{0,8,18} W:{0,8,24,40,44} → 15 窓。窓長 = T·256（T=6 で 1536）。
  - 出力の所有者（owner）: 各トークンの出力は 1 つの窓から取る。LightX2V: 窓を順に走査して `owner[idx] = window*len+local` を上書き（非シフト層は降順、シフト層は昇順に走査）。
    公式実装（`WanShiftWindow2DInferProcessor`, `prefer_front = not do_shift`）と同義。**両実装を parity で確認 [未確認]**。
  - 論文 Table 3（1440p）: 全注意 PSNR 25.86 / MFSWA 25.58 / マスク付き SWA 25.34。**MFSWA は推論時近似**なので、メモリが許す場合は全注意（dense SDPA）を「高品質」オプションとして残す。
- RoPE 位置: 時間オフセット = `chunk.start // 4`（チャンク間で位置を連続させる）、空間は 0 起点。表は必要長まで拡張（LightX2V `ROPE_EXTENSION=256`）。

### §4.2 ReAE [確認: 公式 swiftvr/models/reae.py = LightX2V reae.py]

- Encoder（入力 RGB [0,1], NTCHW）: `pixel_unshuffle(2)`→12ch; conv3x3(12→64)+ReLU; TPool(64,2)[1x1 conv on 2 frames stacked]; conv3x3 s2 (no bias); MemBlock×3; TPool(64,2); conv s2; MemBlock×3; TPool(64,1); conv s2; MemBlock×3; conv3x3(64→48)。空間 16x、時間 4x。
  最終チャンクは末尾フレームを 3 枚複製して 4k+1 に揃える（`is_last`）。
- MemBlock(x, past): `ReLU(conv3(ReLU(conv3(ReLU(conv3(cat_ch[x, past]))))) + skip(x))`、skip は同チャネルなら identity。past = 同ブロックの **前フレーム入力**（最初は 0、チャンク間で状態として持ち越し）。
- Decoder: Clamp(`tanh(x/3)·3`); conv(48→512)+ReLU; MemBlock×3(512); Upsample2x; TGrow(512,1)=1x1 conv; conv(512→256, no bias); MemBlock×3(256); Upsample2x; TGrow(256,2); conv(256→128); MemBlock×3(128); Upsample2x; TGrow(128,2); conv(128→64); ReLU; conv(64→12); `pixel_shuffle(2)`; clamp[0,1]。
  **TGrow(stride 2) = 時間 nearest ×2 → Conv3d(C→C, kernel (3,1,1), pad (1,0,0), no bias)**（TAEHV 原版の 1x1 conv とは異なる。`taeh3` 用 `TinyPreviewDecoder` の TGrow をそのまま使うと誤り）。
- `frames_to_trim = 3`: 最初のチャンクの先頭 3 出力フレームを捨てる。
- 省メモリ（LightX2V, 等価変換）: (a) TGrow は空間的に点ごとなので **nearest 空間アップサンプル前**に実行（1/4 画素）。
  (b) MemBlock/TPool 以外はフレーム独立 → `reae_frame_batch_size` 枚ずつ連続層をまとめて通し、全フレーム分の中間を持たない。
  (c) デコーダは最後の TGrow 以降（フル解像度の conv 2 本 + pixel shuffle）を「出力層」として分離し、`output_batch_size` 枚ずつ流す。
- チャンク規則（LightX2V streaming.py）: `clip_len=24`（4 の倍数）、frame_count ≤ clip_len+4 なら 1 チャンク、
  それ以外 先頭 clip_len+4、以後 clip_len、最後は残り。`latent_count=(frames−1)//4+1`。
  出力範囲 `[max(0,start−3), start+frames−(last?0:3))`。最終チャンクで潜在が不足する分は前チャンクの潜在で前詰め。

### §4.3 FlashVSR-1.1 Tiny（第二フェーズ、要点）[確認: diffsynth/pipelines/flashvsr_tiny.py, models/wan_video_dit.py, examples/WanVSR/utils]
- DiT: Wan2.1-1.3B 形。入力は **ノイズ**（seed 付き）、LQ は `Causal_LQ4x_Proj`（PixelShuffle3d(4,8,8)→CausalConv3d(4,3,3)s(2,1,1)×2→Linear(→1536)×layer_num=1）の出力をブロック 0 の前に加算。
  交差注意 K/V は固定 prompt（`posi_prompt.pth`）から事前計算（`init_cross_kv`）。timestep 1000、1 ステップ、`x − v`。
- ストリーム: 初回 6 潜在フレーム、以後 2 潜在フレームずつ、ブロックごとに K/V 窓ブロックを `kv_ratio=3` 分キャッシュ。
- 注意: 窓 (2,8,8) 分割、`local_range` 9/11 の局所ブロックマスク＋`topk_ratio = sparse_ratio·768·1280/(H·W)` の **データ依存 top-k**（Q/K 窓平均の softmax）→ ブロックスパース注意。Metal では「選択ブロックの gather → dense SDPA」で等価実装可能だが、per-窓・per-ヘッドの選択が異なるため gather 量が大きい [推定]。
- デコーダ: TCDecoder = TAEHV [512,256,128,128] に「各 ReLU 後に 3x3 恒等初期化 conv」を挿入、潜在 16+768（LQ 画素 PixelShuffle3d(4,8,8) 連結）。出力後に wavelet/AdaIN 色補正。
- 制約: 4x、128 の倍数へ中央クロップ、フレーム 8n+1。
- **設計上の含意**: 共通エンジンは (i) Wan ブロック（dim/heads/ffn/in_ch をパラメタ化）、(ii) TAEHV 系デコーダ（層リスト駆動）、(iii) チャンクストリーマ（状態の持ち越し）、(iv) 注意バックエンド（dense / 窓 gather / ブロック選択 gather）を分離する。

## §5 数理・アルゴリズム上の工夫（効果とその根拠）

1. **一段推論の代数**: `z_HQ = z_LQ − v_θ(z_LQ, t=1)`。スケジューラ不要、乱数不要、チャンク毎に DiT 1 回。
2. **定数条件の畳み込み（計算の前倒し）** [確認: 条件が定数]:
   - 30 ブロック×6 本の変調ベクトルを事前計算し、`(1+scale)`、`shift`、`gate` を F32 定数として保持（Wan2.2 の時間埋め込み MLP とテキスト MLP は実行時に一度も走らない）。
   - 交差注意の K,V（RMSNorm 済み）を層ごとに事前計算: 30×512×3072×2×2 B ≈ **189 MB BF16**。交差注意は Q 射影＋512 キーの SDPA＋Wo だけ（線形 2 本分、全体の 1/8 未満）。
   - LayerNorm → 変調は 1 カーネルに融合（既存 `adaln*_bf16` の流儀）。
3. **MFSWA のコスト**（1920x1088, T=6, N=12,240 トークン, 24 head）[推定: 式から]:
   - 全注意: QKᵀ+PV = 2·N²·128·24 MAC ≈ 0.92 TFLOP/層 → 27.6 TFLOP/チャンク。素朴にスコアを実体化すると N²·24·2 B ≈ 7.2 GB。
   - MFSWA: 窓長 1536、12/15 窓 → 0.35/0.43 TFLOP/層 → 約 11.7 TFLOP/チャンク（**約 2.4x 削減**、メモリは窓長² で有界）。
     gather は Q/K/V 各 1.5–1.9x のコピー（≤ 3×23,040×3072×2 B ≈ 425 MB）。
   - 線形層: QKV+O 4·3072² + FFN 2·3072·14336 + 交差 Q,O 2·3072² = 144.7 M MAC/トークン → **3.54 TFLOP/層、106 TFLOP/チャンク**。注意は線形の約 1/9（MFSWA 時）。
     → 高速化の主戦場は **線形層（int8 NAX）**、注意は MPSGraph SDPA の batch 次元で窓を並列実行すれば十分。
4. **int8 行ごと量子化（NAX 経路）**: 既存 `h3_gpu_quantize_weight_int8`／`linear_int8_bf16`（活性は行ごとオンザフライ量子化）。
   形状制約 in%128, out%128: 3072 ✓、14336=112·128 ✓、`proj_out` の 192 ✗ と patch_embed の in=192 ✗ → この 2 本は BF16（NAX BF16 は in%32, out%64 なので 192 ✓）。
   既存の実測: DiT で int8 線形は BF16 比約 2x、ccv int8 注意の per-block PSNR 52–62 dB。SwiftVR では **ブロック毎 PSNR ≥ 40 dB、最終 RGB で BF16 比 ≥ 45 dB** を合格線に置く [未確認: 要測定]。
5. **ReAE の等価省メモリ変換**: TGrow（点ごと）と nearest 空間アップサンプルの可換性（厳密に等価）、フレーム独立層のフレームバッチ処理、MemBlock 状態（各ブロック最後のフレーム入力）のみ持ち越し → 動画長に対して O(1) メモリ。
6. **前処理の数値一致**: bilinear（align_corners=False, antialias なし）を GPU で再実装（MPSGraph `resize` bilinear はモードを確認 [未確認]、不一致なら自前 Metal カーネル）。ゼロパディングは右・下。
7. **RoPE**: `freqs_cos/sin` を (t_off, T, H', W') ごとに F32 で計算して BF16 表に落とし、interleave 形式（`x1·cos − x2·sin`, `x1·sin + x2·cos` を隣接ペアに）で適用。H3 の `rope_text_bf16` と形式が違えば新カーネル。
8. **ゼロコピーと SSD**: 重みキャッシュは `h3_gpu_load_bf16`（mmap, M5）／`stream_file_*`（F_NOCACHE）を流用。常駐 int8 5 GB なら読み直し不要（§6 で決定）。
9. **決定性**: SwiftVR は乱数を使わないので同一入力→同一出力（`h3_determinism_tests` と同じ流儀でテスト可能）。

## §6 エンジン設計（C / Metal / NAX / ANE）

### §6.0 設計の前提となる既存エンジンの事実 [確認: file:line]

| 事実 | 場所 |
|---|---|
| `h3_gpu_create` ごとに独自の `MTLCommandQueue` と MPSGraph キャッシュ → **2 コンテキストを 2 スレッドで並走可能** | `h3_gpu.m:344-357` |
| カーネルは `names` 配列に列挙が必須（TensorOps 系は `tensorOpsEnabled` 分岐内） | `h3_gpu.m:437-512` |
| **`h3_gpu_linear_bf16` の NAX 経路は `tensorOpsMode==1` か H3 形状（7168/5376/21504）のみ** → 3072/9216/14336 は黙って MPSGraph に落ちる。新 setter が必要 | `h3_gpu.m:408-413, 3146-3160` |
| NAX BF16 線形: bias なし, rows≥128, in%32, out%64, 重み `[out,in]` | `h3_gpu.m:3161-3163`, `h3_shaders.metal:1165-1195` |
| int8 線形: rows≥128, in%128, out%128, 行ごと F32 スケール, 活性は行ごと動的量子化（行を 128 にパディング）, BF16 出力, bias なし。K ループは 128 幅タイルの `multiply_accumulate` | `h3_gpu.m:3643-3706`, `h3_shaders.metal:2597-2650` |
| MPSGraph SDPA グラフは `batch` 次元を持つが、BF16 ラッパは batch=1 固定・q/k/v 同要素数 | `h3_gpu.m:1623-1700, 1704-1718, 1770-1776` |
| `h3_gpu_layer_norm_bf16` は F32 演算で `fma(norm, weight, bias)` → **weight=(1+scale), bias=shift とすれば変調 LayerNorm と同一** | `h3_shaders.metal:3159-3203` |
| `h3_gelu_bf16` に tanh 近似フラグあり（bias なし） | `h3_shaders.metal:3219-3243` |
| 既存 RoPE は half-split（d と d+half をペア）、Wan は interleave（2i, 2i+1）→ **新カーネル** | `h3_shaders.metal:3769-3836` |
| 既存 `mlp_*` は SwiGLU 専用 → GELU FFN には使えない | `h3_gpu.m:3297-3435` |
| conv3d は F32・channels-last・パディング別・MPSGraph | `h3_gpu.m:2380-2480` |
| `h3_gpu_dequantize` は F32/F16/BF16/GGUF の生バイト → BF16/F32（ビルダで利用） | `h3_gpu.h:169-173` |
| フィンガープリント = シャード名・サイズ・mtime の FNV-1a 32 B | `h3_weights.c h3_weight_store_fingerprint` |
| アテンションキャッシュ: 64 B ヘッダ `H3AC` v3、`.partial` → rename、失敗時 unlink | `h3_attention_cache.c:7-23, 54-64, 135-220` |
| `h3_av_reader` は全動画一括デコードのみ（逐次なし）→ 読み込みは Swift の AVAssetReader を使う | `h3_av_reader.h` |
| 生成中の拡大はガードされていない（逆はある） → VSR は `isGenerating` もガード | `GenerationViewModel.swift:785`, `VideoUpscale.swift:546,600,745` |

### §6.1 ファイル構成

エンジン（`Makefile` の `LIB_C`/`LIB_M` に追加）:

| ファイル | 役割 |
|---|---|
| `h3_vsr.h` | 公開 C API（§6.2）。`h3.h` 末尾で `#include "h3_vsr.h"`（CH3 モジュールは `h3.h` のシンボリックリンク 1 本なので追加不要） |
| `h3_vsr.c` | モデル非依存ドライバ: `h3_vsr`（重み、DiT 用と ReAE 用の 2 つの `h3_gpu`）、`h3_vsr_session`（チャンク計画・持ち越し状態・1 チャンク遅延パイプライン・進捗/キャンセル）、要件計算、`h3_vsr_model_ops` vtable |
| `h3_vsr_chunk.c` | チャンク規約（28 / 24 / 4b+1 パディング / 先頭 3 捨て / overlap）。純 C、単体テスト |
| `h3_vsr_window.c` | MFSWA 幾何（軸開始位置、窓一覧、gather 索引 `[Nw][Lw]`、owner 写像、**owner 限定クエリのクラス分け**）、RoPE 表（double で計算）、patchify 索引。純 C、単体テスト |
| `h3_vsr_wan.c` | Wan ブロック実行器（`h3_vsr_wan_config` で dim/heads/ffn/text 長を可変）。SwiftVR=MFSWA、FlashVSR=LCSA（第二フェーズ）を同じ gather/scatter 核で |
| `h3_vsr_reae.c` | TAEHV 系 CNN インタプリタ（層表駆動: CONV3x3, CONV3x3_S2, MEMBLOCK, TPOOL, TGROW, UP2_FOLDED_CONV, CLAMP, RELU, PIXEL_(UN)SHUFFLE）、MemBlock/TPool 状態持ち越し、フレームバッチ。表は ReAE enc/dec と（第二）TCDecoder・LQ_proj_in |
| `h3_vsr_weights.c` | 端末内キャッシュビルダ（`.h3vc`）、ローダ、フィンガープリント、定数事前計算 |
| `h3_vsr_swiftvr.c` / `h3_vsr_flashvsr.c` | モデル固有（ファイル名、config.json、重み名写像、前後処理、FlashVSR の色補正） |
| `h3_gpu.m/.h` 追加 | `h3_gpu_set_nax_generic`、`h3_gpu_sdpa_batched_bf16`、`h3_gpu_conv2d_bf16_mps`（M4 用）、新カーネルのディスパッチ |
| `h3_shaders.metal` 追加 | §6.5 の `h3_vsr_*` カーネル（NAX 系は tensorOps 分岐内で登録） |
| `tests/test_vsr.c`（常時）, `tests/test_real_vsr.c`（重み・fixture 必須、skip ガード）, `tests/bench_vsr.c` | §9。Makefile 目標 `h3_vsr_tests`, `h3_real_vsr_test`, `h3_vsr_bench` を `test:` に既存の `@if test -f` 流儀で追加 |
| `tools/dump_swiftvr_fixture.py` | 開発専用の fixture ダンプ（torch + 公式 repo） |

Swift:

| ファイル | 変更 |
|---|---|
| `Sources/H3Engine/H3VSR.swift`（新） | `H3VideoSuperResolver`: open/close、`requirements`、`processChunk`、`buildCache`（`H3Engine.swift:710-745` の `buildAttentionCache` ブリッジを複製）、`CancelFlag` 流用 |
| `VideoUpscale.swift` | `private protocol ChunkedFrameScaler` を `FrameScaler` の隣に追加。`PerFrameChunkAdapter`（既存 scaler をチャンク 1 で包む）と `VSRChunkScaler`。`upscale()` は 3 段パイプライン（§6.6）に置換し **全方式がそれを通る**（既存方式はアダプタ経由、出力はバイト一致）。`VideoUpscaler` に `.swiftVR`（第二で `.flashVSRTiny`）。`UpscaleFactor.fits` に `h3_vsr_requirements` の画素予算 |
| `UpscaleSheet.swift` | 推定時間（`estimated_seconds_per_chunk × chunks`）、「研究・非商用」バッジ、重み／キャッシュ未整備時の「モデルを準備」ボタン |
| `ModelManagerView.swift` / `ModelLibrary.swift` / `ModelDownloader.swift` | 「拡大モデル」節: HF から DL（`ModelDownloader` 再開対応）、ライセンス同意（日時を `ModelLibrary` に保存）、キャッシュ生成の進捗、精度選択（M5=int8 既定／他=BF16） |
| `GenerationViewModel+API.swift:68-107` | `"upscaler":"swiftVR"` を利用可能かつ同意済みのときのみ受理、生成中は 409 |
| `GenerationViewModel.swift` | VSR 系は `isGenerating` もガード |
| `Packaging/{en,ja}.lproj/Localizable.strings` | 新規文字列（`check_localizations.sh`） |
| `THIRD_PARTY_NOTICES.md`, `.gitignore`（`*.h3vc`） | SwiftVR（コード Apache-2.0／重み Research License・非同梱）、TAEHV MIT、FlashVSR Apache-2.0 |

### §6.2 公開 C API（`h3_vsr.h`）

```c
typedef struct h3_vsr h3_vsr;                  /* 読み込んだモデル。動画をまたいで再利用 */
typedef struct h3_vsr_session h3_vsr_session;  /* 1 本の動画: 因果状態を持つ */

typedef enum { H3_VSR_SWIFTVR = 1, H3_VSR_FLASHVSR_TINY = 2 } h3_vsr_kind;
typedef enum { H3_VSR_PRECISION_AUTO = 0, H3_VSR_PRECISION_BF16 = 1, H3_VSR_PRECISION_INT8 = 2 } h3_vsr_precision;
typedef enum { H3_VSR_BGRA8 = 0, H3_VSR_RGB24 = 1 } h3_vsr_pixel_format;

typedef struct {
    h3_vsr_kind kind; char name[64];
    int fixed_factor_x100;            /* 0 = 任意の出力サイズ(SwiftVR), 400 = 4x のみ(FlashVSR) */
    int output_multiple;              /* 32 (SwiftVR: 内部 pad+crop), 128 (FlashVSR: crop) */
    int min_output_side, max_output_side;
    int first_chunk_frames, chunk_frames;   /* 28, 24 */
    uint64_t weight_bytes_bf16, weight_bytes_int8;
    int int8_supported;               /* h3_gpu_is_m5() */
    char license_id[64];              /* "SwiftVR-Research-License" / "Apache-2.0" */
    int commercial_use_allowed;
} h3_vsr_info;
int  h3_vsr_inspect_dir(const char *model_dir, h3_vsr_info *info, char *error, size_t error_size);

/* 重みキャッシュ */
int  h3_vsr_cache_path(const char *model_dir, h3_vsr_precision p, char *path, size_t size);
int  h3_vsr_cache_matches(const char *cache_path, const char *model_dir);  /* ヘッダ+fingerprint のみ */
int  h3_vsr_build_cache(const char *model_dir, const char *cache_path, h3_vsr_precision p,
                        const char *shader_source_path, h3_progress_callback on_progress,
                        void *opaque, char *error, size_t error_size);     /* phase "vsr cache", n/30 */

typedef struct {
    h3_vsr_precision precision;   /* AUTO: tensor HW なら INT8、他は BF16 */
    const char *cache_path;       /* 必須。生 F32 safetensors は直接走らせない */
    const char *shader_source_path;
    int dit_overlap;              /* 前チャンクから前置する潜在数。0 = 公式 restore_video と同じ */
    int reae_frame_batch;         /* 0 = メモリから自動 */
    int reae_separate_queue;      /* 1: ReAE を別 h3_gpu + スレッドで（既定 1） */
    int keep_fc2_bf16;            /* int8 FC2 が parity を落とした場合の品質フォールバック */
    int full_attention;           /* 1: MFSWA ではなく全注意（品質優先モード） */
} h3_vsr_open_params;
h3_vsr *h3_vsr_open(const char *model_dir, const h3_vsr_open_params *params,
                    h3_progress_callback on_progress, void *opaque, char *error, size_t error_size);
void    h3_vsr_close(h3_vsr *vsr);

typedef struct {
    int supported; char reason[256];
    uint64_t gpu_bytes, host_bytes;
    double estimated_seconds_per_chunk;   /* bench の較正値をトークン数で比例 */
    int tokens_per_chunk, windows_even, windows_odd;
} h3_vsr_requirements;
int h3_vsr_check_requirements(const h3_vsr *vsr, int in_w, int in_h, int out_w, int out_h, h3_vsr_requirements *out);

h3_vsr_session *h3_vsr_session_open(h3_vsr *vsr, int in_w, int in_h, int out_w, int out_h,
                                    int total_frames /* 0 = 不明 */, char *error, size_t error_size);
int  h3_vsr_session_next_chunk_frames(const h3_vsr_session *s);   /* 28, 以後 24 */

typedef struct { const uint8_t *pixels; int width, height, stride_bytes; h3_vsr_pixel_format format; } h3_vsr_image;
typedef struct { uint8_t *pixels; int width, height, stride_bytes; h3_vsr_pixel_format format; } h3_vsr_image_out;

/* 入力チャンクを 1 つ渡し、完成した出力フレームを 0..N 枚書く（順序保持、1 チャンク遅延、
 * is_last で全て吐き出す → 総出力数 == 総入力数）。進捗 phase: "vsr encode", "vsr dit"(block/30),
 * "vsr decode"。callback の非 0 または *cancel で次のブロック/段階境界で中断（error "vsr cancelled"） */
int  h3_vsr_session_process(h3_vsr_session *s, const h3_vsr_image *in, int in_frames, int is_last,
                            h3_vsr_image_out *out, int out_capacity, int *out_frames,
                            h3_progress_callback on_progress, void *opaque,
                            const volatile int *cancel_requested, char *error, size_t error_size);
int  h3_vsr_session_pending_frames(const h3_vsr_session *s);
void h3_vsr_session_close(h3_vsr_session *s);
int  h3_vsr_get_stats(const h3_vsr *vsr, double *dit_seconds, double *reae_seconds, uint64_t *peak_live_bytes);
```

出力枚数規則（SwiftVR）: 1 回目（28 枚入力）→ 0 枚、2 回目（24）→ チャンク 1 の 25 枚、以後 24 枚ずつ、`is_last` → 前チャンクの残り＋最終チャンク（内部で 4b+1 にパディング、エンコーダ末尾 3 複製、4b+4 デコード、実枚数に切り出し）。

### §6.3 重みパイプライン（端末内変換・キャッシュ）

- 入力: `reae.safetensors`（**F32 128 テンソル** [確認: ヘッダ]）、`prompt_embedding.safetensors`（`prompt_emb` **BF16 [1,512,4096]** [確認]）、`transformer/diffusion_pytorch_model.safetensors`（**F32 825 テンソル、全線形に bias あり、`attn1/attn2.norm_q/norm_k.weight [3072]` あり** [確認: ヘッダ]）、`transformer/config.json`。
- `h3_vsr_build_cache` は `h3_build_attention_cache` と同形（自前 `h3_gpu`、`h3_weight_store`、ブロック毎進捗、`.partial`→rename、失敗時 unlink）。テンソル毎に `stream_file_i8`（F_NOCACHE で生 F32）→ `h3_gpu_dequantize(F32→BF16)`（Q,K,V は連続行域に読んで `[9216,3072]` に融合）→ int8 なら `quantize_weight_int8` → 読み戻して `fwrite`。20 GB の読み出しが支配的 → **1–2 分 [推定]**、ピーク RSS 数百 MB。
- テンソル別精度:

| テンソル | 保存 | 理由 |
|---|---|---|
| `attn1.to_{q,k,v}` 融合 `[9216,3072]`、`attn1.to_out.0`、`attn2.to_q`、`attn2.to_out.0`、`ffn.net.0.proj [14336,3072]`、`ffn.net.2 [3072,14336]` | int8 行量子化 + F32 スケール（M5）／BF16（M4・品質モード） | 3072=24·128, 9216=72·128, 14336=112·128 で int8 制約を満たす |
| 全線形の bias | F32 | F32 エピローグで加算 |
| `attn1/attn2.norm_q/norm_k.weight` | F32 | head 横断 RMSNorm |
| `norm2.weight/bias` | F32（+BF16 複製） | 交差注意前の affine LN |
| `blocks.b.mod` = `scale_shift_table[6,3072] + time_proj` [6,D]（shift, scale, gate, c_shift, c_scale, c_gate の生値。`(1+scale)` はロード時に作る） | F32 | t=1000 固定 |
| `attn2.to_k/to_v` は保存せず **`K2_b = RMSNorm_k(ctx·Wk+bk)`, `V2_b = ctx·Wv+bv` `[512,3072]`** | BF16 | 固定プロンプト。6.3 MB/層、計 189 MB |
| `patch_embedding [3072,48,1,2,2]` → `[3072,192]` + bias | BF16 + F32 | |
| `norm_out` 変調（`scale_shift_table[2,3072]+temb`） | F32 | |
| `proj_out [192,3072]` + bias | BF16 + F32（bias は unpatchify に畳む） | |
| 条件埋め込み（time 256→3072→3072、time_proj 3072→18432、text 4096→3072→3072） | **保存しない**。ビルド時に CPU で double/F32 評価 | 出力が上の定数 |
| ReAE 40.95M | **キャッシュに入れない**: ロード時に `reae.safetensors`（F32 164 MB）を直接読み、必要な精度に変換。ヘッダに fingerprint のみ記録 | 小さいので変換不要。F32 のまま使う品質モードも可能 |

- 定数評価（CPU、ビルド時）: `temb = L2(SiLU(L1(sinusoidal_256(1000.0))))`（diffusers `get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)`）、`time_proj = L(SiLU(temb))`→6×3072、`mod_b = scale_shift_table_b + time_proj`、`ctx = L(GELU_tanh(L(prompt_emb)))`、`K2_b/V2_b`。**harness で block-0 と突き合わせて attn2 の norm_q/norm_k 適用を確定**。
- キャッシュ形式 `swiftvr-dit-{int8|bf16}.h3vc`（モデルディレクトリ隣、`.h3ac` と同様 gitignore）: 128 B ヘッダ `{magic "H3VC", version, model_kind, precision, blocks, dim, ffn, heads, head_dim, text_tokens, patch_in, latent_channels, fingerprint[32](transformer/), reae_fingerprint[32](reae+prompt), table_offset, table_count}`、名前表 `{name[96], dtype, rows, cols, offset, bytes}`（64 B 整列、`stream_file_*` で直接 pread）、ブロック順ペイロード。名前表にしたのは 2 モデルを 1 形式で扱うため。
- 容量: ブロックあたり 144.7M params → int8 ≈ 30×(144.7+0.13+6.3+0.3) MB ≈ **4.55 GB**、BF16 ≈ **8.9 GB**、ReAE 82 MB。DL 原本 20.2 GB は変換成功後に「原本を削除」可（既定: 保持）。
- 常駐 vs SSD: **全対象で常駐が既定**。M5 int8: 4.6 GB + DiT 活性 ≈1.8 GB + ReAE ≈2 GB + ホスト 0.6 GB + MPS 作業域 ≈1 GB ≈ **10 GB**。M4 BF16 ≈ **14.3 GB**。SSD 二重スロット（`h3_dit.c:894-980`）は 16 GB 機向けに将来追加（API に余地）。4.6 GB の pread ≈ 2–3 s。

### §6.4 チャンク計算（960×544 → 1920×1088、24 fr、T=6）[推定: 式から]

- 幾何: 出力 1920×1088（32 の倍数、パディング不要）→ 潜在 120×68×48 → トークン格子 W=60, H=34 → 2040/潜在フレーム → **N = 12,240**（先頭チャンク T=7 → 14,280）。
- 窓: H=34 偶数層 `{0,16,18}`、奇数層 `{0,8,18}`；W=60 偶数 `{0,16,32,44}`、奇数 `{0,8,24,40,44}` → **Nw = 12 / 15、Lw = 1536**。gather 行数 18,432（1.51N）/ 23,040（1.88N）。owner 幅: 偶数層 h 16/16/2, w 16/16/16/12；奇数層 h 8/10/16, w 8/16/16/4/16。
- FLOPs/ブロック: QKV 0.693 T、out 0.231 T、attn2 q 0.231 T、attn2 out 0.231 T、FFN 2.156 T → **線形 3.54 TFLOP**。自己注意: 全窓 0.348/0.435 T、**owner 限定クエリ（厳密）0.231 T**、dense なら 1.84 T。交差注意 0.077 T。gather/scatter ≈0.65 GB、要素演算 ≈0.8 GB。ブロック ≈3.85 TFLOP、**チャンク ≈116 TFLOP**（H100 328 ms ⇔ 353 TFLOP/s 相当で整合）。
- ReAE/チャンク: encoder ≈1.9 T、decoder 素朴 15.0 T → **アップサンプル折り畳みで 12.4 T**。計 ≈**14.5 TFLOP**（DiT の 12.5%）。
- 時間（前提: NAX BF16 14 TFLOP/s 実測、int8 ≈ 2× [要 bench]、MPSGraph SDPA 11 TFLOP/s、NAX tapconv ≈10 TFLOP/s、M4 Pro MPSGraph ≈6 TFLOP/s・F16 conv ≈5 TFLOP/s **[未測]**）:

| | ブロック | DiT×30 | ReAE | **中間チャンク** | 5 s/124 fr（6 チャンク: 28+4×24+1） | 60 s/1440 fr（60 チャンク） |
|---|---|---|---|---|---|---|
| M5 int8 | 127 lin + 21 attn + 7 cross + 10 ≈ 165 ms | 5.0 s | 1.45 s | **≈6.5 s** | ≈34 s | ≈6.5 min |
| M5 BF16 | ≈291 ms | 8.7 s | 1.45 s | ≈10.2 s | ≈53 s | ≈10 min |
| M4 Pro BF16 | ≈670 ms | 20 s | 2.9 s | ≈23 s | ≈2 min | ≈23 min |
| M4 無印 BF16 | ≈1.1 s | 34 s | 4.8 s | ≈39 s | ≈3.3 min | ≈40 min |

- ピークメモリ: 重み 4.6/8.9 GB、DiT 活性 ≈1.8 GB（hidden 75 MB、qkv 226 MB、gather 3×142 MB、FFN 中間 351 MB、int8 作業 176 MB）、ReAE ≈1.5–2 GB、ホスト 0.6 GB、MPS 作業域 ≈1 GB → **≈10 GB（int8）/ ≈14.3 GB（BF16）**。`h3_vsr_requirements` の係数: ≈150 KB/トークン + ≈1 KB/出力画素 + 重み（`h3_gpu_stats.peak_live_bytes` で較正）。24 GB 機では出力 ≈4.5 M 画素超を拒否（UI で倍率をグレーアウト）。

### §6.5 カーネル計画

**そのまま流用**: `h3_gpu_linear_int8_bf16`（ブロックの線形 5 本）、`h3_gpu_linear_bf16`（BF16 経路。patch_embed 192→3072 は bias 付きで in<256 のため 16×16 タイル核、14 GFLOP で許容；`proj_out` は NAX）、`h3_gpu_quantize_weight_int8`、`h3_gpu_layer_norm_bf16`（norm1/norm3 は weight=(1+scale), bias=shift；norm2 は affine）、`h3_gpu_copy_bf16`（MemBlock 状態）、`h3_gpu_dequantize`、`stream_file_*`、`h3_weight_store_*`、`begin/continue/submit`、`h3_gpu_stats`。

**小さな一般化**:
- `void h3_gpu_set_nax_generic(h3_gpu*, int on)` — in%32/out%64/rows≥128/bias なしを満たす任意形状で `naxShape` を真に（`h3_gpu.m:3151`）。
- `int h3_gpu_sdpa_batched_bf16(gpu, out, q, k, v, batch, q_len, kv_len, heads, head_dim, scale)` — 既存 `h3_gpu_sdpa_graph` に `kv_sequence` を追加（K/V プレースホルダ `[batch, kv_len, heads, head_dim]`、キャッシュキーに kv_len）。MFSWA（batch=Nw または owner クラス）、交差注意（batch 1, kv 512）、FlashVSR の top-k gather に共用。
- `h3_gpu_conv2d_bf16_mps(...)` NHWC・明示パディングの MPSGraph conv（M4 経路）。MPSGraph が BF16 conv を拒めばグラフ内で F16 にキャスト（`h3_gpu_create` で一度プローブ）。

**新カーネル**（全て `h3_vsr_*`、F32 演算・BF16 格納、`tests/test_vsr.c` に CPU double 参照）:
1. `h3_vsr_bilinear_up_to_unshuffled_bf16`（BGRA8 → bilinear `src=(dst+0.5)·scale−0.5` クランプ・antialias なし → /255 → 右下ゼロパディング（32 の倍数）→ pixel_unshuffle(2) チャネル順 (c,ph,pw) → `[T, Hp/2, Wp/2, 12]`）。MPSGraph `resize(bilinear, centerResult=YES, alignCorners=NO)` が OpenCV/TF2 と同じ半画素中心 [確認: SDK ヘッダ] なので代替候補だが、前処理全体を 1 核に融合する方が速い。
2. `h3_vsr_shuffle_to_pixels`（clamp[0,1]→×255→切り捨て（参照）or 丸め（オプション）→ BGRA8/RGB24、crop）。
3. **`h3_vsr_tapconv_bf16_nax`（M5）— タップ累積 matmul2d の汎用核**: `out[rows,Cout] = epi(Σ_tap X[row+off_tap,:]·W_tapᵀ (+ Σ Past[...]·W2_tapᵀ) + bias)`、引数 `{rows, Cin, Cout, taps, offsets[16], relu, residual_ptr, k_split}`。`h3_linear_int8_nax_r128`（`h3_shaders.metal:2597-2650`）と同じ `multiply_accumulate` を bfloat で。入力は `[(T), H+2, W+2, C]` ゼロパディング channels-last（タップ = 定数行オフセット）。用途: 3×3 conv（9 タップ）、MemBlock 先頭 conv（`cat[x,past]` を実体化せず K 分割: x に 9 タップ×W[:, :C]、past に 9 タップ×W[:, C:]）、1×1、TPool（フレーム対の 2 タップ、重み `[C,2C]` 分割）、TGrow conv3d(3,1,1)（±1 フレームの 3 タップ）、stride-2 3×3 は space-to-depth(2)+2×2 タップ（K=16C、+0.2 T/チャンク）、**Upsample2×→conv3×3 の折り畳み**（4 位相核 = 2×2 タップ・`4·Cout` 出力 + depth-to-space；nearest と可換なので厳密、位相重みはタップ和を F32 でロード時生成）。チャネルパディング Cin 12→32、Cout 12/48→64。
4. `h3_vsr_s2d_bf16`/`h3_vsr_d2s_bf16`、`h3_vsr_clamp_tanh_bf16`、`h3_vsr_pad_copy_bf16`。
5. `h3_vsr_patchify_bf16`（`[T,68,120,48] → [N,192]` 順 (c,ph,pw)）、`h3_vsr_unpatchify_sub_f32`（`z_hq = z_lq − (v + b_proj_out)` を F32 潜在へ）。
6. `h3_vsr_qk_rms_inverse_bf16`（`qkv [N,9216]`, bq, bk → `inv[N,2]`）と `h3_vsr_window_gather_qkv_bf16`（bias + head 横断 RMSNorm + **interleave RoPE**（22 t 対, 21 h, 21 w）+ gather → `Qg[rows_q,24,128]`, `Kg/Vg[rows_kv,24,128]` を融合）。
7. `h3_vsr_window_scatter_bf16`（`Og[rows,3072]` → `O[N,3072]`、owner/permutation）。
8. `h3_vsr_bias_gate_add_bf16`（`x += (y+b)⊙g`、g NULL は 1）、`h3_vsr_gelu_bias_bf16`、`h3_vsr_bias_rms_bf16`（attn2 q）。
9. 第二段の融合: `h3_vsr_gelu_bias_quantize_int8`、`h3_vsr_layer_norm_mod_quantize_int8`（351 MB／75 MB の 1 パスずつ節約）。

**厳密なアルゴリズム的工夫 — owner 限定クエリ**: 各出力トークンはただ 1 つの窓から読まれるので、非 owner 位置のクエリは捨てられる。窓ごとに owned クエリ行だけを gather（K/V は窓全体）すれば Σowned = N、注意 FLOPs は両パリティで 0.231 T/ブロック（全窓比 −33%/−47%）、出力はそのまま連続（scatter は置換）。窓を owned 行数でクラス分け（偶数層 4 クラス {1536,1152,192,144}、奇数層 7 クラス）し、クラス毎に `h3_gpu_sdpa_batched_bf16` 1 回。**まず全窓版で parity、次に owner 限定版**。

**RoPE オフセットの工夫**: `dit_overlap=0` なら自己注意はチャンク内に閉じ、RoPE は相対位置のみに依存するので時間オフセットは打ち消される → 全チャンク `t_off=0`（形状クラス毎に 6.3 MB の表 1 枚）。`rope_max_seq_len=1024` の長尺問題も消える。harness で block-0 の t_off 0 vs 6 が ≥60 dB であることを確認。

**ブロックの演算列（SwiftVR）**: LN(norm1-mod) → int8 QKV → rms_inverse → gather(bias, norm, RoPE) → batched SDPA → scatter → int8 out → bias_gate_add(gate_msa) → LN(norm2 affine) → int8 q2 → bias_rms → cross SDPA(K2,V2) → int8 out2 → bias_gate_add(gate NULL) → LN(norm3-mod) → int8 fc1 → gelu_bias → int8 fc2 → bias_gate_add(c_gate)。最終: LN(norm_out-mod) → NAX BF16 proj_out → unpatchify_sub。数ブロックごとに `h3_gpu_continue`、ブロック間でキャンセル確認。

### §6.6 パイプライン並列とキャンセル

Swift（`VideoUpscale.upscale` を 3 段に。既存方式は `PerFrameChunkAdapter`（チャンク 1）で同じ経路）:
1. **Reader**（`h3c.vsr.read`）: AVAssetReader の BGRA8 フレームを `h3_vsr_session_next_chunk_frames` 枚ずつ PTS 付きで束ね、深さ 2 の有界キューへ。
2. **Engine**（`h3c.vsr.engine`、直列）: 入力 `CVPixelBuffer` をロックして `h3_vsr_session_process`、出力は adaptor pool の `CVPixelBuffer`。出力＋PTS を FIFO（入力と 1:1）で writer キューへ。
3. **Writer**: 既存 `pump(videoInput)` が完成フレームを取り出して append。音声 pump・ステージング・置換・メタデータ・transform は不変。

エンジン内: スレッド A が DiT 用 `h3_gpu`、B が ReAE 用（`reae_separate_queue`）。`process(k)`: B が k をエンコード → A が DiT k を走らせる間に B が k−1 をデコード → k−1 のフレームを返す（1 チャンク遅延、`is_last` で flush）。MemBlock/TPool 状態は B に閉じるので因果性は自明。**正直な期待値**: GPU は共有資源なので、2 キュー並走で隠れるのは CPU 側エンコード／アップロード遅延と ReAE の小カーネルによるバブル充填（ReAE 時間の 10–30% [推定]）であり、チャンク時間の下限は GPU 合計。
キャンセル: `cancel_requested` を DiT ブロック間・段階間で確認（`h3_dit.c` の `stop_if_cancelled` 流儀）、両スレッド同じフラグ、`process` は 0 を返し "vsr cancelled"。Swift は writer を止めてステージングを削除（既存どおり）。進捗 = (frames_done + chunk_frames·block/30)/expected。

### §6.7 Neural Engine（Core ML）での ReAE — go/no-go 規則

- 期待効果: ReAE の GPU 時間（M5 int8 で 6.5 s 中 ≈1.45 s ⇒ 最大 −22%、M4 Pro で −13%）を ANE に逃がして DiT と並走。
- 機構（GO の場合）: 重みは再配布できないので、**重みなしの `.mlpackage` テンプレート**（Apache-2.0 のアーキテクチャから coremltools で開発時に生成、重みは `BLOBFILE` 参照）を解像度クラス毎に同梱し、端末内で `reae.safetensors` から `weights/weight.bin`（coremltools `blob_writer` 形式）を書いて `MLModel.compileModel(at:)` [確認: SDK] でコンパイル。MemBlock/TPool 状態はモデル入出力（または `MLState` [確認: macOS 15 SDK]）、F16、静的形状。
- 判定（実装前に測る）: 開発機で stage-3 MemBlock 1 個（480×272×128、12 fr、F16）を coremltools で作り M5 で計測。**GO ⇔ 全 ReAE 換算の ANE 時間/チャンク ≤ 0.8 × DiT 時間（完全に隠れる）∧ 出力 PSNR ≥ 45 dB vs F32 ∧ 入出力コピーが実行時間の ≤20%**。
- v1 の既定: **NO-GO**（GPU NAX tapconv（M5）/ MPSGraph（M4）で出荷し、ANE は P7 の実験）。

## §7 アプリ統合

- `VideoUpscaler` に **`.swiftVR`**（表示名案「AI 超解像（SwiftVR, 非商用）」）を追加。`isAvailable` = 変換済みキャッシュが存在 ∧ Metal あり。
  FlashVSR Tiny は `.flashVSRTiny` を予約（第二フェーズ）。
- **新プロトコル `ChunkedFrameScaler`**（`FrameScaler` と並存、§6.1/§6.6）: `func chunkFrames() -> Int`、
  `func process(_ frames: [CVPixelBuffer], isLast: Bool, into pool: CVPixelBufferPool) throws -> [CVPixelBuffer]`。
  既存 4 方式は `PerFrameChunkAdapter`（チャンク 1）で同じ 3 段パイプラインを通り、出力はバイト一致を保つ。
  出力は PTS 順・1 入力 1 出力（1 チャンク遅延、`isLast` で flush）。進捗はエンジン callback でブロック単位、キャンセルはブロック境界＋`cancel_requested`。
- 倍率: 1.5/2/3/4 は現状の契約を維持。SwiftVR は入力を目標サイズに bilinear してから復元するため任意倍率で動くが、**初期リリースは 2x のみ推奨**（4x 以上は時間・画質を測ってから; `UpscaleSheet` で未対応倍率を無効化できる仕組みを追加）。
- 重み取得 UI: 「拡大」シートの SwiftVR 行に「モデルをダウンロード（約 20.2 GB → 変換後 約 5.3 GB [int8] / 10.3 GB [BF16]）」ボタン、
  `ModelLicenseNotice` と同形式の注記（Research License、非商用、HF リンク）。DL は `ModelDownloader`（再開可）。
  変換はエンジン C で逐次（テンソル単位、ピーク RSS < 2 GiB [目標]）、完了後に元 F32 を削除（設定で保持可）。保存先 `~/Library/Application Support/H3cApp/Upscalers/SwiftVR/`。
- 記録: `upscaler="swiftVR"`、`upscaledSize`、新規 `upscalerModelVersion`（キャッシュのフィンガープリント）。
- API: `upscaler` に `swiftVR` を追加（`API(.ja).md` 更新）。未取得なら 400 で理由を返す。
- 文書: `GUIDE(.ja).md` の「Enlarge afterwards」に 1 項目、`TECHNICAL(.ja).md` に「AI 超解像」節、`THIRD_PARTY_NOTICES.md` に SwiftVR（コード Apache-2.0 の帰属。重みは同梱しないので DL 時表示）、サイト `guide.html` の表に 1 行、`website-facts.md` 更新。
- 継続（enlargement 継続）は既存どおり動く（記録の `upscaler` 文字列のみ増える）。

## §8 実装フェーズと受け入れ条件

| Phase | 内容 | 完了条件（測って記録） |
|---|---|---|
| P0 文書 | 本計画を `UPSCALE_PLAN(.ja).md` として追加（ブランチ `upscale-plan`） | マージ |
| P1 参照データ | scratch venv で公式 SwiftVR を CPU/MPS で小入力（8 fr 256x256 → 512x512）実行し、§9 のチェックポイントと定数を safetensors でダンプ（`tools/dump_swiftvr_fixture.py`、Python は開発専用）。窓開始位置・owner 規則も同スクリプトで書き出す | ダンプ一式と README |
| P2 重み変換 | `h3_vsr_weights.c` + CLI `build_vsr_cache`: F32 safetensors → `.h3vc`（BF16 / int8 行量子化、F32 定数、交差 K/V、融合 QKV、fingerprint）。`h3_gpu_set_nax_generic` もここで | 変換時間・サイズ記録、再読込一致、`test_vsr` のホスト部 |
| P3 ReAE | `h3_vsr_reae.c` + tapconv NAX 核 + MPSGraph 経路（状態持ち越し、フレームバッチ、TGrow 前置、アップサンプル折り畳み） | 参照と ReAE 潜在 ≥ 40 dB(BF16)、デコード RGB ≥ 38 dB；両経路の一致 |
| P4 DiT | `h3_vsr_wan.c` + `h3_vsr_window.c`: 全窓 MFSWA → owner 限定、BF16 → int8、定数条件、交差注意、RoPE（t_off=0 の検証） | block-0 ≥ 40 dB、最終潜在 ≥ 36/33 dB、RGB ≥ 38/35 dB（BF16/int8）；dense 注意との差を記録。実重み（付録 A.1）: 公式 float32 に対し v が bf16 60.0 dB / int8 51.3 dB（MFSWA）、全注意 60.4 / 51.7 dB |
| P5 ストリーマ | `h3_vsr_chunk.c`/`h3_vsr.c`: チャンク分割、状態、最終チャンクの前詰め、前処理・切り出し、2 コンテキスト並走、キャンセル | 124 fr の end-to-end が参照の全体出力と ≥ 35 dB、M5/M4 で時間・ピークメモリ表（`bench_vsr`）。実重み: 57 フレーム 384x272→768x544 の 8 ビット出力が公式と bf16 51.2 dB / int8 49.3 dB、出力数一致 |
| P5.5 品質評価 | §9.5: 評価セット作成、SeedVR2 参照出力（≥64 GB Mac かクラウド）、指標スクリプト、本実装の全変種で表を埋める、ブラインド A/B | SeedVR2 比の達成率（§1 の 95%）を記録し、「品質優先」「速度」モードの既定を決める |
| P6 アプリ | `ChunkedFrameScaler` と 3 段パイプライン、**品質優先／速度モードの切替（全注意・精度・overlap）**、`.swiftVR`、DL/同意/変換 UI、API、記録、文書、ローカライズ、`isGenerating` ガード | 既存 4 方式の出力がバイト一致で不変、新方式で project/file 両入口＋継続が動く。実重み: 新ビルドのアプリで `POST /api/upscale-file`（swiftVR, 2x）により 512x512・39 フレームが 1024x1024 HEVC（音声保持）に約 30 秒で拡大された（2026-10-29）。「SwiftVR のモデルを準備」画面の操作は未確認 |
| P7 最適化 | 進行中（付録 A.2: ReAE の畳み込みを TensorOps に移し、1080p の中間チャンク 17.6 s → 9.6 s） | `h3_vsr_tapconv_bf16_nax`, `h3_vsr_reae.c`, `tests/bench_vsr.c` | `tests/test_vsr_tapconv.c`、`tests/test_vsr_reae.c` の TensorOps 節、実重み 56〜60 dB |
| P8 FlashVSR Tiny | 共通エンジンに第二バックエンド（LQ_proj_in、KV キャッシュ、ブロック選択注意、TCDecoder、色補正） | 同様の parity と速度表 |
| PX 判定実験 | H3 自身による再生成（Open-H3-Regenerate-2K、§11。開始潜在 API → 潜在アップスケーラ → 既存 LoRA 重ね掛け → 1344×768 で SwiftVR と A/B） | §11.3 の A/B 表と判定の記録 |

各フェーズは独立ブランチ可（P2/P3/P4 は P1 のダンプに依存）。

## §9 検証計画

- **fixture**（ローカル、`misc/fixtures/` に gitignore、ライセンス付き重みの派生物なので配布しない）: `tools/dump_swiftvr_fixture.py` が公式パイプラインを 8 fr 256×256（bf16 と f32）で走らせ、
  `x.lq_rgb`, `reae.z_lq`, `dit.block0_out`（t_off=6 版も）, `dit.final_latent`, `z_hq`, `rgb_out`、定数（`temb`, `mod[30,6,3072]`, `context`, `K2_0`, `V2_0`）を safetensors で保存。
- **`tests/test_real_vsr.c`**（skip ガード）: 各チェックポイントで PSNR／相対 RMS。合格線 [目標値、実測で改訂]: 定数 ≥ 60 dB vs f32；ReAE 潜在 ≥ 40 dB（BF16）；block-0 ≥ 40 dB（BF16・int8）；最終潜在 ≥ 36 dB BF16 / ≥ 33 dB int8；デコード RGB ≥ 38 dB BF16 / ≥ 35 dB int8（bf16 参照比）、チャネル平均バイアス < 0.5/255；t_off 0 vs 6 ≥ 60 dB。int8 FC2 が落ちれば `keep_fc2_bf16` を既定に。
- **`tests/test_vsr.c`**（常時 `make test`）: ホスト — 軸開始位置が `{0,16,18} {0,8,18} {0,16,32,44} {0,8,24,40,44}` を再現、全トークンが自分を含む窓にちょうど 1 回 owned、owner クラス和 = N、1…200 フレームのチャンク計画の総和一致、RoPE 表 vs double、patchify 往復、折り畳み重みの等価；GPU — §6.5 の全カーネル vs CPU double、batched SDPA vs ループ `h3_gpu_sdpa_bf16`（相対 1e-2）、cross SDPA vs CPU、決定性（2 回実行がビット一致）。
- **`tests/bench_vsr.c`**: 実形状の合成重みで ms/ブロック・ms/チャンクを出し、§6.4 の推定値と UI の `estimated_seconds_per_chunk` を置き換える。
  原則: 数値だけで判断せず、両出力のフル解像度フレームを並べて見る。
- **画質（忠実度系）**: 既存の比較手順（半解像度コピー→2x→元と比較、PSNR/SSIM/detail/flicker）に SwiftVR 列を追加。生成動画 3 シーン（実写風・アニメ風・人物）で目視。

### §9.5 SeedVR2 に近づけるための品質プログラム

1. **評価セット**（`misc/quality/set-v1/`、gitignore）: H3cApp 生成動画 6 本（960x544・672x384、実写風人物／アニメ風／風景／暗部／毛皮／文字入り、各 2–5 s）＋実写 2 本（1080p を 540p に縮小したもの。元が ground truth になる）。seed・プロンプト・設定を一覧に記録。
2. **SeedVR2 参照出力**（一度だけ、開発時）: 24 GB 機では走らないので **(a) ≥64 GB の Mac で mlx-gen（SeedVR2-3B, Apache-2.0）**、または **(b) クラウド GPU（A100/H100 を数時間借用）で公式 SeedVR2-3B** を、評価セット全本・2x で実行し `misc/quality/seedvr2/` に保存。設定（解像度、ステップ、seed、タイル、カラー補正）を記録。
   どちらも不可なら、SwiftVR 論文・FlashVSR 論文の SeedVR2 比較値を暫定基準にし **[未確認]** と明記。
3. **比較対象**: SeedVR2-3B（基準）／公式 SwiftVR（PyTorch, bf16）／本実装（BF16・全注意、BF16・MFSWA、int8・MFSWA、overlap 0/1、2x 直接 vs 4x→Lanczos 1/2）／Real-ESRGAN General／Lanczos。
4. **指標**（開発用 Python、`tools/quality/`）: 無参照 MUSIQ・MANIQA・CLIP-IQA（pyiqa）、動画 DOVER；時間一貫性 = 隣接フレーム warp error（RAFT 光流）と既存 flicker 指標；実写 2 本は PSNR/SSIM/LPIPS vs ground truth；**SeedVR2 出力との距離** = LPIPS / DISTS（生成結果どうしなので PSNR は参考値）。
5. **目視**: 全解像度の静止フレーム 3 点＋ 2 秒ループを並べ、ユーザーがブラインドで A/B（SeedVR2 vs 本実装）。肌・髪・毛皮・文字・暗部の 5 項目で採点。
6. **品質レバーと評価順**（差が出たときに動かす順）:
   1. **全注意（dense SDPA）**: 論文 Table 3 では全注意が MFSWA より PSNR +0.28 dB／LPIPS 改善。M5 で +≈4.4 s/チャンク（1.84 vs 0.23 TFLOP/層）、MPSGraph の融合 SDPA ならスコア行列は実体化しない [確認: 既存 h3_gpu_sdpa 利用実績]。「品質優先」モードの既定候補。
   2. **精度**: int8 → FC2 のみ BF16 → 全 BF16。ブロック毎 PSNR と最終指標の両方で判断。
   3. **`dit_overlap=1`**（公式 stream の既定）: チャンク境界のフリッカ低減。
   4. **2x の作り方**: 目標 2x へ直接 vs 学習倍率に近い 4x で生成して 1/2 に Lanczos（Real-ESRGAN で既に採る方式）。質感と時間を比較。
   5. ReAE を F32 で実行（デコーダの階調）。
   6. 入力前処理: bilinear 以外（bicubic / Lanczos）を試す。学習時分布と異なるので慎重に。
   7. 第二バックエンド（FlashVSR Tiny または FlashVSR+）で同じ表を埋め、SeedVR2 に最も近いものを既定にする。
7. **記録**: 結果表を `UPSCALE_PLAN(.ja).md` の付録と `SPEEDUP_ROADMAP.md` の流儀（機種・設定・seed・日付）で残す。
- **回帰**: 既存 4 方式はバイト一致（`H3SPIKE_DUMP` 相当のフレームダンプ）。
- **性能**: `/usr/bin/time -l` でピークフットプリント、`H3_PROFILE=1` で段階別時間。M5 と M4（ユーザー所有の実機が無ければ BF16 経路を M5 で `H3_NAX=0` 相当で測り、上限見積とする [未確認]）。
- **ライセンス表示**: DL ダイアログ、About、THIRD_PARTY_NOTICES のスクリーンショットを記録。

## §10 性能・メモリ見積（要約）[推定]

詳細は §6.4。要点: 960x544→1920x1088・24 fr チャンクは **M5 int8 ≈6.5 s／チャンク、ピーク ≈10 GB**（5 s クリップ ≈34 s、60 s 動画 ≈6.5 分）、
**M5 BF16 ≈10 s／≈14.3 GB**、**M4 Pro BF16 ≈23 s**（5 s クリップ ≈2 分、60 s ≈23 分）、M4 無印 ≈39 s/チャンク。
前提（int8 = BF16 の 2 倍、M4 の MPSGraph 実効 TFLOP/s）は未測で、`tests/bench_vsr.c` の実測で置き換える。§1 の目標（M5 2 分以内／M4 6 分以内、5 s クリップ）はこの見積の範囲内。

## §11 判定実験: H3 自身による再生成（hires-fix → Open-H3-Regenerate-2K）

2026-10-31 改訂。以前の案（生成済み潜在を即席に再ノイズする hires-fix）を、同じ考え方を専用に学習した公開重み **Open-H3-Regenerate-2K**（NVlabs、2026-10-27 公開）で置き換える。目的は機能の移植ではなく **判定** であり、判定点はひとつだけ:

> **1344×768 ですら SwiftVR 2x に対して顔・肌・毛髪・文字で明確に良くならなければ、1080p / 2K 対応には進まない。**

### §11.1 対象の事実 [確認: 2026-10-31、HF モデルカード・NVlabs README／プロジェクトページ・ComfyUI 移植 2 件]

- 構成: 凍結した H3 DiT と VAE に、(a) 統合 LoRA（FP32、rank 576 = 精細化 384 + LightX2V FL2VA Turbo v1.2 128 + LMS Sharp v1.0 64 を rank 軸で連結、12.5 GB、**312 線形層** = 50 ブロック × 6 + トークン精製 2 ブロック × 6）、(b) LMS Sharp の AdaLN 曲線分岐（50 本、FP32 1.24 GB。`B·A·curve8(t)` を各ブロックの AdaLN 射影に加える。8 次元の curve は Comfy-Org の pruned チェックポイントにある 1025×8 の `adaln_t_table` から読む）、(c) コミュニティ製 3D 畳み込み潜在アップスケーラ（LBH-123-AI、BF16 691 MB、**345.3M パラメータ・322 テンソル**）、(d) 高速 VAE デコーダ（StreamRender-H3、36 層を 24 層に刈り込んで蒸留、非因果、1.62B、FP32 6.47 GB、FlashAttention-4 を要する）。
- 手順: 潜在を目標格子へ直接拡大（デコードも再エンコードもしない）→ `x = (1−σ₀)·z_up + σ₀·ε`、σ₀ = 0.909375 → Euler 3 ステップ（σ = 0.909375, 0.857143, 0.666667, 0、CFG 1、model time = 1−σ）→ デコード。2 ステップ版は σ = 0.7, 0.494118, 0（README: 「約 30% 速いが、毛皮や苔の細部が目に見えて柔らかい」）。`--start-sigma` で開始 σ を変えられる。音声は「符号化した無音で固定」。**各ステップは出力解像度の全トークンに対する DiT 1 回**で、タイルや窓は無い（sol_engine では Sol-Attn の疎注意）。
- 出力: 768p（精細化のみ、拡大なし）／1080p／2K（既定）／4K（2K 経由で 2x を 2 回）。README が扱う入力は 1344×768 と 1024×576 の 2 つ（384P/512P はブログの記述のみ）。5 秒（121 フレーム、潜在 37 フレーム）が標準、15 秒まで 1 パス。
- 性能（B200 1 基、5 秒、768P → 2K、ウォーム、中央値）: 密 BF16 + 公式 VAE 87.26 s（精細化 71.27 s、デコード 14.99 s、拡大 1.01 s）、sol_engine_fast + 高速 VAE 22.45 s（3.9x。**自社の密実装に対する比**であり、見出しの「公式 API の 5 倍」の根拠はページに無い）。CUDA グラフ版は「1 GPU で約 150 GB」。72 GB カードへの移植例（int8 ベース）: 精細化 122 s（疎）／474 s（密）、高速デコーダ単体で 2K 5 秒 21 GB・15 秒 45 GB。
- 品質主張: ブラインド 200 判定（50 事例）で **公式 H3-Regenerate-2K API に対して** 87%。VBench 平均 0.814 対 0.802、UniPercept 60.0 対 53.4。他のアップスケーラー（SwiftVR、SeedVR2）との比較は無い。README 自身が「同 seed の PSNR は指標として使えない」と書く。高速デコーダは公式デコーダに対し 30.0 dB（移植者の測定では 37.8 dB）。
- 既知の弱点: σ₀ 0.909 は描き直しが強く、移植者は「狐の顔が狼に寄った」と報告（曲線分岐の有無やデコード経由でも同じ → 方式由来）。開始 σ 0.8 で緩和。
- ライセンス: 統合 LoRA・高速 VAE・base_model は MiniMax H3 Community License。潜在アップスケーラは NVIDIA 配布物の `latent_upsampler/LICENSE` が Apache-2.0、**元の LBH リポジトリは MIT**（NVIDIA のラッパも「MIT」と記す）。LightX2V と LMS の分は未記載。モデルカード自身が「一括のライセンスは無い」と明記。Ref2VA への言及は無い。
- H3cApp との対応: アプリの LoRA ローダは diffusers 配置を読み、FC1 の [value; gate] → [gate; value] の入れ替え（いわゆる SwiGLU 変換）と 2 つのトークン精製ブロックをすでに扱う（`h3_lora.c`）。F32/F16/BF16 を読め、rank は形状から取る。生成動画は潜在を `.h3lt` に保存しているので、アプリ製ドラフトの再エンコードは不要。

### §11.2 24 GB Mac での見通し [推定: 実測からの外挿]

根拠にした実測: 768²/124 フレーム（21,728 トークン）で DiT 1 ブロックの **65.7% が注意**（`SPEEDUP_ROADMAP.md` のマイクロベンチ、int8 線形）。768²/362 フレーム（約 6.3 万トークン）で 1 ステップ 657 s（スワップ発生、Stage ⑦）。1344×768/15 秒（約 11 万トークン）では **1 ステップが 2.5 時間で終わらなかった**（`GenerationViewModel.swift` の注記）。5 秒 = 潜在 37 フレームとして動画トークンだけを数えると:

| 出力 | 動画トークン | 見通し（M5 24 GB、int8） |
|---|---:|---|
| 768p 精細化のみ（1344×768） | 37,296 | 3 ステップ + VAE で **10〜15 分**。メモリは収まる見込み。実測前の仮説 |
| 1080p（1920×1088） | 75,480 | スワップした 6.3 万の先。1 ステップ 15〜20 分以上、3 ステップで 1 時間超。FFN と注意のトークン分割が前提 |
| 2K（2560×1440） | 133,200 | 2.5 時間で 1 ステップが終わらなかった構成より大きい。Sol-Attn 級の削減なしでは対象外 |

FLOPs でも整合する: 2K 5 秒 3 ステップ ≈ 7×10¹⁶ FLOP（線形 29 GFLOP/トークン、注意 4·N²·5376/ブロック）。B200 の密実装 71 s は実効約 1 PFLOP/s、M5 の実効 10〜25 TFLOP/s では 1.5 時間級。**24 GB で 2K が無理なのは「メモリに入らない」からではなく、分割して入れても計算時間が実用にならないから**である。したがって本丸は注意機構であり VAE ではない。VAE デコードはアプリではすでに int8 かつタイル化済みで全体の 15〜20%、24 層化で縮むのはその 3 分の 1 程度（全体の 5〜7%）に留まり、しかも損失がある。**24 層 VAE は最後**（Regenerate が採用になり、DiT 側を疎注意で速くした後）。

### §11.3 実装順（判定実験に必要な最小限）

1. **汎用の開始潜在 API**（Open-H3 専用にしない）: `h3_params` に `initial_video_latent`（`.h3lt` または配列）、`video_sigmas[]`（任意の σ 列）、`audio_sigmas[]` または「音声はクリーン固定」、を足す。`h3_forced_trajectory_start` の式をそのまま使う。プロンプトはプロジェクトのものを使う（README の推論はテキストエンコーダを読み込むので、条件付けは生きている）。1344×768 はエンジンの上限（`h3.c` の「released 768*1344」）の内側。同じ API が hires-fix・2/3 ステップ精細化・途中潜在からの再生成に使える。
2. **潜在アップスケーラの Metal 移植**（判定実験の部品として。単独機能にはしない）: 構造は作者ノード `upscaler_node.py`（MIT）の `LatentResizer3D(in_channels=24, in_blocks=12, out_blocks=12, channels=512, temporal_every=2, temporal_kernel=5, attn=False)`。`conv_in` Conv3d(24→512, 3³) → in_blocks 12 → **trilinear 補間（align_corners=False）で目標格子へ** → out_blocks 12 → GroupNorm(32)+SiLU → `conv_out` Conv3d(512→24, 3³)。ブロックは ResBlockEmb3D（GroupNorm(32) → SiLU → Conv3d 3³ → 埋め込みからの FiLM（scale, shift）→ SiLU → Conv3d 3³、残差）で、偶数番目のあとに TemporalConv（depthwise Conv3d (5,1,1) → 1×1×1、残差）。埋め込みは `Linear(1→64) → SiLU → Linear(64→64)` に `scale − 1` を入れる。入力は **正規化潜在**（24 ch の mean/std で割る。アプリの VAE が持つ `latents_mean/std` と同じ役割、値は作者ノードの定数を使う）、出力を逆正規化。**正規化は参照実装と数値で比べる**: 作者ノードの `LATENTS_MEAN/STD` 24 値とアプリの VAE 設定の `latents_mean/std` を突き合わせ、同じ入力潜在で正規化→拡大→逆正規化の各段を参照（作者ノード fp32）と照合して差を dB で記録する（定数が違えば作者ノードの値を採り、その事実を書き残す）。目標は 32 px 整列。作者ノードは潜在 32 フレーム超で時間分割（重なり 5）するが NVIDIA のラッパは無効化しているので、同じく一括で動かす。GPU 側は `h3_gpu` の conv3d（F32 MPSGraph）と既存の GroupNorm+SiLU 核で足りる。倍精度参照との照合は `h3_vsr_reae.c` と同じ流儀。
3. **LoRA は新規実装なしで**: Kijai 版の精細化のみ LoRA（rank 384、NVIDIA の統合 LoRA の rank 0〜383）＋手元の LightX2V FL2VA Turbo v1.2 ＋ LMS Sharp v1.0 r64 を既存の重ね掛け（最大 8 本）で適用する。NVIDIA の統合 FP32 12.5 GB を直接読む場合は、ローダのメモリ（rank 576 の B·A を F32 で保持）を確認する。**Sharp の曲線分岐は最初は無し**（判定後に実装。3 タイムステップ × 50 ブロック × 96768 の F32 差分を読み込み時に事前計算すれば 58 MB。8 次元 curve のために Comfy-Org の `adaln_t_table` 32 KB をバイト範囲で取る）。
4. **正規化した A/B 比較**: 入力はアプリで生成した **1024×576・5 秒**（と 672×384）の潜在に固定。A: 再生成 → 1344×768（3 ステップ標準／2 ステップ保守 σ 0.7, 0.494118, 0／開始 σ 0.8）。B: SwiftVR 2x → 2048×1152 → **1344×768 へ高品質縮小**。C: 通常のリサイズ → 1344×768。D（任意）: 即席 hires-fix（旧 §11 の σ₀ 0.3〜0.5）。指標は `ltx-refine-eval` の道具（`tools/quality_metrics.py`: LPIPS／顔 LPIPS／フリッカ／MUSIQ／MANIQA と顔の切り出し、`frames-sheet.png` を等倍で見る）。**「元への忠実度」と「見た目の品質」を分けて記録**する（再生成なので LPIPS が悪化しても見た目が上がりうる）。構図・同一性の変化、所要時間、ピークフットプリント（`/usr/bin/time -l`）も表に入れる。
5. **判定**: A が B に対して顔・肌・毛髪・文字で明確に良い（ブラインドの目視で過半、顔 LPIPS と無参照指標の両方で優位）場合のみ、次へ進む: Sharp 曲線分岐 → 厳密な Q 分割注意（メモリ）→ Sol-Attn 型ブロック疎注意（時間。NVIDIA 自身もステップ 0 は密、1〜2 を疎で 5〜12% の相対誤差を許容）→ 1080p。それでも 2K は対象外。良くなければ、旧 §11 と同じく「不採用」と記録し、通常生成の高速化（VDN-H3 など）に戻る。**ただし Sharp の AdaLN 曲線分岐を省いた状態の画質だけで Open-H3 全体を不採用と判断しない**: 公開重みは分岐込みで学習・評価されているので、省略版が SwiftVR に届かない場合は分岐を実装して再測定してから判定する（省略版で明確に勝っている場合だけ、分岐の実装を後回しにしてよい）。

### §11.4 ライセンス表示

部品ごとに、**取得した重み／ソースに付属するライセンスをそのまま表示**する（一律「Apache-2.0」とはしない）: 精細化 LoRA は MiniMax H3 Community License（基盤モデルと同じ既存の表示）、潜在アップスケーラは取得元に従い MIT（LBH GitHub）または Apache-2.0（NVIDIA 配布物）、LightX2V / LMS は利用者が自分で入れた LoRA としてライブラリの表示に従う。同梱はせず、SwiftVR と同じダウンロード・同意の流れ。

### §11.5 リスク・未確認

- σ₀ 0.909 の同一性ドリフト（顔）。開始 σ 0.8 と 2 ステップ版を同時に測る。
- 音声: README は符号化無音で固定。アプリはクリーン音声の強制経路で代替し、元の音声はパススルー。
- モデルカードは Ref2VA に触れない。LoRA は base（= FL2VA の DiT）に対して学習。Ref2VA への適用は未検証。
- 768p でも 3.7 万トークンの DiT を 3 回回す。M1〜M4（BF16 経路）では 2〜3 倍遅く、まず M5 で判定する。
- 作者ノードの `precision` 既定は README の fp32 ではなく fp16（コードが正）。BF16 重みを F32 で評価し、参照は作者ノードの fp32 で取る。
- VDN-H3 との関係: VDN-H3 はフレーム毎の線形注意分岐を足したハイブリッド構造を学習したチェックポイント（+ 2 本の LoRA、8 ステップ蒸留、約 82 GB）。精細化 LoRA を機械的に当てることは構造上は可能だが学習分布が違い、意味は薄い。別の開発分岐として扱う。

## §12 範囲外・別タスク（LightX2V 精査メモの他項目）

- LoRA 補正の別計算（`y = D(W_q)x + sB(Ax)`）診断モード: `h3_gpu_lora_delta_bf16` が既にあるので小改修。別ブランチ。
- M4 向け Q 分割注意: 本計画では MFSWA が窓長で有界なので不要。H3 生成側の長系列用として別タスク。
- 部分常駐（予算管理付き）: H3 生成側の SSD 経路向け。別タスク。

## §13 リスク・未確認事項

1. SwiftVR ライセンス（非商用）。アプリは配布せず DL のみだが、法的判断はユーザー。表示文言は MiniMax の既存文に倣う。
2. 2x での画質（学習は 4x 想定の可能性）。P5 で測る。
3. 20 GB の DL（`ModelDownloader` の再開で対応）、変換の一時領域（元 20 GB + キャッシュ 5–10 GB）。
4. MFSWA owner 規則の 2 実装（公式 / LightX2V）の差。P1 で両方ダンプして確認。
5. MPSGraph の bilinear と `F.interpolate` の一致。不一致なら自前カーネル。
6. M4 性能（実機未測）。
7. ANE: 重みの再配布不可のため Core ML モデルは端末内生成が必要（§6 の判断基準）。
8. **SeedVR2 参照出力の作成には ≥64 GB の Mac かクラウド GPU が必要**（24 GB 機では不可）。用意できない場合は論文値を暫定基準にし、達成判定の信頼度を下げて記録する。
9. SwiftVR は 4x 学習の可能性があり、2x 直接では質感が弱いかもしれない（§9.5-6-4 で 4x→1/2 を比較）。
10. 入力サイズが 32 の倍数でない動画（例 1920x1080 → 1088 へパディングし切り出し。問題なし）、縦長、極端なアスペクト。

## §14 参照

- LightX2V 精査メモ（LightX2V 543cff4 vs H3cApp v0.7.0 489ddc3）。
- SwiftVR: https://github.com/H-oliday/SwiftVR , https://huggingface.co/H-oliday/SwiftVR , https://arxiv.org/abs/2606.09516
- LightX2V SwiftVR 実装: `lightx2v/models/networks/swiftvr/{reae,attention,model,streaming}.py`, `lightx2v/models/runners/swiftvr/swiftvr_runner.py`
- FlashVSR: https://github.com/OpenImagingLab/FlashVSR , https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
- TAEHV: https://github.com/madebyollin/taehv （MIT）
- diffusers `transformer_wan.py`（WanTransformer3DModel）
- h3c-app: `SPEEDUP_ROADMAP.md`（NAX/ccv/メモリ実測）、`TECHNICAL.md`、`VideoUpscale.swift`、`TinyPreviewDecoder.swift`、`h3_gpu.h`
- Open-H3-Regenerate-2K: https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K , https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K , https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/ （2026-10-31 確認）
- 潜在アップスケーラ: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler （MIT）, https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
- ComfyUI 移植: https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate （Apache-2.0）, https://github.com/1038lab/ComfyUI-Open-H3-Regenerate-2K （GPL-3.0）
- VDN-H3: https://huggingface.co/OpenVDN/vdn-minimax-h3

## 付録 A: 実装状況（2026-10-28, ブランチ vsr-engine）

| フェーズ | 状態 | 場所 | 検証 |
|---|---|---|---|
| P0 文書 | 完了（main） | `UPSCALE_PLAN(.ja).md` | — |
| P1 参照ダンプ | 完了（2026-10-29。利用者が同意のうえ 20 GB をダウンロード。付録 A.1） | `tools/dump_swiftvr_fixture.py` → `misc/fixtures/swiftvr-768x544/`（gitignore） | `tests/test_real_vsr.c`（重みと fixture があるときだけ走り、なければ skip） |
| P2 重みキャッシュ | 完了 | `h3_vsr_weights.c`, `h3_vsr_internal.h`, `h3_build_vsr_cache.c` | `tests/test_vsr_cache.c`（合成モデル、bf16+int8） |
| P3 ReAE | 完了（F32 conv3d 経路） | `h3_vsr_reae.c` | `tests/test_vsr_reae.c`: 倍精度転写と 2e-7、チャンク／フレームバッチの状態持ち越しはビット一致 |
| P4 DiT | 完了（BF16 / int8、全窓 MFSWA と全注意） | `h3_vsr_wan.c`, `h3_shaders.metal` の `h3_vsr_*`, `h3_gpu_sdpa_batched/cross_bf16` | `tests/test_vsr_wan.c`: 倍精度参照と 61 dB（bf16）/ 60.8 dB（int8） |
| P5 ストリーマ | 完了（1 コンテキスト直列、1 チャンク遅延なし＝デコーダの 3 フレーム遅延のみ） | `h3_vsr.c` | `tests/test_vsr_session.c`: 7/29/52/70 フレームで出力数一致・決定的・キャンセル |
| P5.5 品質評価 | 一部（付録 A.3: 評価セット 5 本、忠実度・無参照指標・フリッカ、顔の A/B 画像。SeedVR2 参照とユーザーのブラインド採点は未） | `tools/quality_metrics.py`, `tools/quality_report.py`, `tests/vsr_run.c`, `misc/quality/set-v1/` | 付録 A.3 の表 |
| P6 アプリ | 完了（実重みでの動作確認は P1 待ち） | `H3VSR.swift`, `VideoUpscale.swift`（`ChunkedFrameScaler`, `.swiftVR`, 生成中ガード `upscaleBlockedReason`）, `UpscaleSheet.swift`, `SwiftVRModelSetup.swift`（ライセンス同意・DL・変換、`SwiftVRSetup.shared`）, `ModelManagerView.swift`（「拡大モデル」節）, API `GET /api/upscalers`, Localizable | 既存 4 方式（standard / metalFX / Real-ESRGAN 汎用・アニメ、2x、39 フレーム 512²）は旧ビルドの出力とデコード後フレーム単位で完全一致（2026-10-28 実測、`compare_frames`）。進捗は DiT ブロック単位（`h3_vsr_wan_predict` の `on_block`）、中止もブロック境界で効く。DL は `ModelDownloader`（再開可）、変換は `h3_vsr_build_cache`。計画にあった品質優先／速度モードの UI 切替（全注意・精度・overlap）は未実装（エンジンの `full_attention` は Swift ブリッジまで、`dit_overlap` は未配線）。「変換後に原本を削除」は未対応（ReAE を `reae.safetensors` から読み、キャッシュの fingerprint が `transformer/` の実体を要求するため。対応するならキャッシュに ReAE と fingerprint の写しを持たせる） |
| P7 最適化 | 未着手（owner 限定クエリ、NAX 任意形状、tapconv、ReAE F16、2 コンテキスト並走、ANE） | — | — |
| P8 FlashVSR | 未着手 | — | — |

実装で判明し本文を修正した点: 124 フレームは 6 チャンク（§6.4）、偶数層の owner 幅は 16/16/16/12（§6.4）、
patch 入力は (c, ph, pw)・出力は (ph, pw, c) 順（`h3_vsr.h` の `h3_vsr_patch_in_index` / `h3_vsr_patch_out_index`）、
int8 線形は 128 行以上を要求するためトークン行を 128 単位でパディング（`h3_vsr_wan.c`）。

### 付録 A.1: 実重みでの検証（2026-10-29, M5 24 GB）

公式実装（github H-oliday/SwiftVR）を float32 で走らせた参照（`tools/dump_swiftvr_fixture.py`。DiT は 20 GB を常駐させず 1 ブロックずつ読む）と、本エンジンを段階ごとに比べた。題材は GUIN の生成動画 57 フレームを 768x544 に切り出し 384x272 に縮小したもの（2x、トークン格子 24x17、FIRST 28 / MIDDLE 24 / LAST 5 の 3 チャンク）。`tests/test_real_vsr.c`（`H3_VSR_STAGES=host,reae,dit,session`、`H3_VSR_PRECISION=bf16|int8|both`）。

| 段階 | bf16 | int8 | 備考 |
|---|---|---|---|
| 窓の索引表・owner（偶奇 × T=6,7） | 一致 | — | `h3_vsr_window.c` と公式 `_WindowRuntimeMetaCache` が同一 |
| RoPE 表 | max 6.5e-7 | — | 公式は MPS では float32 周波数 |
| 前処理（bilinear, align_corners=False） | 一致（max 0） | — | |
| ReAE エンコーダ（3 チャンク、状態持ち越し） | 139.6〜140.9 dB | — | float32 の丸め誤差水準 |
| ReAE デコーダ（チャンク 0） | 140.8 dB | — | |
| キャッシュ定数 `mod` / `k2` / `v2` | 153 / 73.8 / 76.5 dB | 同左 | k2/v2 は BF16 格納 |
| DiT 単体 v（チャンク 0、MFSWA / 全注意） | 60.0 / 60.4 dB | 51.3 / 51.7 dB | ブロック 0, 1 のトークンは 87.8 / 85.3 dB |
| セッション z_hq（チャンク 0/1/2） | 63.4 / 62.9 / 61.9 dB | 54.7 / 51.6 / 52.3 dB | |
| 8 ビット出力 vs 公式（57 フレーム） | 51.2 dB（max 18/255、>4 が 0.002%） | 49.3 dB（max 31/255、>4 が 0.16%） | 元画像に対しては本エンジン 27.4 dB、公式 27.6 dB（bilinear 39.3 dB。公式出力は元より明るく高コントラストで、目視では鮮明） |
| チャンク時間（28/24/5 フレーム） | 5.8 / 4.6 / 3.9 s | 4.5 / 3.7 / 2.6 s | GPU DiT bf16 4.6 s（3 チャンク計）、int8 7.9 s は dit 単体 3.9/2.4 s と矛盾し要再測 |
| GPU 常駐ピーク（`peak_live_bytes`） | 12.3 GiB | 8.3 GiB | プロセスのピークフットプリントは約 20 GB（MPSGraph の作業域を含む）。P7 で内訳を取る |

実重みで初めて出た不具合 2 件（どちらも修正済み、合成モデルの試験では出なかった）:

1. FFN の GELU（tanh 近似）で Metal の `tanh` が大きな引数に NaN を返し、特定の 16 トークンから全体に広がった。引数を ±20 にクランプ（`h3_geglu_f32` と ReAE の Clamp も同様に保護）。
2. ReAE デコーダの MemBlock 状態をフレーム解像度で確保していた（正しくは潜在解像度）。768x544 で約 17 GB を無駄に確保し、ピークフットプリント 29 GB でスワップと OS による強制終了を招いていた。`h3_vsr_reae_state_new` を修正し、常駐は int8 4.7 GiB（セッション開始時）になった。

アプリ本体でも確認した: 新ビルドを起動し、`GET /api/upscalers` が SwiftVR を使用可（int8 キャッシュ）と報告、`POST /api/upscale-file` で 512x512・39 フレームの生成動画を 2 倍にして 1024x1024 の HEVC（音声そのまま）が約 30 秒で書き出された。bilinear と並べると髪・歯・服の柄が明らかに鮮明で、明るさは入力と同じ。

速度は計画の見積（§6.4: 1920x1088 で int8 6.5 s/チャンク）より遅い。768x544（2,856 トークン）で int8 3.7〜4.5 s なので、1920x1088（12,240 トークン）では線形外挿で 16〜19 s/チャンク。P7 の出発点とする。

### 付録 A.2: P7 速度最適化の記録（2026-10-29, M5 24 GB, int8）

`tests/bench_vsr.c`（`make h3_vsr_bench`、`./h3_vsr_bench [in_w in_h out_w out_h] [chunks] [int8|bf16|auto] [full]`、`H3_VSR_TRACE_STAGES=1` で GPU 完了を待った段階別の時間、`H3_VSR_TRACE_DIT=1` でブロック 1 の演算別の時間）で 960x544→1920x1088（MIDDLE チャンク 12,240 トークン）を計測。注意: 進捗コールバックは CPU 側のエンコード時点で呼ばれるので、コールバック間隔から出した「DiT 2.2 s」は誤りで、GPU 完了で測ると DiT は 7.4 s だった。

| 版 | MIDDLE チャンク | 前処理+ReAE encode | DiT（GPU 完了） | ReAE decode | GPU 常駐ピーク |
|---|---|---|---|---|---|
| 着手時（ReAE は F32 MPSGraph、チャンク一括） | 17.6 s | 1.7 s | — | 13.6 s（DiT の GPU 時間を含む） | 23.7 GiB（フットプリント 29 GB、強制終了） |
| ReAE 全層を TensorOps の「格子」上へ | **9.6 s** | 0.47 s | 7.4 s | 1.65 s | 9.9 GiB |

- ReAE: 新カーネル `h3_vsr_tapconv_bf16_nax`（`h3_shaders.metal`）は活性を「パディング付き格子」（境界行をゼロにした channels-last BF16、チャネルは 64 の倍数に詰め物）に置き、3x3 の 9 タップを行オフセット付きの `matmul2d` 累積で計算する。MemBlock の `cat[x, past]` は第 2 入力、bias・残差・ReLU はエピローグ。TPool は 2 タップ（フレーム行オフセット）、TGrow は 1x1 タップ（TGrow(2) はフレーム 2 分割）、stride 2 の畳み込みは全画素で計算して間引き、Upsample・pixel (un)shuffle・Clamp は格子用の小カーネル。デコーダの 128 ch MemBlock（272x480、2 フレーム）で 22 ms（13.8 TFLOP/s）、F32 MPSGraph の同じ 3 畳み込みは 52 ms（実行器内では 205 ms）。ReAE は中間チャンクで計 2.1 s（encode 0.47、decode 1.65）になり、もはや支配的ではない。
- 精度: 公式 float32 に対し ReAE エンコーダ 56〜57 dB、デコーダ 59 dB（F32 経路は 140 dB）。`tests/test_vsr_tapconv.c`（倍精度参照 65 dB）、`tests/test_vsr_reae.c`（F32 経路との比較 55〜60 dB）、`tests/test_real_vsr.c` の reae 段で検証。`H3_VSR_REAE_F32=1` で従来経路に戻せる。分割幅（4/8/12 フレーム）は時間に影響せず、メモリ下限の 4 を既定にした（`reae_frame_batch` 0 = 4）。
- DiT（ブロック 1、15 窓、14,280 トークン）の演算別 GPU 時間: qkv 32 ms、out 12、norm2+q2 15、out2 15、fc1 53、**fc2 100**（線形計 227 ms ≈ 65%）、窓 SDPA 67、交差 SDPA 9、gather/scatter 11（注意計 87 ms ≈ 27%）、LN・GELU・残差 25 ms。int8 線形は 25 TFLOP/s 前後だが fc2（K=14336）だけ 13 TFLOP/s。SwiftVR の形状（3072/9216/14336）向けに K を静的にしたカーネル（`h3_linear_int8_local_scales_nax_r128_k3072_n*`、`..._k14336_n3072`、full-K 版）を追加したが差は 5〜10% で、fc2 は変わらない（原因は未特定。`H3_VSR_INT8_GENERIC=1` / `H3_VSR_FC2_LOOP=1` で比較できる）。
- owner 限定クエリ（§6.5）を実装した: 各窓の「所有」行だけをクエリにし、所有数が近い窓を最大 3 つのバケットにまとめて（各窓の行はバケット長まで先頭トークンの複製で埋める。複製行は同じ値を同じトークンに書くだけ）バケットごとに `h3_gpu_sdpa_batched_kv_bf16` を 1 回走らせ、`h3_vsr_scatter_rows_bf16` でトークンへ書き戻す。ブロック 1 の注意（gather+SDPA+scatter）は 84 → 64 ms、DiT は中間チャンクで 7.3 → 6.9 s（9.4 → 9.0 s/チャンク）。厳密なクラス分け（7 バケット）は dispatch が増えるだけで同じ時間、バケット 1（全行）は 82 ms。`H3_VSR_QUERY_BUCKETS` で変えられ、`H3_VSR_ALL_QUERIES=1` で以前の全窓 SDPA に戻る。精度は変わらない（実重み v 51.4 dB）。
- 2 コンテキスト並走（§6.6）を実装して測った: デコーダを第 2 の `h3_gpu`（独自のコマンドキュー）で非同期に走らせ（`h3_vsr_reae_decode_begin` / `h3_vsr_reae_finish`）、次チャンクの DiT の後で回収する（出力は 1 チャンク遅れ、`is_last` で全部吐く）。1080p の中間チャンクは 9.19 s → 9.08〜9.33 s で差がなく、encode と出力は競合で 0.5 → 1.2 s に伸びた。DiT が GPU を飽和させているので ReAE を重ねる余地がない（§6.6 で予告したとおり、下限は GPU の合計）。コードは残し既定は 1 コンテキスト（`reae_separate_queue` 1 か `H3_VSR_OVERLAP=1` で有効。`tests/test_vsr_session.c` が両経路の出力一致を確認）。
- fc2（K=14336）の低効率を `tests/bench_int8.c`（`make h3_int8_bench`、`H3_BENCH_SKIP_QUANT=1` で量子化抜き、`H3_BENCH_K_SWEEP=1` で K を振る）で調べた。行の量子化は 6 ms で無関係。同じ N=3072 で K を振ると、K≤8192 で 25〜26 TFLOP/s、12288 で 22.8、14336 で 16.8、20480 で 14.7 と K とともに落ち、ストライドも効く（K=14464 だと 19.3）。試して効かなかったもの: 静的 K のカーネル（5〜10%）、full-K 1 回の `matmul2d`（同じ）、128x256 タイル（fc2 は 114 ms に悪化）、素朴な split-K（Morton 順でないスレッドグループ割り当てだと 16 TFLOP/s しか出ない。Morton 順にしても 14336 ストライドのままでは 67 ms）。効いたもの: **FFN を K 連続な 2 分割**にする（fc1 は出力行の前後半、fc2 は入力列の前後半を読み込み時に詰め直し、各半分を F32 部分和で `h3_linear_int8_split_k_nax_r128` に通して `h3_linear_sum_partials_bf16` で足す）。ブロック 1 の FFN は 166 → 139 ms、中間チャンクの DiT は 6.95 → 6.53 s（`H3_VSR_FFN_WHOLE=1` で元に戻る）。精度は変わらない（部分和は F32）。
- GELU の融合: fc1 の出力に bias と GELU をかけて int8 に行量子化するまでを 1 パスのカーネル `h3_vsr_gelu_quantize_int8_rows`（値をレジスタに保持して最大値→量子化）にまとめ、fc2 の部分和には量子化済み入力を渡す。ブロック 1 の FFN は 139 → 129 ms、中間チャンクの DiT は 6.42 → 6.28 s（`H3_VSR_GELU_SEPARATE=1` で元に戻る）。精度は変わらない。
- FFN の 4 分割も測った（`H3_VSR_FFN_SPLITS=4`）: ブロック 1 の FFN は 2 分割 127 ms、4 分割 131 ms、中間チャンクの DiT は 6.20 s と 6.27 s で、分割の固定費（量子化・部分和の書き出し）が K 短縮の利得を食う。既定は 2 分割のまま。
- P7 の到達点は 1080p 中間チャンク 17.6 → 約 8.2 s（encode 0.46、DiT 6.0〜6.3、decode 1.66）。これ以上は int8 線形そのものの効率（約 25 TFLOP/s）と MPSGraph SDPA に依存する。
- 5 秒クリップ（124 フレーム ≈ 5 チャンク）は約 50〜55 s の見込みで、§1 の目標（M5 で 2 分以内）に入る。LAST チャンクは参照プロトコルどおり前チャンクの潜在を前置した 7 潜在の DiT を走らせるので、残り 1 フレームでも 1 チャンク分かかる。

### 付録 A.3: P5.5 品質評価（2026-10-29, M5, int8）

§9.5 の品質プログラムのうち、この Mac でできる範囲を実施した。**SeedVR2 の参照出力は作れていない**（64 GB 以上の Mac かクラウド GPU が要る）ので、§1 の「SeedVR2-3B の 95%」は未判定。ここにあるのは、元動画に対する忠実度（PSNR/SSIM/LPIPS）、無参照の画質指標（MUSIQ・MANIQA、pyiqa、8 フレーム）、フリッカ（隣接フレームの平均絶対差）と、顔の切り出しの同じ指標、そして目視の所見である。評価セットは `misc/quality/set-v1/`（gitignore）: 生成動画 4 本と実写 1 本を 33 フレームずつ切り出し、元を 2 分の 1（bicubic）に縮小したものを入力とし、元を正解とする。方式の生成は `h3_vsr_run`（`make h3_vsr_run`、精度と注意を選べる）とアプリの API、指標は `tools/quality_metrics.py`、表と顔の A/B 画像は `tools/quality_report.py`。顔の切り出し枠は `crops.txt`。

| クリップ | 元 | 入力 | 内容 |
|---|---|---|---|
| face-photo-tall | 1024x1792 | 512x896 | 生成（実写風の少女、教室） |
| face-photo-square | 512x512 | 256x256（4x 比較は 128x128） | 生成（実写風の女性、室内） |
| face-real-webcam | 1600x1088 | 800x544 | 実写（Web カメラ、男性） |
| leopard-man | 1920x1088 | 960x544 | 生成（豹頭の男、毛皮と肌） |
| anime-face | 672x384 | 336x192 | 生成（アニメ調の少女） |

### anime-face

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| 元（GT） | — | — | — | 70.8 | 0.3943 | 4.698 | — | — | — | — |
| bilinear | 32.56 | 0.9479 | 0.1282 | 46.1 | 0.2609 | 3.550 | 29.10 | 0.1709 | 26.2 | 0.2116 |
| SwiftVR int8 全注意 | 25.68 | 0.8533 | 0.1020 | 73.4 | 0.4306 | 5.602 | 22.26 | 0.1128 | 50.2 | 0.3563 |
| SwiftVR int8 窓 | 25.65 | 0.8530 | 0.1028 | 73.4 | 0.4320 | 5.594 | 22.20 | 0.1134 | 50.2 | 0.3550 |
| Real-ESRGAN アニメ | 31.45 | 0.9280 | 0.0937 | 70.9 | 0.3715 | 4.586 | 28.50 | 0.0638 | 46.8 | 0.3231 |
| Real-ESRGAN 汎用 | 29.15 | 0.9235 | 0.0886 | 74.5 | 0.4552 | 5.306 | 26.13 | 0.0764 | 54.3 | 0.3842 |
| 標準 | 34.04 | 0.9510 | 0.0802 | 53.4 | 0.2675 | 4.176 | 30.77 | 0.1219 | 32.3 | 0.2405 |

### face-photo-square

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| 元（GT） | — | — | — | 77.4 | 0.5439 | 1.403 | — | — | — | — |
| bilinear | 34.16 | 0.9496 | 0.1243 | 60.9 | 0.3134 | 1.030 | 32.72 | 0.1110 | 34.7 | 0.3130 |
| SwiftVR int8 全注意 | 29.29 | 0.8980 | 0.0794 | 78.4 | 0.5477 | 1.333 | 26.63 | 0.0771 | 57.7 | 0.5076 |
| SwiftVR int8 4x→1/2 | 29.47 | 0.8838 | 0.0775 | 77.9 | 0.5143 | 1.162 | 27.38 | 0.0578 | 52.2 | 0.4874 |
| SwiftVR int8 窓 | 29.29 | 0.8980 | 0.0794 | 78.4 | 0.5477 | 1.333 | 26.63 | 0.0771 | 57.7 | 0.5076 |
| Real-ESRGAN 汎用 | 32.53 | 0.9401 | 0.0654 | 78.0 | 0.5140 | 1.073 | 30.33 | 0.0698 | 54.3 | 0.4513 |
| 標準 | 34.85 | 0.9486 | 0.0998 | 69.0 | 0.3473 | 1.098 | 33.77 | 0.0734 | 39.5 | 0.3419 |

### face-photo-tall

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| 元（GT） | — | — | — | 45.9 | 0.3051 | 6.021 | — | — | — | — |
| bilinear | 44.09 | 0.9906 | 0.0289 | 35.7 | 0.3031 | 5.759 | 39.85 | 0.0482 | 38.9 | 0.3313 |
| SwiftVR int8 全注意 | 30.45 | 0.9227 | 0.1042 | 63.5 | 0.3614 | 5.773 | 27.05 | 0.1678 | 63.3 | 0.4354 |
| SwiftVR int8 窓 | 30.09 | 0.9206 | 0.1054 | 64.3 | 0.3611 | 5.908 | 26.89 | 0.1724 | 63.1 | 0.4288 |
| Real-ESRGAN 汎用 | 33.93 | 0.9635 | 0.0889 | 71.5 | 0.4092 | 6.312 | 30.66 | 0.1373 | 66.8 | 0.4843 |
| 標準 | 38.30 | 0.9859 | 0.0217 | 40.8 | 0.3118 | 5.926 | 38.00 | 0.0215 | 42.1 | 0.3330 |

### face-real-webcam

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| 元（GT） | — | — | — | 37.0 | 0.2083 | 0.991 | — | — | — | — |
| bilinear | 44.76 | 0.9917 | 0.0308 | 30.1 | 0.2107 | 0.917 | 45.15 | 0.0328 | 36.6 | 0.2041 |
| SwiftVR int8 全注意 | 30.19 | 0.9280 | 0.1472 | 67.0 | 0.3572 | 1.407 | 30.51 | 0.1363 | 59.9 | 0.3172 |
| SwiftVR int8 窓 | 29.59 | 0.9233 | 0.1580 | 64.8 | 0.3667 | 1.423 | 30.30 | 0.1296 | 59.3 | 0.3094 |
| Real-ESRGAN 汎用 | 33.30 | 0.9575 | 0.1075 | 54.3 | 0.3536 | 1.065 | 35.87 | 0.0765 | 51.8 | 0.2762 |
| 標準 | 40.12 | 0.9835 | 0.0237 | 32.9 | 0.2090 | 1.031 | 41.52 | 0.0292 | 38.0 | 0.1958 |

### leopard-man

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| 元（GT） | — | — | — | 46.2 | 0.1896 | 4.319 | — | — | — | — |
| bilinear | 43.53 | 0.9880 | 0.0455 | 38.0 | 0.1950 | 3.896 | 38.28 | 0.0459 | 48.2 | 0.3571 |
| SwiftVR int8 全注意 | 28.96 | 0.8471 | 0.2781 | 71.6 | 0.3053 | 5.324 | 25.06 | 0.2727 | 75.7 | 0.5133 |
| SwiftVR int8 窓 | 28.81 | 0.8498 | 0.2647 | 70.9 | 0.2956 | 5.454 | 25.11 | 0.2632 | 75.6 | 0.5073 |
| Real-ESRGAN 汎用 | 35.51 | 0.9534 | 0.1099 | 58.0 | 0.2418 | 4.359 | 31.34 | 0.1008 | 67.3 | 0.4101 |
| 標準 | 41.21 | 0.9773 | 0.0285 | 42.1 | 0.1938 | 4.176 | 39.62 | 0.0187 | 51.3 | 0.3586 |

所見（フレーム 16 の顔の切り出しを並べて見た。`misc/quality/set-v1/*/ab-crops.png`）:

- **SwiftVR は細部を描き足す方式なので、元に対する忠実度（PSNR/SSIM/LPIPS）は 5 本中 4 本で最低**だが、無参照の画質指標（MUSIQ・MANIQA）は実写風・実写の顔で最高か同等。生成動画の元自身が柔らかい（MUSIQ 37〜46）ので、「元より鮮明」になる。
- **肌**: 実写（face-real-webcam）では SwiftVR が毛穴・陰影のある肌を描き、Real-ESRGAN は滑らかで蝋のよう、標準はぼける。生成の少女（face-photo-tall）では SwiftVR の肌はやや平滑で明るく、Real-ESRGAN は絵のような硬さ。女性（face-photo-square）では SwiftVR が肌の赤みをやや強める。4x→1/2 は同じクリップで顔の LPIPS が 0.077 → 0.058、フリッカ 1.33 → 1.16 と、2x 直接より穏やかで元に近い（無参照指標は少し下がる）。
- **系統的な明るさの上昇**: SwiftVR の出力は顔の輝度が元より +2.9〜+3.7（0〜255）一様に明るく、コントラスト（標準偏差）は 1.01〜1.07 倍。他方式は ±0.3 以内（Real-ESRGAN は −1.5〜−2.5）。これは内容によらず一定なので、入力の平均輝度に合わせる補正（FlashVSR の色補正と同じ考え）で消せる見込み。未実装。
- **フリッカ**: SwiftVR は実写と豹で元より 1.2〜1.4 倍（描き足した質感がフレーム間で揺れる）。4x→1/2 は元より低い。
- **アニメ**: SwiftVR は平坦な面にノイズ状の質感を足してしまい、顔の LPIPS は Real-ESRGAN アニメ（0.064）の方がずっと良い。アニメ調の動画には Real-ESRGAN アニメを使うべき。
- **全注意 vs 窓（MFSWA）**: 差はわずか（PSNR で +0.1〜0.6 dB、顔の指標は同等）。速度優先の窓で十分。256x256 入力（単一窓）では両者は同一。
- 精度 bf16 vs int8 はディスク容量の都合で同じ比較にかけていない。付録 A.1 の公式実装との一致（bf16 51 dB / int8 48 dB の 8 ビット出力）から、両者の差は約 48 dB 相当で目視では分からない水準。

未了: SeedVR2 との比較、ユーザーのブラインド A/B（肌・髪・毛皮・文字・暗部の採点）、評価セットの実写の追加、DOVER、overlap=1、前処理（bicubic/Lanczos）の比較。

**ブラインド採点（2026-10-29、利用者。フレーム 16 の顔の切り出し、順番シャッフル、元も候補）**

| クリップ | 最良 | 次点 | 指標との関係 |
|---|---|---|---|
| face-photo-tall（生成・少女） | **標準**（リサイズ） | — | 忠実度（PSNR 38.3、顔 LPIPS 0.022）が最良の方式。元が柔らかい生成動画では描き足し（SwiftVR の顔 LPIPS 0.17、Real-ESRGAN 0.14）が好まれなかった |
| face-photo-square（生成・女性） | **SwiftVR 4x→1/2** | — | 顔 LPIPS 最良（0.058）、フリッカ最小（1.16）。2x 直接（0.077）より穏やかな方が選ばれた |
| face-real-webcam（実写） | **SwiftVR 窓** | SwiftVR 全注意 | 無参照指標が最高（顔 MUSIQ 59）。忠実度は最低だが、実写の肌では描き足しが好まれた |
| leopard-man（生成・毛皮と肌） | **Real-ESRGAN 汎用** | — | 忠実度と無参照の中間（PSNR 35.5、顔 MUSIQ 67）。SwiftVR（顔 MUSIQ 76）は明るさ上昇とフリッカが不利に働いたとみられる |
| anime-face（アニメ） | **Real-ESRGAN アニメ** | Real-ESRGAN 汎用 | 顔 LPIPS 最良（0.064）。指標の順位と一致 |

読み: 実写の顔では SwiftVR の肌の描き足しが選ばれ、元が柔らかい生成の顔ではより穏やかな方式（標準、または SwiftVR の 4x→1/2）が選ばれ、毛皮とアニメでは Real-ESRGAN が選ばれた。SwiftVR の 2x 直接が最良になったクリップはない。示唆: (1) SwiftVR は「4x で生成して 1/2 に縮める」経路を既定の候補にする価値がある（LPIPS・フリッカ・採点で 2x 直接より良い）、(2) 系統的な +3/255 の明るさ上昇は補正すべき、(3) 方式の既定は内容で分ける（アニメは Real-ESRGAN アニメ、実写は SwiftVR、生成の柔らかい顔は標準か 4x→1/2）。SeedVR2 との比較は依然として未実施。

**色補正と 4x→1/2 の実装後の再計測（2026-10-29、同じ評価セット、int8 窓。`int8-window` = 補正なし、`int8-match` = 色補正あり（既定）、`int8-4x-half` = 同じ入力から 4 倍で復元して Lanczos で 1/2（色補正あり）、`standard` = Lanczos リサイズ（アプリの標準と同じ処理、ここでは無圧縮）**

| クリップ | 変種 | PSNR | LPIPS↓ | MUSIQ | フリッカ | 顔 PSNR | 顔 LPIPS↓ | 顔 MUSIQ |
|---|---|---|---|---|---|---|---|---|
| face-photo-tall | 補正なし / 補正あり | 30.09 / 31.43 | 0.105 / 0.105 | 64.3 / 64.7 | 5.91 / 5.90 | 26.89 / 27.40 | 0.172 / 0.172 | 63.1 / 63.5 |
| face-photo-square | 補正なし / 補正あり / 4x→1/2 | 29.29 / 30.08 / **33.85** | 0.079 / 0.077 / **0.042** | 78.4 / 78.4 / 78.3 | 1.33 / 1.33 / **1.03** | 26.63 / 26.97 / **30.91** | 0.077 / 0.075 / **0.039** | 57.7 / 57.7 / **64.1** |
| face-real-webcam | 補正なし / 補正あり | 29.59 / 30.51 | 0.158 / 0.156 | 64.8 / 65.6 | 1.42 / 1.43 | 30.30 / 31.41 | 0.130 / 0.128 | 59.3 / 59.2 |
| leopard-man | 補正なし / 補正あり | 28.81 / 29.43 | 0.265 / 0.262 | 70.9 / 70.7 | 5.45 / 5.39 | 25.11 / 25.27 | 0.263 / 0.260 | 75.6 / 75.6 |
| anime-face | 補正なし / 補正あり / 4x→1/2 | 25.65 / 25.90 / **29.59** | 0.103 / 0.102 / **0.079** | 73.4 / 73.1 / 74.8 | 5.59 / 5.59 / **5.19** | 22.20 / 22.32 / **26.65** | 0.113 / 0.114 / **0.063** | 50.2 / 50.7 / **57.2** |

- 色補正（チャネルごとの平均を入力に合わせる。補正量は −0.8〜−4.7/255、青が最も大きい）は PSNR を +0.3〜+1.3 dB 上げ、無参照指標とフリッカは変えない。既定で有効（`h3_vsr_open_params.color_match`、`H3_VSR_NO_COLOR_MATCH=1` で無効）。
- 4x→1/2 は、測れた 2 本（256x256 と 336x192 の入力）で全指標が 2x 直接より良く、アニメでは顔の LPIPS が Real-ESRGAN アニメ（0.064）と同等になった。ただし 4 倍の中間出力は GPU メモリを食い、1024x1792・1600x1088・1920x1088 を 2 倍にする 3 本では `h3_vsr_check_requirements` が拒否した（推定 45〜53 GB。推定式は ReAE の 4 フレーム分割に合わせて 1.5 KB/画素に改めたが、それでも 3840x2176 は約 33 GB）。24 GB の Mac では、4 倍の中間出力が約 250 万画素まで（入力 448x256 → 1792x1024 など）が目安。672x384 → 2688x1536 は `H3_VSR_IGNORE_BUDGET=1` で無理に走らせるとフットプリント 20.3 GB でスワップし、中間チャンク 26 s かかった（推定 18.2 GB は予算 19.1 GB の 90% を超え、既定では拒否される）。アプリでは「4倍で生成してから縮小する」の切替（2 倍のときだけ表示、既定オフ）で選び、入らないときは黙って 2 倍直接に戻る。時間は約 4 倍（anime 2.1 → 8.8 s、square 2.2 → 9.2 s）。

**濡れた肌の顔アップ（2026-10-29、利用者の見本画像に合わせて生成したクリップ、M5、int8 窓）**

利用者が示した比較画像（濡れた金髪、水滴の付いた光る肌、暖色照明、笑顔と斜め向き）に合わせ、同じプロンプトと seed（20261029）で 544x960（portrait-large、生成 964 s）と 256x448（portrait-small、140 s）の 5 秒クリップをアプリで生成した。評価セット: `~/Movies/H3cApp/skin-eval-2026-10-29/`（クリップ、切り出しシート、スクリプト）。2 通りで測った。

(a) 実際の手順: 生成したクリップをアプリの API から 5 方式で 2 倍にし、第 46 フレームの顔の切り出し（544x960 は頬・鼻・口元の 600x700、256x448 は目元から口元の 400x460）で無参照指標を取った（正解はない）。544x960 の 4 倍経由は中間 2176x3840 がメモリ予算を超えて黙って直接 2 倍に落ち（出力はフレームまで同一、46 s）、256x448 では走った（11 → 39 s）。

| 入力 | 方式 | 時間 | 顔 MUSIQ↑ | 顔 MANIQA↑ | 顔フリッカ↓ |
|---|---|---|---|---|---|
| 544x960 | 標準 / MetalFX | 1 s / 1 s | 56.3 / 57.9 | 0.247 / 0.228 | 5.33 / 5.66 |
| 544x960 | Real-ESRGAN 汎用 | 13 s | 62.5 | 0.268 | 5.19 |
| 544x960 | SwiftVR 2x | 47 s | **70.1** | **0.331** | 6.23 |
| 256x448 | 標準 / MetalFX | 1 s / 1 s | 56.7 / 59.2 | 0.391 / 0.385 | 4.87 / 5.25 |
| 256x448 | Real-ESRGAN 汎用 | 3 s | 65.8 | 0.419 | 4.85 |
| 256x448 | SwiftVR 2x / 4x→1/2 | 11 s / 39 s | **72.4** / 72.2 | 0.526 / **0.608** | 5.81 / 5.54 |

(b) 正解あり: 生成クリップの第 30〜62 フレームを正解とし、bicubic で 1/2 にした入力から各方式で戻して比較（`h3_vsr_run`、Real-ESRGAN は x264 crf 6 の入力をアプリで）。544x960 でも 4 倍→1/2（中間 1088x1920）が入る。

| 正解 | 変種 | PSNR | LPIPS↓ | フリッカ | 顔 PSNR | 顔 LPIPS↓ | 顔 MUSIQ | 顔 MANIQA |
|---|---|---|---|---|---|---|---|---|
| 544x960 | Lanczos / Real-ESRGAN | 43.09 / 37.01 | **0.052** / 0.080 | 3.31 / 3.46 | 40.86 / 35.21 | **0.070** / 0.090 | 52.9 / 65.5 | 0.408 / 0.445 |
| 544x960 | SwiftVR 窓 / 全注意 | 31.55 / 31.70 | 0.119 / 0.115 | 4.13 / 4.07 | 30.25 / 30.39 | 0.143 / 0.141 | 71.0 / 70.4 | 0.513 / 0.518 |
| 544x960 | SwiftVR 4x→1/2 | 33.98 | 0.107 | 3.98 | 32.51 | 0.110 | **71.7** | **0.604** |
| 256x448 | Lanczos / Real-ESRGAN | 39.25 / 34.13 | 0.084 / 0.094 | 3.21 / 3.49 | 37.60 / 32.97 | 0.100 / 0.100 | 49.5 / 59.8 | 0.446 / 0.469 |
| 256x448 | SwiftVR 2x（窓＝全注意） | 30.01 | 0.102 | 4.21 | 28.79 | 0.095 | 69.8 | 0.564 |
| 256x448 | SwiftVR 4x→1/2 | 33.28 | **0.057** | 3.81 | 31.92 | **0.060** | **72.3** | **0.602** |

等倍の切り出しで確認したこと（ユーザーの見本の SeedVR2 列に相当する「毛穴まで出る」仕上がりかどうか）:
- **肌の質感**: SwiftVR は毛穴と水滴を一番はっきり描き、無参照指標でも全方式の中で最高（顔 MUSIQ 70〜72、Real-ESRGAN 60〜66、標準 50〜57）。Real-ESRGAN は元の形を保ったまま少し締める程度で毛穴は増えない。標準・MetalFX は柔らかいまま。
- **描き直し**: 544x960 の SwiftVR 2x 直接は、目の周りを描き直す。まつ毛の形が変わり、元には無い涙袋の皺と、目の下の肌に同心円状の波紋が加わる（(a) の実際の手順でも (b) でも出た。全注意でも同じ）。4x→1/2 は目の形が正解に近く波紋も消えるが、涙袋の線はうっすら残る。Real-ESRGAN と Lanczos は形を変えない。
- **4x→1/2 の効き**: 544x960 では顔 LPIPS 0.143 → 0.110、PSNR +2.3 dB、フリッカ −4%、256x448 では顔 LPIPS 0.095 → 0.060 で、無参照指標は落ちない。この見本のような肌のアップには 2x 直接より 4x→1/2 が良い。ただし 544x960 ではアプリの 4 倍経路がメモリで入らず（中間 8.4 M 画素）、(b) は半分の入力だから入ったにすぎない。
- **忠実度**: 正解あり比較では Lanczos が最良で、SwiftVR は PSNR で 8〜12 dB 低い。「元の生成フレームをそのまま大きくしたい」なら標準、「毛穴や水滴を見せたい」なら SwiftVR、と目的で分かれる。

結論: 見本の SeedVR2 列に一番近い質感は SwiftVR（できれば 4x→1/2）で出る。544x960 以上の生成動画で 4x→1/2 を使えるよう、下の「タイル分割」を実装した。2x 直接は目元を描き直す癖があるので、顔のアップでは利用者が確認して選ぶ前提。

**タイル分割（2026-10-29, M5 24 GB, int8）。** 1 チャンクの見積りが GPU の推奨作業領域（この機種 19.07 GB）の 90% を超えるとき、エンジンがフレームを 32 px 格子上の等しいタイル（重なり 128 px 以上）に分け、タイルごとに ReAE の状態と DiT を持って復元し、重なりを線形の傾斜（軸ごとに正規化、合計 1）で混ぜる（`h3_vsr_tile_plan_build` / `h3_vsr_tile_weights`、カーネル `h3_vsr_tile_accumulate_f32`、`h3_vsr_check_requirements` が分割を返す）。見積りには各タイルの ReAE 状態（約 210 B/画素）と F32 の合成フレームを加えた。あわせて、チャンクごとに `decoded` を二重に確保していた漏れ（1920x1088 でチャンクあたり約 0.7 GB、常駐が 10.6 → 12.1 GiB と増え続けていた）を直した。

| 条件 | 分割 | 見積り | 常駐ピーク | 最初のチャンク | 以後 | 備考 |
|---|---|---|---|---|---|---|
| 544x960 → 2176x3840（4 倍） | 自動 = 2x2 の 1152x1984 | 16.4 GB | 16.0 GiB | 44 s | 約 40 s | 1x3（2176x1376）は 18.6 GB と見積もられ除外。強制すると 17.4 GiB で同じ速さ |
| 272x480 → 1088x1920 | 1 枚 | 11.5 GB | 11.7 GiB | 9.7 s | 8.2 s | 漏れ修正後は 3 チャンクとも同じピーク |

品質（gt-large, 272x480 → 1088x1920 → 1/2 を正解と比較）: 1 枚 PSNR 33.98 / 顔 LPIPS 0.110 / 顔 MUSIQ 71.7 に対し、2x2 強制は 33.88 / 0.108 / 71.2。タイル版と 1 枚版の間は 40.9 dB（2x2）〜44.7 dB（2x1）で、差は帯ごとに 1〜2/255 と均一（重なりの帯に偏らない）。544x960 の 4 倍フレーム（上の自動分割）を目視で確認: つなぎ目の帯（頬、口元の角）に線や段差は見えない。アプリの経路相当（4x タイル→Lanczos 1/2）と 2x 直接の比較（正解なし、同じ顔切り出し）: 顔 MUSIQ 55.4 → 57.3、MANIQA 0.389 → 0.448、顔フリッカ 6.09 → 5.69。眉の描き直し（2x 直接で増える濃い毛）は 4x 経由で元の形に近い。

アプリの API 経由（拡大の方法「SwiftVR 4倍で生成してから縮小」= `swiftVRFourX`。以前の「4倍で生成してから縮小する」トグルはこの方法に置き換えた。`HalvingChunkScaler`、4 倍タイル→標準リサイズ）では、この 544x960・121 フレームのクリップが 243 秒（2x 直接は 42 秒、約 5.5 倍）。別の重い処理（解析スクリプト）と同時に走らせた 1 回目は 15 分かかったので、GPU と RAM を独占させること。合成テストは `tests/test_vsr_session.c`（2x2 強制で全フレーム決定的に出力、1 枚版と 26 dB、収まらない分割は拒否）と `tests/test_vsr.c`（配置と重みの和）。

### 付録 A.4: Open-H3 再生成の判定実験（2026-10-31, M5 24 GB, ブランチ regen-api）

§11.3 の判定実験を実施した。実装（main から分岐した `regen-api`）: (1) `h3_params.initial_latent_path` と `video_sigmas[]`（開始潜在と任意の σ 列。潜在ファイルの音声はクリーン固定で持ち越し）、(2) `h3_latent_upscale.c`（LBH の 3D 畳み込み潜在アップスケーラ。作者ノード fp32 と 122〜126 dB、rel-L2 2e-6 で一致。960×544・37 潜在フレーム → 1344×768 が 19 秒、ピーク 5.8 GB）、(3) LoRA ローダの AdaLN 曲線分岐対応（`adaln_proj.linear` の `lora_A/B` と NVIDIA の `curve_A/B` を、曲線形式チェックポイントの [96768, 8] 射影に `B·A` として加える。`H3_LORA_NO_ADALN=1` で無効）、(4) CLI の `H3SPIKE_INITIAL_LATENT` / `H3SPIKE_SIGMAS` / `H3SPIKE_LORAS`、`h3_latent_upscale`、`tools/dump_latent_upscaler_fixture.py`。

**確認した事実**

- 潜在の正規化: 作者ノードの `LATENTS_MEAN/STD` 24 値は VAE 設定の `latents_mean/std` と**完全一致**（差 0）。ComfyUI の H3 潜在（`samples`）は VAE ラッパが正規化した DiT 空間の潜在で、ノードはそれにもう一度同じ定数をかけてからネットに入れ、出力を逆正規化する。NVIDIA のパイプラインも正規化済みの潜在をノードに渡しており同じ。本実装も `.h3lt` の潜在に対して同じ手順を取る。
- **Sharp の AdaLN 曲線分岐は空である。** LMS Sharp v1.0 の LoRA ファイルも NVIDIA の `sharp_curve_branches.safetensors` も、50 ブロック全ての `B` が厳密に 0（`A` は非零、max 0.354）。したがって分岐の有無で出力は変わらず（本実装でも有無でビット一致を確認）、公開パイプラインでも恒等である。§11.3-5 の「分岐なしで不採用と判断しない」条件は、分岐が結果に影響しないことをもって満たされる。
- Kijai の精細化 LoRA（rank 384、BF16、ComfyUI 配置、QKV 融合・FC1 入れ替え済み）、LightX2V Turbo v1.2、LMS Sharp r64 は既存のローダで 312 線形層に当たる。
- H3 の 1344×768・124 フレーム（37 潜在フレーム、約 3.7 万トークン）は、int8 キャッシュ方式で **DiT 1 ステップ 282 秒**、VAE デコード 181 秒。3 ステップの再生成は 1027〜1087 秒、ピークフットプリント 14.3〜15.4 GB（LoRA 3 本）、LoRA なしで 8.9 GB。

**条件**: モデルはアプリの有効モデル（MiniMax-H3 GGUF、Q4_K pruned、曲線形式 AdaLN）、int8 キャッシュ方式、seed 1。ドラフト 2 本: 「osaka」（960×544・124 フレーム、抽象的なデジタル映像、元の生成物の潜在）、「face」（1024×576・124 フレーム、skin-eval の濡れ髪の女性の接写、Turbo 4 ステップで 503 秒）。目標 1344×768。A = 潜在拡大 → σ 0.909375/0.857143/0.666667/0 の 3 ステップ、LoRA = 精細化 r384 + Turbo v1.2 + LMS Sharp r64。B = SwiftVR 2x（int8、窓注意）→ Lanczos で 1344×768（69 秒）。C = ドラフトを Lanczos で 1344×768。指標は `tools/quality_metrics.py`（忠実度は C を基準にした「拡大ドラフトへの近さ」、MUSIQ/MANIQA は無参照、フレーム間差は隣接フレームの平均絶対差、顔は 380,220,560,320 の切り出し）。

| osaka | PSNR vs C | LPIPS | フレーム間差 | MUSIQ | MANIQA | 時間 |
|---|---:|---:|---:|---:|---:|---:|
| C 単純リサイズ | — | — | 7.04 | 44.6 | 0.282 | — |
| B SwiftVR | 29.9 dB | 0.22 | 5.58 | 51.3 | 0.359 | 163 s（A2 と同時実行） |
| A0 LoRA なし 3 ステップ | 22.6 dB | 0.42 | 8.9 | 44.2 | 0.248 | 1036 s |
| A1 精細化+Turbo+Sharp | 20.0 dB | 0.51 | 14.1 | 51.2 | 0.281 | 1032 s |

| face | PSNR vs C | LPIPS | フレーム間差 | MUSIQ | MANIQA | 顔 LPIPS | 顔 MUSIQ | 顔 MANIQA | 時間 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C 単純リサイズ | — | — | 6.26 | 68.5 | 0.262 | — | 65.3 | 0.394 | — |
| B SwiftVR | 30.8 dB | 0.126 | 6.52 | 76.6 | 0.361 | 0.162 | 70.8 | 0.489 | 69 s |
| A1 精細化+Turbo+Sharp（3 ステップ） | 21.5 dB | 0.275 | 7.05 | 74.0 | 0.328 | 0.291 | 69.1 | 0.501 | 1027 s |
| A1 と同じ、分岐なし | A1 とビット一致 | | | | | | | | 1087 s |
| A2 同 LoRA、2 ステップ σ 0.7 | 24.9 dB | 0.167 | 6.83 | 71.7 | 0.299 | 0.201 | 65.8 | 0.462 | 894 s（B と同時実行） |

**目視**: A は構図と動きを保ったまま細部を描き起こす（osaka では配線図や文字列、face ではまつ毛・毛穴・肌理）。その分、元にあった細部が置き換わる（左目下の水滴が消える）。B は元の細部を保ったまま鮮明化し、肌理も十分に出る。osaka では A のフレーム間差が 2 倍になり、σ 0.909 の描き直しが揺れとして見える。

**判定（§11.3-5 の基準）**: 1344×768 で、顔・肌について A は B（SwiftVR）に明確には勝っていない。無参照指標は B が上（MUSIQ 76.6 対 74.0、MANIQA 0.361 対 0.328、顔 MUSIQ 70.8 対 69.1）、顔 MANIQA のみ A がわずかに上（0.501 対 0.489）、忠実度と揺れは B が明確に良く、時間は 15 倍の差。2 ステップ σ 0.7 の A2 は元に近づく（LPIPS 0.167）が無参照指標はさらに下がり（MUSIQ 71.7、顔 65.8）、B には届かない。osaka でも B は A1 と同じ MUSIQ（51.3 対 51.2）で、MANIQA・忠実度・揺れは B が上。Sharp 分岐は恒等なので条件 2 は充足。**よって 1080p / 2K 対応（Q 分割注意・疎注意）には進まない。** 再生成は「アップスケーラ」ではなく「同じ構図で細部を描き直す」機能として別に価値があり得る（ドラフトを 768p に仕上げる用途）。これを製品化するかは別途判断する。

**未検証・残課題**: 開始 σ 0.8、4x 相当の比較、他の題材（毛皮・文字）、NVIDIA の統合 FP32 LoRA（rank 576）との差、ユーザーのブラインド目視、M1〜M4 の時間。潜在アップスケーラの `make test` 向け合成重み試験は未追加（実重みの参照比較のみ）。
