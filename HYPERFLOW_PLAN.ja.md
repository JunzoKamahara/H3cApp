# HyperFlow H3 Flash（13.7B・4/6/8ステップ）対応 — 計画書

作成 2026-11-08（基準コミット main ba59ec7）。HyperFlow H3 Flash（Video Rebirth による MiniMax-H3 の派生 Transformer）の **GGUF 版**を、H3cApp の C/Metal エンジンで生成できるようにする計画。Python は使わない。

- モデル: https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash （`fl2va_gguf/`, `ref2va_gguf/`）
- コード: https://github.com/Video-Rebirth/hyperflow-h3-flash （Apache-2.0。`integrations/stable-diffusion.cpp/hyperflow-two-time.patch` は MIT、ComfyUI ノード `Video-Rebirth/ComfyUI-HyperFlow-Flash` は GPL-3.0）
- 重みのライセンス: MiniMax H3 Community License Agreement の派生物（地域制限・「MiniMax H3」表示・AI 生成の明示は元モデルと同じ）

## 1. 目的と期待値

- HyperFlow は MiniMax-H3 の Transformer を 13.7B（fl2va）/ 13.9B（ref2va）に圧縮し、CFG なし・4/6/8 ステップで映像＋音声を出す。テキストエンコーダと映像・音声 VAE は元の H3 と同じファイルを使う。
- H3cApp には既に Turbo（4 ステップ LoRA）と DMAD がある。**M5 では速度の利点は小さい**: 注意計算は隠れ次元 5376・56 ヘッドのままで、1344×768 では注意が計算の大半を占める。1 ステップあたり元の H3 の 8〜9 割と見積もると、6 ステップは Turbo の 4 ステップより遅く、4 ステップで 1〜2 割速い程度。
- 期待する利点は 2 つ。
  1. **M1〜M4 の SSD ストリーミング**で 1 ステップに読む量が約半分（Transformer Q4_K_M 9.05 GB 対 約 17 GB）。
  2. **少ステップでの品質**（蒸留済みモデル対 Turbo LoRA。測るまで不明）。
- M5 の常駐メモリは、今の int8 常駐（約 18 GiB）に対して約 15 GiB の見込みで大差はない。
- 採用はフェーズ 5 の実測で決める。Turbo/DMAD/標準 20 ステップに品質で並ばなければ「試験的」止まりか不採用とし、結果をこの文書の付録に残す。

## 2. 確定した事実

### 2.1 HyperFlow の GGUF（実物ヘッダ、2026-11-08、Q4_K_M）

fl2va `minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf`（9.05 GB、647 テンソル）、ref2va 同 `ref2va`（9.22 GB、683 テンソル）。どちらも GGUF v3、**メタデータは 0 件**（`general.architecture` なし、alignment 既定 32）。

- **テンソル名は ComfyUI の MiniMax-H3 配置**（`blocks.N.attn.*`、`blocks.N.mlp.*`、接頭辞なし）。dense なブロックは融合 `attn.qkv_proj.weight` [21504, 5376]、`attn.out_proj.weight`、`mlp.fc1.weight` [28672, 5376]、`mlp.fc2.weight`（Q4_K または Q6_K）。
- **削除ブロックは詰めて番号が振り直されている。** fl2va は `blocks.0`〜`blocks.46` の 47 個（config の 6・21・22 を除いた順）、ref2va は `blocks.0`〜`blocks.48` の 49 個。欠番は末尾（47〜49 / 49）だけで、穴はない。
- **低ランク層**: `blocks.N.attn.{q,k,v}_proj.{down,up}.weight`、`attn.out_proj.{down,up}.weight`、`mlp.fc{1,2}.{down,up}.weight`。`down` は [rank, in]、`up` は [out, rank]、bias なし、`y = up(down(x))`。**ランクは 256 の倍数に詰めてあり、すべて Q4_K/Q6_K で量子化**（K 量子化のブロック 256 に合う）。Q・K・V は別々で、ランクも別（例: fl2va block 1 は q 768 / k 768 / v 1024 / out 1792 / fc1 1792 / fc2 2560、最小 256、最大 3328）。
  - fl2va の低ランクブロック（GGUF 番号）: 1, 2（fc2 は dense）, 4, 6〜17, 19, 20、および 46 の fc2 だけ。残り 29 ブロックは dense。
  - ref2va: 1, 4〜8, 10〜18, 20〜23 のあたりが低ランク（ヘッダの先頭 32 MB で確認できた範囲。実装時に全件を列挙する）。
- **時間条件**: `time_embedder.proj_in` [5376, 256] + bias、`proj_out` [**16**, 5376] + bias、`endpoint_time_embedder.*` も同形、すべて F32。`blocks.N.adaln_proj.linear.weight` は **[96768, 16]** BF16 + bias [96768]、`final_layer.adaln_proj.linear.weight` は [10752, 16]。曲線表 `adaln_t_table` はない。`hyperflow_gate` F32[1] = **0.25**。
- `hyperflow_sigmas` F32[7] = 1, 0.982655942, 0.914642572, 0.674755692, 0.385223746, 0.159055322, 0（6 ステップの映像側、shift 12 適用後）。
- そのほかは元の H3 と同じ: `video_patch_proj`、`audio_patch_proj`、`condition_proj`、`token_refiner.blocks.{0,1}`（融合 QKV、Q4_K/Q6_K）、`final_layer.{video_out,audio_out,norm}`、`rope.inv_freq`。

### 2.2 HyperFlow の算法（公開コード transformer.py / schedule.py / sd.cpp パッチ / ComfyUI ノードの読み）

- 2 時刻の条件: 各トークンに (t, r)。`temb = E_t(t) + gate × (E_r(r) − E_t(t))`（gate 0.25）。t = 1 − σ_i、r = 1 − σ_{i+1}（そのステップの到達時刻）。最終ステップは r = 1。条件トークン（キーフレーム・参照画像/動画/音声）は r = t。テキスト行は映像の (t, r)。
- 映像と音声は同じ基準格子に別の shift（映像 12、音声 3）: `σ' = shift·s / (1 + (shift−1)·s)`。基準格子は 4: (1, .681872, .205019, .036496, 0)、6: (1, .825217, .471725, .147401, .049626, .015517, 0)、8: (1, .931506, .839236, .703462, .5, .296538, .160764, .068494, 0)。他のステップ数は拒否。CFG = 1、Euler。最初のステップは映像も音声も t = 0 だが r が違う。
- AdaLN 投影の直前に SiLU が入るかは、公開コードからは確定できない（基本モデルの `adaln_proj` をそのまま呼んでいる）。**フェーズ 0 の参照ダンプで確定する。**
- INT8 convrot 版（safetensors、Hadamard 回転付き W8A8）は GGUF には入っていない。対象外。

### 2.3 h3c-app 側（main ba59ec7）

- サンプラー: CFG なし・Euler（`h3_host.c:706`、GPU `h3_euler_bf16`）、映像と音声は同じステップで別の Δσ。HyperFlow の更新則と同じで、**サンプラー本体は変更不要**。
- スケジュール: `h3_serving_schedule_build_shifted(evaluations, audio_shift)`（`h3_host.c:184`）は等間隔の基準格子に shift 12/3 を掛ける。`params->video_sigmas` は映像だけを上書き（`h3.c:2967`）。
- AdaLN: `prepare_rows`（`h3_dit_schedule.c:81`）は t だけで行を割り当て、映像と音声の t が等しければ 1 行を共有する。`time_embeddings`（`:173`）は proj_in → SiLU → proj_out → SiLU、出力幅は `H3_DIT_TIME_DIM` 2688 固定。曲線形の `curve_projection`（`:397`）は任意の係数 [rows, rank] × [out, rank]ᵀ + b を F32 で計算する。**rank 16 の HyperFlow 投影にそのまま使える。**
- ブロック: `h3_dit_block`（`h3_dit.c:38`）は融合 QKV [INNER×3, HIDDEN]（ヘッドごとに [q;k;v]）、out、fc1、fc2 と int8 版。`run_block`（`:3090`）。int8 QKV 経路は query/key/value を別バッファに出す。トークン精製器 `run_refiner_block`（`:1301`）は非融合の素直な列で、低ランク版の雛形になる。
- 読み込み: 常駐 `load_block`（`:673`）、SSD ストリーミング `prepare_stream_layer`（`:829`、4 行列固定、スロット 2 組、staging は最初のブロックの大きさ）、注意キャッシュ（50 レコード固定）。`h3_gpu_tensor_read_file_mode`（`h3_gpu.m:870`）は要素数の完全一致を要求。重みストアの `qkv_split` 判定（`h3_weights.c:140`）は「GGUF で `video_patch_proj.weight` がある」で立つので、HyperFlow の融合 QKV も同じ扱いになる。
- ブロック飛ばし: `block_active[50]`、`configure_active_blocks`（`:1719`）、`configure_gate_ranked_blocks`（`:1743`、2〜48 を対象、スケジュールのブロックが欠けると早期 return）。スケジュール・注意キャッシュ・LoRA 表は 50 ブロック全部を前提。
- 行列積: `h3_gpu_linear_bf16` は MPSGraph（rows≥32, in≥256, out≥256）、16×16 スカラーカーネル（任意形状）、M5 tensor-op カーネル（in%32, out%64。既定は qkv/attn の形だけ、`h3_gpu_set_nax_linear_generic` で汎用化）。int8 線形は in/out が 128 の倍数。**ランクが 256 の倍数なので、どの経路の条件も満たす。**
- LoRA は dense 重みに差分を足す方式（`h3_lora.c:811`）。低ランク層には当てられない。Turbo/DMAD/利用者の LoRA は HyperFlow には使えない。
- アプリ: モデルはフォルダ（`FL2VA/transformer/model.gguf`、任意で `Ref2VA/`）。`h3_gguf_inspect`（`h3_gguf.c:438`）は `video_patch_proj.weight` で DiT と判定、`dense_adaln` は `blocks.0.adaln_proj.linear.weight` の shape[1]==2688。Ref2VA はファイル名の `ref2v` で判定。`h3_params` にモデル識別・CFG の項目はない。`SpeedMode {quality, fast, fastest}`、`ComputeMode {attentionCache, resident, ssdStreaming}`。記録 `ProjectDraft` にモデル識別はない。常駐メモリの見積りは定数（`h3.c:2061`）。
- 参照環境: CUDA 機（RTX PRO 4000、LTX 評価で使用）は SSH で届かず、ユーザーがスクリプトを持ち込んで実行し、結果を NAS 経由で戻す。

## 3. 設計

### 3.1 方針
- **HyperFlow は「モデルの性質」として扱い、速度プリセットにはしない。** 重みストアの検出で `hyperflow` を立て、そのときだけ (a) 2 時刻 AdaLN、(b) 固定格子のスケジュール、(c) 低ランク対応と 47/49 ブロックのローダー、(d) 非対応機能の拒否、が有効になる。通常の H3 の経路は bit 一致で不変（既存テストで固める）。
- 既存部品を使う: サンプラー、`curve_projection`、`block_active`、`run_refiner_block` 型の非融合列、`h3_gpu_linear_bf16`、`h3_gguf_inspect`、`LoRAProvenance` の出典表示の型。
- 順序は **常駐（M5）→ SSD ストリーミング（M1〜M4）**。注意キャッシュ・LoRA・再ノイズ・denoise/core reuse・token reduction・層数減らしは HyperFlow では無効（理由付きエラー）。

### 3.2 検出とパラメータ（h3_weights.c / h3_gguf.c / h3.c / h3.h / h3_dit.c）
- `h3_hyperflow_info`（新）: `two_time`（`endpoint_time_embedder.proj_in.weight` あり）、`gate`（`hyperflow_gate`）、`time_dim`（`time_embedder.proj_out.weight` の shape[0]）、`block_count`（`blocks.N.norm1.weight` が連続して存在する数: 47 / 49）、ブロックごとの dense/低ランクと各ランク（`down` の shape[0]）、`model_sigmas`。`h3_gguf_inspect` にも `hyperflow`・`time_dim`・`block_count` を足し、アプリの取り込み判定で使う。
- 混在（q だけ低ランク等）、融合 `qkv_proj` と分離 `q_proj` の両方がある形、ランクが 256 の倍数でない形は明示的なエラーで拒否（実物にはない）。
- `h3_params` に新項目は足さない。HyperFlow のときは `steps ∈ {4, 6, 8}` 以外、`loras`、`sampler=RENOISE`、`denoise_reuse>1`、`core_reuse>1`、`token_reduction`、`dit_layers` の変更、`video_sigmas` を検証で拒否し、`h3_model_info` にモデル種別（`hyperflow`、`hyperflow_steps`）を出す。`forced_audio_clean`（リップシンク・続きの生成）は許可し、音声行は t = r = 1。
- `h3_dit_prep` と prepared-DiT のキーに `hyperflow` を含める。

### 3.3 スケジュール（h3_host.c / h3.c）
- `h3_hyperflow_schedule(steps, h3_sigma_schedule *out)`（新）: 基準格子の定数表に映像 shift 12・音声 shift 3。`h3.c:2961` で HyperFlow のときはこれを使う。GGUF の `hyperflow_sigmas` は 6 ステップ時に照合し、1e-5 以上ずれていれば警告。
- テスト: 3 格子の shift 後の値を公式 README の `--sigmas` の値と 1e-6 で照合（6: 1, .982655942, .914642572, .674755692, .385223746, .159055322, 0。実物の `hyperflow_sigmas` と一致を確認済み）。

### 3.4 2 時刻 AdaLN（h3_dit_schedule.c/.h）
- `h3_dit_schedule_precompute_two_time(weights, gpu, sigmas, visual_condition, audio_condition, gate, block_count, …)`（新）。
  - 行の同一性を **(t, r)** にする。映像 i: (1−σv_i, 1−σv_{i+1})、音声 i: (1−σa_i, 1−σa_{i+1})、視覚条件行 (0.999, 0.999)、音声条件行 (1, 1)。`prepare_rows` を (t, r) 対応に一般化し、通常経路（r = t）では今と同じ行割り当てになること（`test_real_dit_schedule` の 40 行配置）を保つ。条件行の別名化は (t, r) が一致するときだけ。
  - `time_embeddings` を接頭辞と出力幅を引数に取る形に一般化し、E_t(t)・E_r(r) を F32 で計算 → 混合 → （SiLU: フェーズ 0 で確定）→ `curve_projection` で [96768, 16] と bias に掛ける（F32 計算、BF16 格納）。最終層も同様。
  - ブロック数は `block_count`。`h3_dit_schedule_gate_score`・`configure_gate_ranked_blocks`・`h3_dit_schedule_prune` は `block_count` を上限にする。
- 行マップ・最終層マップ・消費側（`run_block`）は無変更。

### 3.5 ブロックのローダーと実行（h3_dit.c / h3_gpu.m / h3_shaders.metal）
- `h3_dit_block` に低ランク用の任意フィールド: `qkv_down`（q/k/v の `down` を縦に積んだ [rq+rk+rv, HIDDEN]）、`q_up/k_up/v_up` [INNER, r]、`out_down/out_up`、`fc1_down/fc1_up`、`fc2_down/fc2_up`、各ランク。NULL なら dense（今の経路）。
- ブロック数 47/49: `configure_active_blocks` の前に `block_count` を反映し、以後の全経路（読み込み・ストリーミング・前取り・順位付け）で `block_count` を上限にする。`H3_DIT_BLOCKS` 50 は配列の上限のまま。
- `run_block` の低ランク分岐（`run_refiner_block` の列を手本に）:
  1. `linear_bf16(lr_tmp, mod_attention, qkv_down, rows, HIDDEN, rq+rk+rv)`
  2. `linear_bf16(query, lr_tmp の q 部分, q_up)`、k、v も同様（列範囲を取るより、`down` を 3 回に分けて `lr_tmp_q/k/v` に出す方が単純）
  3. **新カーネル** `h3_gpu_qk_norm_rope_split_bf16(query, key, q_norm, k_norm, rope_cos, rope_sin, rows, HEADS, HEAD_DIM, ROPE_HALF)`（既存 `grouped_qkv_rope` と同じ数式を分離バッファに適用し、int8 経路と同じ形で SDPA に渡す）
  4. 既存の SDPA（query/key/value 版）→ out: `linear_bf16(lr_tmp, attention, out_down)` → `linear_bf16(attention_output, lr_tmp, out_up)`
  5. gate/norm2/AdaLN は既存 → fc1: down → up を `fc1` バッファ [seq, FFN×2] に → `swiglu_bf16` → fc2: down → up。HyperFlow では非融合 MLP 用の `fc1` バッファを常に確保。
  - 作業バッファ `lr_tmp`: seq × 最大ランク（BF16）。
  - 行列積は `h3_gpu_linear_bf16`（ランク ≥ 256 なので MPSGraph 以上）。フェーズ 2 の計測後に tensor-op 汎用経路を低ランク形状に限って有効化するか決める。
- dense ブロックは今と同じ（M5 では int8 量子化・融合 MLP、ストリーミングでは BF16）。
- 注意キャッシュは HyperFlow では作らない・使わない。LoRA は拒否。

### 3.6 SSD ストリーミング（h3_dit.c、フェーズ 3）
- `h3_dit_stream_layer` の sources を可変長（最大 12）にし、各 source に field 種別とランクを持たせる。
- スロットは dense 4 枚に加え、低ランク用に**モデル内最大ランクで確保**した down/up テンソルを持つ。読み込みは要素数「一致」から「先頭 n 要素」へ緩める（`h3_gpu_tensor_stream_file_*` に長さ引数。dense 経路は従来どおり一致を要求）。staging は各 field のモデル内最大バイト数で確保（Q4_K/Q6_K が混在するため、最初のブロック基準では足りない）。
- `expand_streamed` は既存の `h3_gpu_dequantize`。前取り・スロット交互は既存の流れ。

### 3.7 アプリ（Swift、フェーズ 4）
- 取り込み: `HFRepoProbe` の既知リポジトリに `VideoRebirthAI/hyperflow-h3-flash`（`fl2va_gguf/`・`ref2va_gguf/`）。`h3_gguf_inspect` の `hyperflow` で判定し、`ModelImportManifest` に `family: "hyperflow"`、`steps: [4, 6, 8]`、`defaultSteps: 6` を記録。Ref2VA 判定は今の `ref2v` 名規則で足りる。基本モデル（`ImportBase`）からの tokenizer/VAE/config の複製は今と同じ。
- 生成 UI: モデルが HyperFlow のとき、ステップは 4/6/8 の 3 択（既定 6）、速度プリセット・LoRA・再ノイズ・denoise reuse・層数は無効化して理由を表示。API は `steps` だけ。
- 記録: `ProjectDraft`/`ResolvedResult`/XMP に `modelFamily`（"hyperflow"）を任意項目で追加。無指定は従来どおり。進捗の較正キーにも含める。
- 表示: Model Manager の詳細に「HyperFlow H3 Flash（13.7B、4/6/8 ステップ）」と出典・ライセンス。`ModelLicenseNotice` に 1 行。
- メモリ見積り: `dit_resident_bytes` をブロック構成（dense/低ランク・ランク）から計算（フェーズ 2 の実測で係数を置く）。
- API: `GET /api/models/detail` に family と許可ステップ、`POST /api/generate` の `steps` 検証に family を反映。
- 文書: GGUF.ja/.md（新しい節「HyperFlow H3 Flash」）、README、TECHNICAL、API、THIRD_PARTY_NOTICES（モデルの出典）、サイト gguf.html/guide.html（日英、アンカー同一）。

## 4. フェーズと完了条件（ブランチは main から 1 本ずつ、ff-merge、削除）

| # | ブランチ | 作業 | 完了条件 |
|---|---|---|---|
| 0 | （調査のみ） | ディスクの確保（不要ファイルを Google Drive などへ移す。移動・削除はユーザーの選択後）。GGUF の取得（fl2va、必要なら ref2va）。**参照ダンプ用キット**を作る: CUDA 機で ComfyUI + ComfyUI-GGUF + HyperFlow ノードを「ツールとして」動かし、同じ GGUF・同じ入力で、小さな構成（256×256 級・5 フレーム・6 ステップ）のステップ 0 の各ブロック出力と最終速度、全ステップの潜在、temb（SiLU 前後）を F32 で保存するスクリプトと README。ユーザーが CUDA 機で実行し NAS で戻す。マニフェスト（コミット、GGUF SHA-256、入力、許容差） | SiLU の有無が確定し、fixture が揃う |
| 1 | `hyperflow-schedule` | 3.2 の検出、3.3 の格子、3.4 の 2 時刻 AdaLN、パラメータ検証、`h3_gguf_inspect` 拡張。`tests/test_hyperflow.c`（重みなし: 格子、(t, r) 行割り当てと重複排除、混合式、通常経路の行配置が不変）。fixture があれば temb と AdaLN 行の比較 | `make test` が通り、`test_real_dit_schedule` の行配置不変、fixture の temb/AdaLN 行が BF16 丸め以内で一致 |
| 2 | `hyperflow-blocks` | 3.5 のローダーと低ランク実行、新カーネル、47/49 ブロック、`h3_generate_cli` で生成。fixture とのブロック出力・最終速度の比較（既存の MLX 整合境界 rel 2.5% から）。M5 で 512×512・3 秒と 672×384・5 秒の時間・ピークメモリ（int8 常駐） | fixture 一致、通常 H3 の生成出力が変更前後で bit 一致、HyperFlow の 6 ステップ生成が M5 で完走 |
| 3 | `hyperflow-streaming` | 3.6。M5 上の非 TensorOps 強制（2026-11-02 の方法）で M1〜M4 相当の動作とメモリを測る | ストリーミングで常駐と同じ出力（bit 一致か BF16 丸め以内）、1 ステップの読み量が約半分 |
| 4 | `hyperflow-app` | 3.7（取り込み・UI・記録・API・文書・日英）。`check_localizations.sh` | ウィザードから fl2va GGUF を取り込み、6 ステップで生成、記録に family が残る。ref2va も同じ流れ |
| 5 | （測定のみ） | 同じプロンプト・seed で HyperFlow 4/6/8 対 Turbo 4・DMAD・標準 20: 時間、ピークメモリ、品質（既存の eval kit）。M5 と非 TensorOps 強制の両方 | Turbo に品質で並ぶか勝つ構成が 1 つ以上あれば「試験的」として採用、なければ結果を付録に残して止める |

## 5. 検証
- `make test`（新テストを含む）。重みと fixture があるときの比較は `H3_HYPERFLOW_STRICT=1` で skip を失敗にする。
- 回帰: 通常 H3 の生成（672×384・3 秒・固定 seed）と `test_real_dit_schedule` が変更前後で bit 一致。
- 実機: フェーズ 2・3・5 の計測表（時間・ピーク footprint・品質）を付録に残す。蓋を閉じた時間は pmset で除外。
- アプリ: 取り込み → 生成 → 記録の再読込 → API の `steps` 拒否。

## 6. リスク・未確認
- ~~AdaLN 投影の前の SiLU の有無~~ → 2026-11-09 の参照ダンプで確定（付録 A.2）。
- M5 での速度は Turbo と同等以下の見込み。利点はストリーミング機と品質で、測るまで約束しない。
- ディスク空きが約 9 GiB。GGUF と取り込みコピーで約 18 GB 必要。
- 参照ダンプは GPL の ComfyUI ノードを使うが、コードは読まず・写さず、出力だけ使う。
- 16 GB Mac での可否は、ストリーミング時の活性ピーク次第（フェーズ 3 で測る）。

## 7. 対象外
INT8 convrot（safetensors）版、7B、LoRA の低ランク層への適用、HyperFlow 用の注意キャッシュ、既定モデルの変更、ComfyUI ノードのコード流用。

## 付録 A. 進捗

### A.1 2026-11-08 フェーズ 1 と フェーズ 2 の重みなしで書ける部分（ブランチ hyperflow-schedule）

GGUF 本体（9.05 GB）はディスクの空きを作るまで取得しない。その間に、実物ヘッダと小さなテンソルだけで確かめられる部分を実装した。

実装したもの:
- `h3_hyperflow.c/.h`: 検出（`h3_hyperflow_inspect`）。2 時刻の埋め込み、`hyperflow_gate`、`hyperflow_sigmas`、詰めた番号のブロック数、ブロックごとの dense/因子化とランクを、テンソルヘッダから読む。食い違う形（Q/K/V の一部だけ因子化、dense と因子化の両方がある、番号の穴）は拒否。
- `h3_host.c`: `h3_hyperflow_schedule_build`（4/6/8 の基準格子に映像 shift 12・音声 shift 3）。
- `h3_dit_schedule.c`: `h3_dit_schedule_precompute_two_time` と `h3_dit_two_time_rows`。行は (t, r) の組で重複を除く。E_t と E_r を F32 で計算し、混合 → SiLU → [96768, 16] の投影（曲線形と同じ F32 の投影関数を共用）。ブロック数ぶんだけ作り、残りは NULL。書き出し・読み戻し（AdaLN キャッシュ）は NULL のブロックを許す。
- `h3_dit.c`: 因子化ブロックの常駐読み込み（`load_hyperflow_block`）と実行。Q/K/V はそれぞれ down → up、Q と K は新カーネルで RMS norm + RoPE、V は value に直接。出力射影と MLP も down → up。FC1 の up は SwiGLU の gate/up の半分に分けて持ち、[seq, 2·FFN] のバッファを使わない。dense な部分は今までどおり int8 化。次のブロックが因子化 QKV のときは、前のブロックの int8 量子化の融合をしない。47/49 ブロックは `block_active` で表し、ゲート順位の層削減はしない。注意キャッシュは無視（警告）、層削減・コア再利用・トークン削減・LoRA は拒否、因子化ブロックの SSD ストリーミングは「未実装」と拒否。
- `h3.c`: GGUF の Transformer なら検出し、ステップ 4/6/8 以外・LoRA・再ノイズ・各種再利用・層数変更・独自 σ・音声 shift 変更を理由付きで拒否。HyperFlow の格子を使う。
- `h3_gguf.c`: `h3_gguf_info` に `hyperflow`・`dit_blocks`・`factored_blocks`（取り込み画面用）。
- `h3_shaders.metal` / `h3_gpu.m`: `h3_qk_rope_split_bf16_coop`（既存の協調カーネルを入力だけ分けて機械的に派生）と `h3_swiglu_split_bf16`。

確かめたこと:
- `tests/test_hyperflow.c`（`make test` に追加）: 格子が公式 README の `--sigmas` と 1e-6 で一致。(t, r) 行の配置（6 ステップ・条件あり 14 行、音声を固定したとき 7 行）。新カーネル 2 つが既存の融合カーネルと bit 一致。ランク 768/256 の down → up が CPU 参照と一致。
- 実物の GGUF の「疎なコピー」（ヘッダ全体と、時間埋め込み・AdaLN・gate・sigmas の 173 MB だけをバイト範囲で取得し、残りは穴）で、FL2VA と Ref2VA の両方について: 検出（47/49 ブロック、因子化 18/19 ブロック、最大ランク 3328、gate 0.25、16 次元）、`hyperflow_sigmas` が自前の 6 ステップ格子と 1e-6 で一致、2 時刻 AdaLN が倍精度の CPU 参照と rel-L2 1.65e-3（BF16 の丸めの範囲）で一致、AdaLN キャッシュの書き出し・読み戻し。`H3_HYPERFLOW_DIR` で実行する。
- 通常の H3 への影響なし: 取り込み済みの GGUF（曲線形 AdaLN）で 256×256・25 フレーム・4 ステップを main と比べ、注意キャッシュ・SSD ストリーミング・常駐 int8 の 3 経路とも出力フレームが bit 一致。

まだ確かめていないこと:
- **因子化ブロックを含む DiT の順伝播を実物の重みで動かしていない。** 部品（カーネル、行列積、読み込みの形の検査）は確かめたが、配線の正しさは GGUF 本体と参照ダンプ（フェーズ 0）が要る。
- AdaLN 投影の前に SiLU を掛ける（元の H3 と同じ）とした点は推定のまま。参照ダンプで確定する。
- 常駐時のメモリ（dense 30 ブロックの int8 と因子化部分の BF16 で約 15 GiB の見込み）は未測定。

### A.2 2026-11-09 参照ダンプ（フェーズ 0）の結果

RTX PRO 4000 Blackwell（24 GB）の ComfyUI で取った記録。
- 環境: ComfyUI v0.43.0-18-gb5a43215、ComfyUI-GGUF 6ea2651、ComfyUI-HyperFlow-Flash 20a22ab、torch 2.15.0+cu132。
- GGUF: FL2VA Q4_K_M。SHA-256 は公開値 `ee251661…` と一致。
- 記録: t2v を 2 回、i2v を 1 回、4 ステップの t2v を 1 回。すべて 256×256・39 フレーム・seed 42。
- 置き場所: NAS の `work/h3c-dump`（約 11 GB）。
- 道具: 自作の記録キット `h3c/hyperflow-ref-kit`。HyperFlow ノードのコードは読んでいない。

確定したこと:
- **AdaLN 投影の入力は SiLU(混合)。** 混合は E_t + 0.25·(E_r − E_t)。
  - 記録の `time_embedder` 出力は混合済みの値で、記録の E_t・E_r から作った混合と rel 4.6e-7 で一致した。
  - AdaLN 投影の入力は、その SiLU と rel 1.6e-3（BF16 の丸め）で一致した。
  - 疎な GGUF の重みから倍精度で計算した値も、記録と一致した: 時間埋め込み rel ≤ 5.9e-5、ブロック 0 の AdaLN 出力 rel ≤ 1.5e-3、最終層の AdaLN 出力 rel ≤ 1.3e-3（t2v・i2v、ステップ 0/3/5）。
  - フェーズ 1 の式（正弦の並び cos→sin、角度は t × 周波数、周波数の初項は 1）がそのまま正しい。
- **(t, r) の割り当ては実装どおり。**
  - 生成する映像の行は (1−σv_i, 1−σv_{i+1})、音声の行は shift 3 の同じ形。
  - i2v の先頭フレーム（条件行）は t = r = 0.999。
  - テキスト行は映像の行と同じ。
  - 4 ステップでも同じ形だった（ステップ 0 の r は映像 0.037424、音声 0.134586）。
  - ComfyUI はステップ 0 の音声の t を 1.0e-6 とする（映像 σ から音声 σ へ float32 で換算した誤差）。自前の実装は 0。影響は無視できる大きさ。
- **サンプラーは別々の Euler と等価。** ComfyUI は σ を映像の 1 本しか持たない。
  - 音声潜在は X = (σv/σa)·y（データ側を 4 倍、`audio_scale` 4.0）の空間で保持する。
  - 出力はモデル内で −3y − (1+3σa)·f に換算して返す（f は最終層の音声出力）。
  - この形で、全 6 ステップの出力と rel ≤ 6.8e-3 で一致した。
  - y に戻すと、どのステップも「音声 σ（shift 3）で別に進める Euler」と rel ≤ 3.4e-3 で一致した。
  - よって、自前の映像 Δσ・音声 Δσ を別々に使う Euler で正しい。
  - 映像の速度は最終層出力の符号反転で、Euler の符号は +1。
  - 時刻はモデルへ σ×1000 で渡される。
- **ブロック構成。**
  - 47 ブロックで番号の穴はない。因子化は 18 ブロック（1, 2, 4, 6–17, 19, 20, 46）。ランクは GGUF ヘッダの値と同じ（例: ブロック 1 の q 768 / k 768 / v 1024 / out 1792 / fc1 1792 / fc2 2560）。
  - dense ブロックの QKV は GGUF 上で融合済みの `qkv_proj`（[21504, 5376]）。
- **系列の並びと大きさ（t2v）。**
  - 映像 768 行（24×12×16×16 を 2×2 パッチ）、音声 130 行（[32, 2, 65]）、テキスト 24 行で、計 922 行。
  - i2v は条件フレーム 64 行（`video_patch_proj` を通る）とテキスト 96 行が加わり、計 1058 行。
  - テキストはブロックに入る時点で 5376 次元（精製済み）。精製は順伝播の外で行われるため、記録にない。
- **再現性。** 同じ設定の 2 回は全テンソルが完全一致（rel 0）。出力の mp4 もバイト一致した。比較の許容差は、GPU の非決定性を考えなくてよい。
- 出力の映像は 3 本とも破綻なし（凧・丘・草）。

フェーズ 2 で使う材料:
- ステップ 0 の全ブロック出力と、ブロック 0・1 の内部（norm・qkv/q,k,v・注意出力・MLP の各段）。
- 全ステップのモデル入出力（潜在、時刻）と、最終層の映像・音声出力。
- これを `blocks.0` の入力から自前のブロックに流して比べる。テキスト精製は別に確かめる（Qwen の隠れ状態は記録にないため）。


### A.3 2026-11-09 フェーズ 2: 実重みでの順伝播（ブランチ hyperflow-blocks）

- **フィクスチャ。** 参照キットの `make_fixture.py` で t2v の記録から 114 MiB を抜き出した（リポジトリ外 `h3c/hyperflow-fixtures/t2v_fixture.safetensors`）。中身は、ステップ 0・1 のモデル入力、精製済みテキスト、パッチ投影の入力、ブロック 0〜5・18・23・46 の出力、最終層の映像・音声出力。
- **テスト。** `tests/test_hyperflow_real.c`（`make test` に追加。`H3_HYPERFLOW_DIR` と `H3_HYPERFLOW_FIXTURE` がなければ skip）。
  - テキスト精製は ComfyUI では記録の外なので、記録の精製済みテキストを `h3_dit_debug_set_refined_text` で差し込む。精製器そのものの照合は残っている。
  - ブロック出力は `H3_DUMP_BLOCK_HIDDEN`（ステップ 0、常駐のみ）で書き出して比べる。
- **結果（int8 常駐、M5）。** BF16 常駐は約 27 GB で 24 GB 機に載らないため、量産と同じ int8 経路で比べた。
  - パッチの並び（映像・音声）は参照の投影入力と完全一致。映像 σ 格子の差は 1.5e-8。
  - ブロック出力の rel-L2: ブロック 0 は 0.98 %、1〜5 は 0.58〜0.67 %、18 は 0.85 %、23 は 1.9 %、46 は 8.9 %（cos 0.996）。因子化ブロック（1, 2, 4）と dense ブロック（0, 3, 5）で誤差の大きさは同じ。後段での増え方は int8 の誤差が積み重なった分と見ている（BF16 経路では確かめられていない）。
  - 最終層の出力: ステップ 0 は映像 4.7 %・音声 3.4 %、ステップ 1 は映像 4.0 %・音声 7.9 %（cos ≥ 0.997）。
  - エンジンの速度は最終層の出力そのもの。符号を反転し尺度を変えるのは ComfyUI のラッパー側（A.2）。
- **生成（`h3_generate_cli`、6 ステップ、seed 42、凧のプロンプト）。** 3 本とも破綻なし。音声は平均 −41 dB で、CUDA 出力の −47 dB と同程度。

| 大きさ | モデル | 1 ステップ目 | 2 ステップ目以降 | 全体 | ピーク footprint |
|---|---|---|---|---|---|
| 512×512・73 フレーム | HyperFlow（MPSGraph の線形） | 50.6 s | 14.4 s | 178.6 s | 16.0 GiB |
| 512×512・73 フレーム | HyperFlow（TensorOps の線形） | 14.5 s | 14.5 s | 140.4 s | 17.2 GiB |
| 672×384・121 フレーム | HyperFlow（TensorOps の線形） | 28.1 s | 28.0 s | 244.7 s | 17.0 GiB |
| 672×384・121 フレーム | 通常 H3 GGUF（注意キャッシュ、6 ステップ Euler） | 32.9 s | 43.7 s | 326.9 s | 15.7 GiB |

- **最初のステップの遅さ。** 因子化した線形はランクの組み合わせで約 200 通りの形がある。MPSGraph は形ごとに初回にグラフを作るため、最初のステップが 36 s 余計にかかった。M5 では因子化した線形を TensorOps カーネルに回すようにした（ランクは 256 の倍数で条件を満たす。`H3_HYPERFLOW_MPSGRAPH=1` で元に戻せる）。当初は DiT の線形すべてを回していたが、フェーズ 3 で因子化分だけに絞った（A.4。速さは同じ）。
- **M5 の速度。** 通常 H3 の既定（注意キャッシュ）に比べ、1 ステップあたり 64 %。計画の見込み（8〜9 割）より速い。ただし比べた相手は注意キャッシュ経路で、int8 常駐の通常 H3 ではない。
- **回帰。** 通常 H3 GGUF の生成（384×384・25 フレーム・4 ステップ・seed 7）は、main と映像・音声のデコード結果が完全一致（mp4 の差は 21 バイトの時刻情報だけ）。`make test` は全部通過。
- **フェーズ 4 に回すこと。**
  - 生成前の見積りが通常 H3 の定数のままで、23.1〜23.9 GiB と出て「スワップの恐れ」と警告する。実測は 16〜17 GiB。ブロック構成から計算する形に直す。
  - `h3_load_dir` は `FL2VA/transformer/config.json` の存在を要求する。HyperFlow の GGUF には付いていないので、取り込み時に書く（中身は読まれていない。今回は通常 H3 のものを置いた）。
  - 同じフォルダに `.gguf` が 2 つあると、重みストアが両方を読もうとする。取り込み先は `model.gguf` 1 つにする。

### A.4 2026-11-09 フェーズ 3: SSD ストリーミング（ブランチ hyperflow-streaming）

- **作り。**
  - 1 ブロックの読み込み元を可変長にした（最大 13 個: q/k/v・out・FC1・FC2 の因子、または dense FC1 の SwiGLU 半分）。FC1 の半分は、チェックポイントの同じテンソルの行範囲として読む（行は量子化ブロックの境界に揃う）。
  - 2 つのスロットを「BF16 の領域 1 つ + 量子化のままの読み込み領域 1 つ」にした。大きさは最大のブロック 1 つ分。各ブロックの行列は、その中のページ境界からの別名（`h3_gpu_tensor_alias_*`、`newBufferWithBytesNoCopy`）にした。種類ごとに別のバッファを持つ形では、dense と因子化の両方を同時に抱えて 1.2 GiB 多かった。
  - 同じ行列構成のブロックは別名を共有する。ブロックごとに別名を作ると、通常 H3 のストリーミングが 6 % 遅くなった（バッファ数 8 → 376）。共有で元の速さに戻った。
  - 読み込みは要素数の完全一致の代わりに「先頭 n 要素」を許す関数を追加（`h3_gpu_tensor_stream_file_prefix_*`）。
- **照合。** フィクスチャのテストに `H3_HYPERFLOW_SSD=1` を追加した。ストリーミングは BF16 の行列で計算するので、24 GB 機で BF16 経路を参照と比べられる唯一の形になる。
  - ブロック 0〜5 は 0.38〜0.56 %、23 は 1.0 %、46 は 5.2 %（cos 0.9986）。
  - 最終層の出力: ステップ 0 は映像 2.3 %・音声 1.9 %、ステップ 1 は映像 3.9 %・音声 5.4 %。
  - int8 常駐（ブロック 46 で 8.9 %）より参照に近い。BF16 経路でも後段で数 % まで増えるので、増え方の大半は BF16 の丸めの積み重ね。許容差は BF16 で 8 %、int8 で 15 % とした（公開版 H3 の照合テストは 25 %）。
- **回帰。** 通常 H3 GGUF のストリーミング（384×384・25 フレーム・4 ステップ）は、main と映像・音声のデコード結果が完全一致。ノイズ除去 36.44 s 対 36.45 s、ピーク 7.82 GB で同じ。

| 672×384・121 フレーム・6 ステップ | 1 ステップの読み量 | 2 ステップ目以降 | 全体 | ピーク footprint |
|---|---|---|---|---|
| HyperFlow、ストリーミング（M5） | 7.8 GiB | 39.0 s | 303 s | 8.5 GiB |
| HyperFlow、ストリーミング（M1〜M4 相当の模擬） | 7.8 GiB | 36.1 s | 415 s | 10.9 GiB |
| 通常 H3 GGUF、ストリーミング（M5） | 10.1 GiB | 51.9 s | 378 s | 8.9 GiB |
| HyperFlow、int8 常駐（M5） | — | 30.3 s | 256 s | 16.5 GiB |

- 読み量は通常 H3 GGUF（Q4_K_M、11.4 GB）の 77 %。計画の「約半分」は、17 GB 級の H3 を想定した見込みだった。
- M1〜M4 相当の模擬は 2026-11-02 と同じ方法（作業用の複製で TensorOps を切り、GPU ファミリーを 8 に抑え、シェーダーを MSL 3.2 で作る）。時間は M5 の GPU のもので、実機の M1〜M4 の時間ではない。全体が長いのは VAE デコードが TensorOps なしで遅いため。ピークの 10.9 GiB も VAE デコードの時点。映像は M5 の出力とほぼ同じで、破綻なし。
- **計測のぶれ。** NAS への移動（rsync）が裏で SSD を読んでいたため、1 回きりの計測は ±2 s/ステップほどぶれた。512×512・4 ステップで「因子化分だけ TensorOps」「全部 TensorOps」「模擬」を交互に 2 回ずつ測ると、ストリーミングは 19.0〜20.8 s、常駐は 13.7〜14.0 s で、差はぶれの範囲だった（常駐の 1 回だけ 20.3 s の外れ）。表の時間も同じ程度の幅で読むこと。
- **残り。** 16 GB 機でのストリーミングの可否は、ピーク（M5 で 8.5 GiB、模擬で VAE デコード時 10.9 GiB）から見て収まる見込み。実機では確かめていない。

### A.5 2026-11-09 フェーズ 4: アプリ（ブランチ hyperflow-app）

- **エンジン。** 生成前の見積りが、HyperFlow の常駐メモリをブロック構成から数える（int8 の dense と BF16 の低ランク）。672×384・5秒で 23.9 → 19.6 GiB（実測 16.5 GiB）、スワップの警告は出なくなった。注意キャッシュは前提にしない。
- **判定。** `H3ModelFamily.detect` が FL2VA の DiT の GGUF ヘッダから `hyperflow` を読む（`H3GGUFInfo` に `hyperflow`・`ditBlocks`・`factoredBlocks` を追加）。
- **生成画面。** HyperFlow のとき:
  - ステップは 4/6/8（既定 6）。通常モデルに戻すと元のステップ数に戻る。
  - 速度・再利用・層数・LoRA の欄を隠し、理由を表示する。
  - 計算方式「int8 キャッシュ」は、24 GB 以上の M5 で常駐（int8）、それ以外で SSD ストリーミングとして動く。
  - AdaLN 曲線と Open-H3 再生成は使わない。
  - 送る値は `effective*` で HyperFlow 用に置き換える。フォームの値は通常モデル用に残す。プロジェクトの下書き・プリセット・結果の読み込みでも、20 ステップや LoRA は入らない。
- **API。** `POST /api/generate` は、4/6/8 以外のステップ、LoRA、速度プリセット、reuse、層数、再ノイズ、3 以外の audio shift に理由付きの 400 を返す。既定は 6 ステップ・reuse 1。`GET /api/models/detail` は `family` と `steps` を返す。
- **記録。**
  - `ProjectDraft.modelFamily`、結果の要約（「モデル: HyperFlow H3 Flash」。層数・速度・再利用の行は出さない）。
  - XMP の `AISystemUsed`、進捗の較正キー `.hyperflow`（初期値は M5 の実測）。
- **取り込み。**
  - ウィザード・Hugging Face の下見に種別を表示する。FL2VA と Ref2VA の種別の混在は止める。
  - `h3c-import.json` に `family` を残す。
  - 既知の配布元に VideoRebirthAI/hyperflow-h3-flash を加え、FL2VA Q4_K_M を動作確認済みにした。ライセンスの 1 行も足した。
  - `transformer/config.json` は通常どおり基本モデルから複製する（中身は読まれない）。
- **確認（API 経由、M5）。** 次の順に確かめた。
  1. 取り込みの dry run と実際の取り込み（manifest に `family`）。
  2. 詳細表示。
  3. 拒否 5 種。
  4. 672×384・5 秒の既定設定の生成: 6 ステップ・常駐・309 s。映像は破綻なし、音声 −41 dB、記録と XMP も正しい。
  5. 通常モデルへの戻し（ステップの規則も 3〜40 に戻る）。
  - 詳細設定の画面そのものは目で確かめていない。
- **文書。** GGUF.ja/.md に「HyperFlow H3 Flash」の節、README、API、THIRD_PARTY_NOTICES を更新した。サイト（docs/）は公開時に更新する。
- **残り（フェーズ 5）。** Turbo 4・DMAD・標準 20 ステップとの画質の比較。Ref2VA の HyperFlow（49 ブロック）は取り込み・生成とも未確認。

### A.6 2026-11-09 フェーズ 5: 画質と時間の比較（測定のみ）

- **方法。**
  - アプリの API で 27 本を生成した（M5 24GB、672×384・5 秒、プロジェクト HyperFlow-eval-20261109）。
  - プロンプトは DMAD 評価と同じ 3 つ（猫と毛糸・広場のギター・廊下で話す生徒）。
  - seed 42 で 6 構成すべてを作った。T4・D4・HF6 は seed 314 でも作った。
  - 6 構成:
    - R20: 通常 H3 GGUF の 20 ステップ（int8 キャッシュ）。
    - T4: Turbo v1.2 LoRA で 4 ステップ。
    - D4: DMAD LoRA で 4 ステップ（LoRA ごとの自動サンプラー: 再ノイズ・音声 shift 2）。
    - HF4・HF6・HF8: HyperFlow FL2VA Q4_K_M、常駐 int8。
  - 採点は 8 フレームごと: MUSIQ・MANIQA（pyiqa）、CLIPScore（プロンプトとの一致）、ちらつき（隣り合うフレームの輝度差の平均）、音量。
  - 比較画像は、フレーム 0/40/80/最後を原寸で並べた。
  - キットは `h3c/hyperflow-eval/`（`run_all.sh`・`score.py`・`scores.md`・`sheet_*.png`）。
- **結果（seed 42 の 3 プロンプトの平均）。**

| 構成 | 生成時間 | MUSIQ | MANIQA | CLIPScore | ちらつき |
|---|---|---|---|---|---|
| R20 | 645 s | 64.6 | 0.382 | 0.914 | 0.82 |
| T4 | 182 s | 68.1 | 0.402 | 0.858 | 1.97 |
| D4 | 182 s | 64.3 | 0.372 | 0.831 | 1.40 |
| HF4 | 170 s | 65.9 | 0.369 | 0.866 | 0.96 |
| HF6 | 216 s | 67.5 | 0.365 | 0.877 | 1.13 |
| HF8 | 264 s | 68.7 | 0.388 | 0.853 | 1.22 |

  - seed 42 と 314 の 6 本の平均では、T4 が MUSIQ 68.7・MANIQA 0.397・CLIP 0.862・ちらつき 2.18、HF6 が 67.3・0.366・0.890・1.17、D4 が 64.6・0.377・0.852・1.34。
  - 1 本ごとの値は `scores.md` にある。プロンプトごとのばらつきは、構成どうしの差より大きい。
- **目で見た比較（原寸）。**
  - HyperFlow は 4・6・8 ステップとも破綻がない。通常側（T4・D4・R20）と同じく、構図・人物・楽器・毛糸は保たれる。
  - T4 はいちばんくっきりして、コントラストが強い。そのぶん、フレーム間のちらつきも最も大きい。
  - HF4・HF6 は、猫（seed 42）の毛並みが T4 や HF8 より柔らかく、ぼやけ気味。最後のフレームでは、HF6 の前足と毛糸がぶれて混ざる。HF8 は毛の縞が T4 並みに見える。
  - ギターと話す生徒では、HF は T4 と同等。D4 はギター（seed 314）で、遠くの顔がマイクと混ざってつぶれる。
  - 音は聞いていない。HyperFlow の音量は R20 に近い。T4 は話す生徒・ギターで 10〜15 dB 大きい。
- **時間（M5、常駐）。**
  - HF4 は T4 とほぼ同じ速さ（170 対 182 s。通常側の 15 本は NAS へのコピーと重なり、数秒ずつ遅くなっている可能性がある）。
  - HF6 は T4 の約 1.2 倍、HF8 は約 1.45 倍の時間がかかる。
  - HyperFlow のピーク footprint は 17〜18 GiB（常駐）。通常側は int8 キャッシュを mmap するので、footprint（5〜7 GiB）は比べられない。
  - 非 TensorOps（M1〜M4 相当）の時間とメモリは、フェーズ 3 の測定（SSD ストリーミングで 1 ステップ 36 s、ピーク 10.9 GiB）を使う。画質は計算経路によらず同じ重み・同じ式なので、測り直していない。
- **判定。** HF6・HF8 は Turbo（T4）と画質で並ぶ。
  - MUSIQ の差は 1 点以内で、MANIQA はやや低い。
  - CLIPScore（プロンプトとの一致）は上、ちらつきは約半分。目で見ても同等。
  - 計画の基準（Turbo に並ぶか勝つ構成が 1 つ以上）を満たすので、**「試験的」として採用**する。既定のモデルは変えない。
  - M5 では速さの利点はない（HF4 で T4 並み）。利点は、ちらつきの少なさとプロンプトとの一致、M1〜M4 の SSD ストリーミング（1 ステップの読み込み 7.8 対 10.1 GiB）。
- **残り。**
  - Ref2VA の HyperFlow（49 ブロック）の取り込みと生成。→ A.7 で確かめた。
  - アプリでの「試験的」の表示。
  - サイト（docs/）の更新は公開時に行う。

### A.7 2026-11-10 Ref2VA の HyperFlow（49 ブロック）の取り込みと生成（測定のみ）

- **方法。**
  - HyperFlow の Ref2VA Q4_K_M（`ref2va_gguf/`、9,220,352,512 バイト、SHA-256 は a6b30182…ec2d）を取得し、取り込み済みのモデル「HyperFlow H3 Flash Q4_K_M」に、アプリの API（`/api/models/add_ref2va`）で Ref2VA として加えた。
  - アプリの API で 8 本を生成した（M5 24GB、672×384・5 秒、seed 42、プロジェクト HyperFlow-ref2va-20261109）。
  - 参照は 3 種類で、プロンプトは種類ごとに 1 つ:
    - 画像: 顔の画像 1 枚。「参照画像の女性が大学の廊下でカメラに話す」。
    - 動画: 話す人物の動画 1 本。「参照動画の女性がカフェの席でコーヒーを持って話す」。
    - 画像＋音声: 顔の画像と 8 秒の声。「参照音声の声で話す」。
  - 構成:
    - R20: 通常 H3 GGUF の Ref2VA、20 ステップ（int8 キャッシュ）。3 種類とも。
    - HF4・HF6・HF8: HyperFlow Ref2VA、常駐 int8。画像は 3 つとも、動画と画像＋音声は HF6 だけ。
  - 採点はフェーズ 5 と同じ（8 フレームごとの MUSIQ・MANIQA・CLIPScore、ちらつき、音量）。
  - 声は、16 kHz モノラルにして、基本周波数の中央値（自己相関）と、32 帯域の長時間スペクトルの参照音声との RMS 距離を測った。
  - 比較画像は、フレーム 0/60/120 を原寸で並べた。画像＋音声は 10 フレームごとの帯も作った。
  - キットは `h3c/hyperflow-ref2va-eval/`（`run_all.sh`・`score.py`・`audio_stats.py`・`scores.md`・`frames/`）。
- **結果。**

| 参照 | 構成 | 生成時間 | ピーク | MUSIQ | MANIQA | CLIPScore | ちらつき | 音量（平均 / 最大） |
|---|---|---|---|---|---|---|---|---|
| 画像 | R20 | 638 s | 4.4 GiB | 74.8 | 0.490 | 0.793 | 1.59 | −22.1 / −1.5 dB |
| 画像 | HF4 | 184 s | 17.0 GiB | 70.4 | 0.457 | 0.711 | 1.42 | −16.7 / 0.0 dB |
| 画像 | HF6 | 225 s | 18.0 GiB | 73.4 | 0.451 | 0.703 | 1.30 | −17.2 / 0.0 dB |
| 画像 | HF8 | 280 s | 18.0 GiB | 75.7 | 0.516 | 0.680 | 1.43 | −17.0 / 0.0 dB |
| 画像＋音声 | R20 | 698 s | 5.8 GiB | 75.1 | 0.486 | 0.715 | 1.05 | −17.6 / −1.8 dB |
| 画像＋音声 | HF6 | 249 s | 17.0 GiB | 75.2 | 0.422 | 0.789 | 1.26 | −13.9 / 0.0 dB |
| 動画 | R20 | 2,285 s | 8.5 GiB | 71.6 | 0.393 | 0.745 | 1.38 | −30.3 / −14.1 dB |
| 動画 | HF6 | 1,180 s | 19.0 GiB | 70.4 | 0.382 | 0.818 | 1.60 | −23.1 / −2.5 dB |

  - R20 のピークは、int8 キャッシュを mmap するので比べられない（フェーズ 5 と同じ）。
  - 各 1 本（1 プロンプト・1 seed）なので、値の差は目安にとどまる。
- **取り込みと動作。**
  - 49 ブロック（うち 19 ブロックが低ランク分解）の Ref2VA は、コードの変更なしで取り込めて、画像・動画・画像＋音声のどの参照でもエラーなく生成できた。
  - 配布元の README は、GGUF の Ref2VA は画像の参照だけに対応すると書いている（配布元の実行環境についての記述）。このエンジンでは、動画と音声の参照も通常の Ref2VA と同じ経路で動いた。
- **目で見た比較（原寸）。**
  - 画像の参照: HF4・HF6・HF8 とも、参照の女性（髪・顔・ネックレス）が保たれ、廊下のプロンプトに従う。服は R20 を含む 4 本とも参照と違う。破綻はない。
  - 動画の参照: R20 と HF6 のどちらも、同じ女性がカフェの席でカップを持つ。
  - 画像＋音声: R20 と HF6 のどちらも、口が動いている（フレーム 0/60/120 はたまたま口が閉じていて、10 フレームごとの帯で確かめた）。
  - 人物の同一性は、数値では測っていない。
- **声。**
  - 参照音声の基本周波数の中央値は 172 Hz。
  - 参照音声なしの 4 本は 225〜239 Hz。参照音声ありは R20 が 174 Hz、HF6 が 182 Hz。
  - 参照音声とのスペクトル距離は、音声なしが 5.3〜6.3、音声ありが R20 2.57・HF6 2.65。
  - 参照音声は、通常のモデルと HyperFlow の両方で効いている。
  - 動画の参照では、参照動画の声との距離が R20 9.30・HF6 8.87 と大きく、どちらのモデルでも声が引き継がれたとは言えない。
  - HyperFlow の Ref2VA の音量は、R20 より約 5 dB 大きく、最大値がフルスケールに届く。331,648 サンプルのうち 0.999 以上は 2〜14 個（HF4 14、HF6 10、HF8 3、HF6 画像＋音声 2）。R20 の画像の参照は最大 0.837 で、届かない。
  - 音は聞いていない。
- **時間（M5、常駐）。**
  - 画像の参照: HF4 184 s、HF6 225 s、HF8 280 s。R20 の 638 s に対して 0.29〜0.44 倍。文章からの生成（A.6 の平均）より 9〜16 s 長い。
  - 動画の参照は、どちらのモデルでも重い（R20 2,285 s、HF6 1,180 s）。HF6 のピークは 19.0 GiB で、24GB の Mac では余裕が小さい。
  - Turbo LoRA の Ref2VA とは比べていない。
- **測定中の問題（アプリ側）。**
  - 最初の実行は 7 時間止まった。参照ファイルを `~/Documents` の下に置いて API で渡したので、macOS の「書類フォルダへのアクセス」の確認が出て、答えるまでファイルの読み込みが待ち続けた。
  - 入力を `~/models` に移して実行し直した。プロジェクトの下書きに古いパスが残っていると、次の起動でも同じ確認が出る。
  - API で渡すファイルは、`~/Documents`・デスクトップ・ダウンロードの外に置く。API 文書への注意書きは未対応。
- **判定。** HyperFlow の Ref2VA は、画像・動画・音声の参照で動く。「試験的」の範囲に Ref2VA を含める。既定のモデルは変えない。
- **確かめていないこと。**
  - 音を聞いての確認。
  - 別のプロンプト・seed。参照画像が複数のとき。
  - SSD ストリーミング（M1〜M4 相当）での Ref2VA。
  - HyperFlow での最初と最後の画像からの生成。
  - 続きの生成（参照あり）と HyperFlow の組み合わせ。
- **残り。**
  - アプリでの「試験的」の表示。
  - サイト（docs/）の公開は次のリリースで行う。
  - HyperFlow の Ref2VA の音量（フルスケールに届く）を、聞いて確かめる。
