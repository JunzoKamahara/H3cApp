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
- AdaLN 投影の前の SiLU の有無（フェーズ 0 で確定）。
- M5 での速度は Turbo と同等以下の見込み。利点はストリーミング機と品質で、測るまで約束しない。
- ディスク空きが約 9 GiB。GGUF と取り込みコピーで約 18 GB 必要。
- 参照ダンプは GPL の ComfyUI ノードを使うが、コードは読まず・写さず、出力だけ使う。
- 16 GB Mac での可否は、ストリーミング時の活性ピーク次第（フェーズ 3 で測る）。

## 7. 対象外
INT8 convrot（safetensors）版、7B、LoRA の低ランク層への適用、HyperFlow 用の注意キャッシュ、既定モデルの変更、ComfyUI ノードのコード流用。
