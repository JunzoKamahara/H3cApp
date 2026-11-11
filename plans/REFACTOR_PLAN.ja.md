# リファクタリング計画 — src への移動・拡大の分離・切り替えられる設計

2026-11-11 作成（main e61cc28、v0.8.1 の直後）。この文書は計画だけで、コードはまだ変えていない。
[確認] はリポジトリを調べて確かめたこと、[推定] は見積もりや未確認の見込み。

## 1. 結論（先に）

- **src への移動はできる。** 5 つのフォルダ（core・io・model・gpu・upscale）に分け、公開ヘッダーは `include/` に置く。`#include "h3_x.h"` の書き方は `-I` で保つので、ソースの中身は変えずに `git mv` だけで済む。
- **その前に、ビルドの生成物をルートから出す。** いまルートには、ソース 78 個のほかに `.o`・`.d`・実行ファイルが 200 個ほど混ざっている。散らかりの大半はこちらで、ソースを動かすより安全に片づく。
- **拡大は「ビルド時に組み込むモジュール」として分離する。** 実行時に読み込むプラグイン（dylib）にはしない（§5.3）。エンジンは `libh3upscale.a` に分け、アプリは `Upscaler` プロトコルと登録表にする。
- **切り替えられる設計は、切り替えの軸ごとに「記述子の表」を置く形にする。** 軸は 6 つある（§6）。GPU カーネルの中まで抽象化はしない。
- **合格の条件は全フェーズ共通で「出力が 1 ビットも変わらない」。** 動作を変える変更は、この計画には入れない。

## 2. 現状 [確認]

- **ルート直下**：C／Objective-C のソース 42、ヘッダー 36、`h3_shaders.metal`、合わせて約 55,000 行。Markdown が 23、`Makefile`、ビルドの生成物（`.o`・`.d`・`libh3.a`・テストとツールの実行ファイル約 60）も同じ場所にある。
- **大きいファイル**：`h3_gpu.m` 8,243 行、`h3_shaders.metal` 7,075 行、`h3_dit.c` 5,313 行、`h3.c` 3,544 行。この 4 つで全体の 44%。
- **拡大のエンジン側**（`h3_vsr*`・`h3_seedvr2*`・`h3_latent_upscale`、12 ファイル約 9,400 行）：
  - `h3.c`・`h3.h`・`h3_dit.c`・`h3_host.c`・`h3_plan.c`・`h3_weights.c` からの参照は 0。公開ヘッダーも別（`h3_vsr.h`・`h3_seedvr2.h`・`h3_latent_upscale.h`）。C の層ではすでに分かれている。
  - 残っている結びつきは GPU の層：`h3_gpu.m` に VSR の関数 34・SeedVR2 の関数 27（言及は 264 行）、`h3_gpu.h` に 41 行、`h3_shaders.metal` に 108 行。
  - `libh3.a` に一緒に入っている（`Makefile` の `LIB_C`）。
- **拡大のアプリ側**：
  - `VideoUpscaler` は 9 つの case を持つ enum。case での分岐が `VideoUpscale.swift`（1,523 行）に 48 か所、`ModelManagerView.swift` に 9、`GenerationViewModel+API.swift` に 4。
  - モデルの準備は `SwiftVRModelSetup.swift`（714 行）・`SeedVR2ModelSetup.swift`（480 行）・`OpenH3ModelSetup.swift`（588 行）の 3 つで、形がほぼ同じ。
- **切り替えの現状**：
  - `getenv("H3_…")` の名前が 158 種類、ソースに散らばっている。
  - HyperFlow・GGUF・Ref2VA の分岐は `h3.c` に 67 行、`h3_dit.c` に 42 行。
  - 登録式の先例が 1 つある：ccv の注意計算（`h3_gpu_ccv_attention.mm`）は、起動時のコンストラクタで `h3_gpu_register_ccv_*` を呼び、`libh3.a` の外のオブジェクトとしてリンクされる。
- **パスに依存する場所**：
  - `native/H3Spike/Sources/CH3/include/` の 6 つのシンボリックリンク（`../../../../../h3.h` など）。
  - `Package.swift` の `-L<リポジトリの根> -lh3` と `h3_gpu_ccv_attention.o`。
  - `package_app.sh` の `cp "$repo_root/h3_shaders.metal"`。
  - シェーダーは、CLI とテストでは作業フォルダの `h3_shaders.metal` を名前で読む（`h3_gpu.m:401`）。
  - テストの `#include "../h3_gpu.h"` など 5 か所。
  - 文書の中のファイル名：`UPSCALE_PLAN.md` 56、`SPEEDUP_ROADMAP.md` 30、`HYPERFLOW_PLAN.ja.md` 21、`TECHNICAL.md` 18、`BUILD.md` 6。
- **上流との関係**：antirez/h3.c のフォークだが、remote は origin だけで、独自のコミットが 524 ある。上流を取り込む運用はしていないので、移動で失うものは小さい。

## 3. 目標と、やらないこと

- **目標**
  - ルートを、入口になるものだけにする（README・LICENSE・Makefile・主な文書）。
  - 「どのファイルがどの部分か」をフォルダで分かるようにする。
  - 拡大を外しても、生成のエンジンがビルドでき、テストが通る。
  - 新しい拡大方式・モデルの系統を足すとき、触る場所が 1 か所の表と 1 つのファイル群で済む。
- **やらないこと**
  - 計算の書き換え、カーネルの統合、速度の改善（別の作業）。
  - 公開 C API（`h3.h` など）の名前と引数の変更。
  - アプリの保存形式・API の値の変更（`VideoUpscaler` の rawValue、プロジェクトのファイル）。
  - 実行時に読み込むプラグイン。

## 4. フォルダの案

```
include/            公開ヘッダー（Swift が読む 6 つ）: h3.h h3_vsr.h h3_seedvr2.h
                    h3_latent_upscale.h h3_lora_extract.h h3_refmod.h
src/
  core/             h3.c h3_internal.h h3_host.* h3_memory.* h3_plan.* h3_json.h
  io/               h3_safetensors.* h3_gguf.c h3_weights.* h3_quant.*
                    h3_av_reader.* h3_av_writer.* h3_tokenizer.*
  model/            h3_dit.* h3_dit_schedule.* h3_adaln_curve.* h3_attention_cache.c
                    h3_hyperflow.* h3_lora.* h3_lora_extract.c h3_refmod.c
                    h3_text_encoder.* h3_vision_encoder.* h3_multimodal.*
                    h3_video_vae.* h3_video_encoder.* h3_audio_vae.*
  gpu/              h3_metal.* h3_gpu.* h3_gpu_ccv_attention.mm shaders/
  upscale/          h3_vsr*.c h3_vsr_internal.h h3_seedvr2*.c h3_seedvr2_internal.h
                    h3_latent_upscale.c
tools/              いまの tools/ に h3_build_*.c の 3 つを足す
tests/              そのまま
build/              .o .d libh3.a と実行ファイル（git には入れない）
plans/              *_PLAN*.md SPEEDUP_ROADMAP.md website-*.md note-handoff.md
native/H3Spike/     そのまま（名前の変更は §9 の保留）
docs/               そのまま（Pages の公開元）
```

- **5 つに分ける理由**：1 フォルダ 6〜20 ファイルに収まり、依存の向きが一方向になる（upscale → model → io・gpu → core の順に下を使う。`h3.c` だけは全体をまとめる役で、model を使う）。
- **これ以上は分けない**：`model/` を vae・encoder・dit にさらに割る案は、ファイル数が少なすぎるので採らない。
- **`#include` は変えない**：`-Iinclude -Isrc/core -Isrc/io …` を `CFLAGS` に足す。フォルダをまたぐ include の禁止は、フェーズ 4 のあとで必要なら入れる。
- **ルートに残す文書**：README・BUILD・GUIDE・API・GGUF・REFMOD・TECHNICAL・THIRD_PARTY_NOTICES（各 2 言語）。サイトとリリースノートからリンクされているので動かさない。

## 5. 拡大の分離

### 5.1 エンジン側

- **`libh3upscale.a` を分ける。** 中身は `src/upscale/` の全部。`libh3.a`（生成）は拡大なしでリンクできるようにする。
- **GPU の層を割る。** `h3_gpu.m` の VSR・SeedVR2 の部分（3 か所：3635 行〜、3858 行〜、7148 行〜と SeedVR2 の節）を `src/upscale/h3_gpu_vsr.m`・`h3_gpu_sv2.m` に移す。
  - 前提：`h3_gpu.m` の中だけで見える構造体と補助関数を、`src/gpu/h3_gpu_internal.h` に出す必要がある。どれだけ出すことになるかは未調査 [推定：数十個]。フェーズ 4 の最初に数える。
- **シェーダーを割る。** `src/gpu/shaders/` に `common.metal`・`dit.metal`・`vae.metal`・`vsr.metal`・`seedvr2.metal` などを置き、ビルドでつなげて 1 つの `h3_shaders.metal` を作る。
  - 実行時の読み込み（1 ファイルを名前で読む、バンドルの Resources だけを信用する）は変えない。
  - つなげた結果が、いまのファイルと同じカーネルの集合になることをテストで確かめる。
- **登録式にする範囲**：C の拡大は、もともと別の API（`h3_vsr_*`・`h3_seedvr2_*`）で呼ばれるので、エンジン側に登録表は要らない。要るのは GPU の起動時の準備（パイプラインの生成）だけで、ccv と同じ `h3_gpu_register_*` の形にする。

### 5.2 アプリ側

- **`Upscaler` プロトコル**を作り、9 つの方式をそれぞれ 1 つの型にする。持つもの：
  - id（いまの rawValue と同じ文字列）、表示名、説明、試験的かどうか。
  - 使える条件（OS・GPU・モデルの有無）と、使えないときの理由。
  - 選べる倍率と大きさの上限、見積もり時間。
  - 必要なモデル（配布元・ファイル・ハッシュ・ライセンス）と、準備の画面。
  - 実行（フレームの入力 → 出力、進捗、中止）。
- **登録表**（`UpscalerRegistry`）に並べ、選択画面・モデル管理・API・メニューはこの表を回す。case の分岐 60 か所あまりを、型の中に移す。
- **モデルの準備を共通化する。** 3 つの `*ModelSetup.swift` の同じ部分（ダウンロード・照合・変換・進捗の画面）を 1 つにし、方式ごとの違いは記述子に持たせる [推定：1,800 行 → 900 行前後]。
- **Swift のターゲット**：`H3Engine` から `H3VSR.swift`・`H3SeedVR2.swift`・`H3LatentUpscaler.swift` を `H3Upscale` ターゲットに分ける。アプリの拡大のコードは `Sources/H3cApp/Upscale/` にまとめる。
- **Real-ESRGAN と MetalFX・標準**は、エンジンを使わない方式として同じ表に載せる。

### 5.3 実行時プラグインにしない理由

- **署名と公証**：Hardened Runtime では、同じチームの署名がない dylib は読み込めない。外部の人が作ったプラグインは、制限を緩めない限り動かない。
- **安全**：このアプリは、作業フォルダのシェーダーを読まないようにするなど、外から来たコードを実行しない方向で作っている。プラグインはこれに逆行する。
- **内部の境界が安定していない**：拡大は GPU の内部関数とシェーダーを直接使う。これを ABI として固定すると、速度の改善のたびに足かせになる。
- **需要**：配布は 1 つの DMG で、方式を足すのは開発者だけ。ビルド時の切り替えで足りる。
- **将来の選択肢**：メモリの都合（拡大のあいだ生成のモデルを確実に手放す）で分けたくなったら、別プロセス（XPC）にする道がある。プロトコルで分けておけば、そのときの変更は実行の部分だけで済む。

## 6. 切り替えられる設計

切り替えの軸と、いまの持ち方、提案：

| 軸 | 例 | いま | 提案 |
|---|---|---|---|
| モデルの系統 | 通常 H3・HyperFlow（FL2VA／Ref2VA） | `h3.c`・`h3_dit.c` の分岐 109 行 | 記述子 `h3_model_family`（ブロック数、低ランクの配置、使えるステップ数、使えない機能、サンプラー）の表 |
| 重みの形式 | safetensors・GGUF・int8 ConvRot | `h3_weights.c` に集約済み | 形式ごとの読み手を関数の表にするだけ（小さい変更） |
| 計算方式 | int8 キャッシュ・常駐・SSD ストリーミング | `h3_plan.c` と `h3.c` | 方式の記述子（必要メモリ、TensorOps の要否）を `h3_plan` に集める |
| 注意計算 | MPSGraph・ccv | 登録式（済み） | そのまま |
| サンプラー | 通常・再ノイズ（Turbo／DMAD）・HyperFlow | `h3_dit_schedule.c` と LoRA ごとの自動設定 | サンプラーの記述子の表 |
| 拡大 | 9 方式 | アプリの enum | §5.2 |

- **アプリは能力をエンジンに聞く。** いまは「HyperFlow では LoRA が使えない」などの知識が、エンジンとアプリの両方にある。記述子を `h3.h` から読めるようにし、アプリの画面はそれに従う。
- **環境変数は 1 つの表にする。** `src/core/h3_options.c` に名前・既定値・説明を並べ、ソースは `h3_option_int("…")` で読む。一覧を出すツールと、文書にない名前を見つけるテストを足す。158 個のうち、実験の名残りで使っていないものはこのとき消す。
- **抽象化しない場所**：ノイズ除去の 1 ステップの中、カーネルの呼び出し、VAE のタイル。ここは関数ポインタを入れると読みにくくなるだけで、切り替えの需要もない。
- **大きいファイルの分割**（切り替えとは別、ファイルを割るだけ）：
  - `h3_gpu.m` → 起動とパイプライン／テンソル／線形・注意／畳み込み／量子化／（拡大は §5.1）。
  - `h3_dit.c` → 重みの読み込み／ブロックの実行／ストリーミング／参照の条件付け。
  - `h3.c` → 読み込みと破棄／生成の流れ／つなげて生成／ツール用の API。

## 7. フェーズ

各フェーズは 1 つのブランチ、main へのマージは 1 フェーズずつ。機能の追加とは混ぜない。

| # | 内容 | 変わるもの | 危険 | 規模 [推定] |
|---|---|---|---|---|
| 0 | 基準を取る：`make test`、HyperFlow の実重みの照合、通常 H3・HyperFlow・拡大 3 方式の出力のハッシュ、アプリのビルドと起動 | なし（記録のキットだけ） | なし | 半日 |
| 1 | 生成物を `build/` に出す | `Makefile`・`.gitignore`・`Package.swift`（`-L`）・`BUILD.md` | 小 | 半日 |
| 2 | ソースを `src/`・`include/` に移す（`git mv` だけのコミット）、続けてパスの修正 | `Makefile`・シンボリックリンク 6 つ・`package_app.sh`・テストの include 5 か所・シェーダーの置き場所 | 中（機械的だが範囲が広い） | 1 日 |
| 3 | 計画書を `plans/` に移し、文書の中のファイル名を直す | 文書だけ | 小 | 半日 |
| 4 | 拡大のエンジン側の分離（§5.1）：GPU の層とシェーダーを割り、`libh3upscale.a` を作る | `h3_gpu.m`・`h3_shaders.metal`・`Makefile`・`Package.swift` | 中〜大（内部ヘッダーの切り出し） | 2〜3 日 |
| 5 | 拡大のアプリ側（§5.2）：プロトコル・登録表・準備の共通化 | Swift だけ | 中（画面と API の回帰） | 3〜4 日 |
| 6 | 記述子の表（§6）：モデルの系統 → サンプラー → 計算方式の順 | `h3.c`・`h3_dit.c`・`h3_plan.c`・アプリの条件分岐 | 中 | 3〜4 日 |
| 7 | 環境変数の表 | エンジン全体（機械的） | 小 | 1 日 |
| 8 | 大きいファイルの分割 | `h3_gpu.m`・`h3_dit.c`・`h3.c` | 小〜中 | 2 日 |

- **順番の理由**：1〜3 は中身を変えないので先に済ませる。4 を 5 より先にするのは、Swift のターゲットの分け方がライブラリの分け方に従うから。6 以降は独立で、必要になったものから進めればよい。
- **フェーズ 2 の決まり**：移動だけのコミットと、パスを直すコミットを分ける。こうすると `git log --follow` と `git blame` が移動をまたいで追える。
- **止めどころ**：1〜3 だけで「ルートが片づく」という目的は達する。4・5 で「拡大の分離」、6・7 で「切り替えられる設計」。どこで止めても、途中の形が残らないようにフェーズを切ってある。

### 進み具合

- **フェーズ 0（2026-11-11）**：基準は `h3c/refactor-baseline/`。`check_engine.sh`（小さな生成 5 本、潜在の拡大、SwiftVR、SeedVR2 の 1 枚と動画）、`check_app.sh`（アプリの拡大 8 方式と生成 1 本、`/api/upscalers` の応答）、`compare.py`。
  - 分かったこと：同じコードでも、H.264 にしたあとの映像は 3 回に 1 回ほど数レベル違う（連続した一部のフレームだけ。エンコーダーの揺れ）。照合は、潜在ファイル・エンコード前のフレーム・生の出力で行い、mp4 は揺れの範囲かどうかを見る。
  - アプリは `~/Documents` の下のファイルを読めない（macOS の許可の画面で止まる）。検査の入出力は `~/models/refactor-check/` に置く。
- **フェーズ 1（9eb052d）**：生成物は `build/`（`build/libh3.a`・`build/obj/`・`build/bin/`）。`make h3_tests` などの名前はそのまま使える。
- **フェーズ 2（7662bf7 が移動だけ、1411e82 がパスの修正）**：`make test`、エンジンの照合、アプリの照合（拡大 8 方式・生成 1 本）が基準と一致。シェーダーは `build/bin/h3_shaders.metal` に複製を置く（ツールは作業フォルダになければ実行ファイルの隣を読む）。
- **フェーズ 3**：計画書とサイトの保守メモを `plans/` に移した。
- **フェーズ 4（197f491）**：`libh3.a`（生成）と `libh3upscale.a`（拡大）に分けた。
  - `h3_gpu.m` から拡大の GPU コード約 1,450 行を `src/upscale/h3_gpu_vsr.m`・`h3_gpu_sv2.m` に移した。共有するのは `src/gpu/h3_gpu_internal.h`（GPU とテンソルのオブジェクト、補助関数 11 個）だけ。
  - カーネル名は `h3_gpu_register_kernels` で読み込み時に登録する。拡大をリンクしないプログラム（`h3_generate_cli` など）は、拡大のパイプラインを作らない。
  - シェーダーは `h3_shaders_core.metal`・`h3_shaders_vsr.metal`・`h3_shaders_seedvr2.metal` の 3 つ。つなげた `build/h3_shaders.metal` は、割る前のファイルと 1 バイトも違わない（MD5 一致）。本体をさらに dit・vae に割るのはフェーズ 8。
  - `libh3.a` に拡大のシンボルは 0（`nm` で確認）。`make test`、エンジンとアプリの照合が基準と一致。
  - 分かったこと：`tests/bench_dit.c`・`test_real_dit.c`・`test_semantic_dit.c` は main の時点で引数の数が合わず、コンパイルできない（`make test` には入っていない）。この計画では直していない。
- **フェーズ 5**：アプリの拡大を `Sources/H3cApp/Upscale/` にまとめた。
  - 方式は `UpscalerMethod`（`Upscale/Methods/` に 6 ファイル、9 方式）、ダウンロードするモデルは `UpscalerModel`（SwiftVR・SeedVR2・Open-H3 の `…Kit`）。`UpscalerRegistry` に並べる。
  - `VideoUpscaler` の enum は、API と記録に使う名前として残した。方式ごとの `switch` は 0 か所になり、選択画面・モデル管理・`GET /api/upscalers` は登録表を回す。新しい方式は「enum の case・型・登録表の 1 行」の 3 か所。
  - Swift のラッパー（`H3VSR.swift`・`H3SeedVR2.swift`・`H3LatentUpscaler.swift`）は `H3Upscale` ターゲットに分けた。
  - **見送ったこと**：3 つの `*ModelSetup.swift` のダウンロードと照合の共通化。20GB 級のダウンロードをやり直さないと確かめられないため、手を付けていない。
  - **確かめていないこと**：選択画面とモデル管理の画面を目で見る確認（「モデルを準備…」のシートの開閉、準備中の表示の更新）。コンパイルと API の応答、拡大 8 方式と生成の出力は基準と一致。
- **フェーズ 6**：記述子の表を `src/model/h3_model_family.c` に置いた（公開側は `h3.h`）。
  - **モデルの系統**（`h3_model_family`）：ステップ数の選択肢と既定値、できること（LoRA・再ノイズ・ステップの再利用・層の間引き・独自の sigma・音声シフト・注意キャッシュ・曲線形 AdaLN）、スケジュールの作り方。`h3_generate` の「HyperFlow では使えない」の判定は `h3_model_family_problem` になった（文言と順番は同じ）。`h3_dit.c` も、何が使えるかを問う場所は表を読む。
  - **サンプラー**（`h3_sampler`）と**計算方式**（`h3_compute_mode`）も表にした。
  - アプリは HyperFlow のステップ数の複製（`hyperflowSteps` など）を捨て、`H3ModelFamily.stepChoices`・`defaultSteps`・`can(_:)`、`ComputeMode.needsTensorOps` でエンジンに聞く。
  - **名前で分岐したままの場所**：HyperFlow そのものの説明文、メモリの目安、API の断りの文言（`isHyperFlow` が 7 か所）。因子化した線形層や 2 時刻 AdaLN など、ブロックの形の違いは `h3_dit.c` の `dit->hyperflow` のまま（これは構造で、方針ではない）。
  - **重みの形式**の表（§6 の 2 行目）は作っていない。`h3_weights.c` にすでに集まっていて、表にしても読む側が変わらないため。
- **フェーズ 7**：`src/core/h3_options.c` に、エンジンが読む環境変数 160 個すべてを、分野・取る値・説明つきで並べた。
  - ソースは `h3_option("H3_…")` で読む（中身は `getenv` だけ）。`tools/check_options.py`（`make test` に入っている）が、表にない名前・読まれない行・`getenv` の直呼びを落とす。`build/bin/h3_options` で一覧。
  - **消した変数はない。** どれが不要かは使う人の判断なので、手を付けていない。
- **フェーズ 8**：大きいファイルを割った（中身は移動だけ）。
  - `h3_gpu.m` → `h3_gpu.m`（デバイス・テンソル・コマンド）、`h3_gpu_blocks.m`、`h3_gpu_linear.m`、`h3_gpu_attention.m`、`h3_gpu_conv.m`。共有は `h3_gpu_private.h`。
  - `h3_dit.c` → `h3_dit.c`（公開 API とノイズ除去の流れ）、`h3_dit_setup.c`、`h3_dit_weights.c`、`h3_dit_block.c`。共有は `h3_dit_private.h`。ファイルをまたぐようになった補助関数 40 個は `h3_dit_` を前に付けた。
  - `h3.c` → `h3.c`（`h3_generate`）、`h3_params.c`、`h3_references.c`、`h3_conditioning.c`、`h3_plan_generation.c`。共有は `h3_private.h`。
  - シェーダーの本体 → `h3_shaders_core.metal`・`h3_shaders_tensorops.metal`・`h3_shaders_blocks.metal`。つなげた結果は割る前と同じ（MD5 一致）。
  - 最大のファイルは約 2,000 行（割る前は 8,243 行）。
  - **残っていること**：`h3_generate` は 1 つの関数で約 1,350 行ある。関数の中を割るのは移動では済まないので、この計画では触っていない。

## 8. 合格の条件

- **全フェーズ**：`make test` と翻訳キーの検査が通る。フェーズ 0 のハッシュ（生成の mp4 のデコード結果、拡大の出力フレーム）が一致する。アプリが署名付きでビルドでき、起動して 1 本生成できる。
- **フェーズ 2**：ソースの中身の差分が 0（`git diff -M --stat` で rename 100% が並ぶ。例外はテストの include 5 か所）。
- **フェーズ 4**：`libh3.a` だけで `h3_generate_cli` がリンクでき、拡大のシンボルを含まない（`nm` で確かめる）。つなげたシェーダーのカーネル名の集合が、割る前と同じ。
- **フェーズ 5**：`VideoUpscaler` を switch する場所が 0。保存済みのプロジェクトと API の `upscaler` の値がそのまま通る。
- **フェーズ 6**：`h3.c`・`h3_dit.c` に、系統の名前での分岐が残らない（記述子を読む形になる）。

## 9. 決めてほしいこと・保留

**2026-11-11 の決定**：フォルダは §4 の 5 つに分ける。計画書は `plans/` に移す（古いリンクが切れるのは認める）。まずフェーズ 5 まで進める。上流（antirez/h3.c）は更新がなく、取り込む予定はない。`native/H3Spike` の名前は変えない。

以下は決定の前の問い（記録として残す）。

1. **フォルダの分け方**：§4 の 5 つでよいか。もっと平ら（`src/` と `src/upscale/` だけ）にする案もある。平らな案は楽だが、55,000 行が 1 フォルダに残る。
2. **計画書の移動**（フェーズ 3）：`plans/` に移すと、過去のリリースノートやコミットの中のリンク（例：`UPSCALE_PLAN.md`）は古い場所を指したままになる。`blob/main/…` のリンクは切れる。切れてよいか、ルートに残すか。
3. **`native/H3Spike` の名前**：試作のころの名前で、中身は H3cApp。`app/` に変えると分かりやすいが、ビルドのキャッシュと手順書、リリースの手順が全部変わる。この計画では触らない案にしている。
4. **どこまでやるか**：フェーズ 1〜3 だけ／5 まで／全部。
5. **時期**：次の機能の作業と重ねない。進行中のブランチがあると、移動のあとのマージが面倒になる。

## 10. 危険と、未確認のこと

- **`h3_gpu.m` の内部の切り出し量**が分からない。多すぎる場合は、拡大の GPU コードを別ファイルにするだけにとどめ、`libh3upscale.a` への分離は見送る（フェーズ 4 の中で判断）。
- **シェーダーの分割**：共通の関数と定数をどのカーネルが使っているかは調べていない。割ったあとの順番を誤るとコンパイルできない（すぐ分かる種類の失敗）。
- **TensorOps を切ったビルド**（M1〜M4 相当）でも、同じ条件で確かめる必要がある。
- **`.o` の置き場所を変えると ccv のリンク**（`Package.swift` が `h3_gpu_ccv_attention.o` を直接指す）が変わる。リリースのビルドは ccv 付きなので、フェーズ 1 で必ず確かめる。
- **上流（antirez/h3.c）を今後取り込む予定があるなら**、移動のあとは手作業が増える。予定がある場合は、フェーズ 2 の前に言ってほしい。
- 規模の見積もりはすべて [推定]。
