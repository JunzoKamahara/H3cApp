# h3c-app

[English](README.md) | [日本語](README.ja.md)

[MiniMax-H3](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
（テキスト／画像／動画から、音声付き動画を生成する拡散トランスフォーマー）を
Apple Siliconだけで動かすネイティブmacOSアプリ、H3cAppです。すべて
Metal/MPSGraph上でプロセス内実行され、Python・PyTorch・クラウド通信は一切不要です — メディアの
入出力はネイティブのAVFoundation/ImageIO経由です。基盤となるC/Objective-Cの
推論エンジン（`libh3.a`）はSalvatore
Sanfilippo氏（antirez）の[h3.c](https://github.com/antirez/h3.c)を
ベースにしています。詳しいライセンス表記は[ライセンス](#ライセンス)を
参照してください。

アプリを起動している間は、ウィンドウ操作に加えてローカルのJSON APIも
`127.0.0.1`上で提供されるので、スクリプトから操作することもできます。
詳しくは[API](#api)を参照してください。

## ダウンロード

はじめての方は、日本語の案内サイト（https://junzokamahara.github.io/H3cApp/ ）で、動作条件の確認から最初の動画づくりまでを順に確認できます。

もっとも簡単なインストール方法は、[リリースページ](https://github.com/JunzoKamahara/H3cApp/releases/latest)
にある署名・公証済みのビルド済み`.dmg`を使うことです — Xcode Command Line
Toolsもソースからのビルドも不要です。`.dmg`をダウンロードして開き、
`H3cApp`を`Applications`にドラッグしてください。初回起動時もGatekeeperに
よる「開発元が未確認」といった警告は出ません。

ソースからビルドする場合は、下記の
[ソースからのビルド](#ソースからのビルド)を参照してください。

## 動作要件

- Apple Siliconを搭載したMac、macOS 15（Sequoia）以降。エンジンがMPSGraphの
  Attention関数（macOS 15）とBF16（macOS 14）を使うため。0.4.3までは
  macOS 13以降と表記していたが、エンジンはすでにこれらのAPIを必要としていた。M5クラスの
  GPU（Metal 4 TensorOps）で最速のint8経路が有効になる。それより古いApple
  Siliconでも動作するが、BF16/MPSGraphに自動的にフォールバックする。
- Xcode Command Line Tools（`clang`、`swift`、`ar`） —
  ソースからビルドする場合のみ必要。ビルド済み`.dmg`では不要。
- Hugging Faceの
  [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  チェックポイント — `FL2VA/`だけで約134GiB（約37GiBのBF16トランスフォーマー
  本体に加え、Qwen3-VLテキストエンコーダーと動画/音声VAEを含む）。参照
  条件付き生成用の`Ref2VA/`も加えると合計で約268GiBになります。アプリの
  モデルマネージャーから、どちらも直接ダウンロードできます。
- 実行時に他の依存は不要。

## クイックスタート

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/H3cApp.app
```

初回起動時、まだモデルが登録されていなければ、アプリがHugging Faceから
MiniMax-H3を直接ダウンロードするか尋ねます（[機能](#機能)を参照）。すでに
チェックポイントを持っている場合は「モデル管理」からその場所を指定して
ください。

対応GPUでは、計算方式選択からint8アテンションキャッシュを作成（高速経路）
するか、「省メモリ（SSDストリーミング）」を選んでこの作成手順自体を省略し、
元のBF16重みをストリーミングすることもできます。

## 機能

- **テキストから動画+音声**（T2VA）: プロンプトだけで同期したH.264+AAC
  クリップを生成。
- **最初・最後フレーム条件付け**（FL2VA）: 生成の開始・終了フレームを固定
  （アプリの「最初・最後の画像」モード、またはAPIの
  `first_frame_path`/`last_frame_path`）。
- **参照条件付け**（Ref2VA）、順序付きで組み合わせ可能: 画像・動画
  （音声あり／なし）に加えて音声単体のファイルもスタイル・内容の参照
  として使用できます（アプリの「参照画像・動画・音声」モード、または
  APIの`reference_paths`）。ただし音声だけを参照にすることはできず、
  画像か動画の参照が少なくとも1つ必要です。参照は順番に`<Picture N>`/
  `<Video N>`としてモデルに提示されます。参照の読み込みと生成動画の
  書き出しはすべてネイティブのAVFoundation/ImageIO経由です。
- **メモリ連動の参照動画サイズ調整**: 大きな参照動画は、マシンの物理
  メモリ搭載量に応じて（[h3_host.c](h3_host.c)内の
  `h3_reference_max_pixels()`）モデルに渡す前に自動的に縮小されます。
  マシンの余力に関わらず常に同じ固定上限を狙うのではありません。なぜ
  これが重要かは下の
  [参照条件付けのコスト](#参照条件付けのコストなぜ参照動画を大きくすると遅くなるのか)
  を参照してください。
- **LoRAの重ねがけ**: モデルマネージャーがLoRAファイルのライブラリを
  保持し、そのうち最大8個を同時にオンにできます（それぞれ個別の強さ付きで、
  効果は足し合わされます）。公開されているH3用LoRAの形式はそのまま読み込めます
  — diffusers/PEFT（`to_q`/`to_k`/`to_v`）、ComfyUI
  （`diffusion_model.blocks.N...`）、kohya（`lora_unet_...`）、ネイティブ名、
  BF16/F16/F32、任意のrank、全ブロック・一部ブロックのどちらも対応し、
  マネージャーには各ファイルの形式・対象ブロック数・rankが表示されます。
  融合済みキャッシュファイルは作らず、エンジンが各重みをロード／ストリーム
  するたびにGPU上で差分を加えるため、3つの計算方式すべてで使えます
  （[LoRA](#lora)参照）。4ステップのTurbo蒸留LoRA
  （[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
  由来）はエンドツーエンドで動作します。LoRAごとに推奨ステップ数を
  持たせられ（`..._4step_...`のようなファイル名から自動判定）、そのTurbo
  LoRAをオンにすると生成ステップ数が自動で切り替わります。
- **3つの計算方式**（[計算方式](#計算方式)参照）: 高速なint8アテンション
  キャッシュ、キャッシュ不要で大容量メモリのMac向けの常駐モード、
  低メモリだが低速なSSDストリーミング。
- **速度モード**: 標準／高速／最速。エンジン側の近似手法（ステップ間での
  Transformerコアの再利用`core_reuse`（ステップ数に応じて調整）、トークン
  削減）をまとめて切り替え、20ステップでDiT部分が約2.7倍／約2.9倍速く
  なります。詳細は[速度モード](#速度モード)。
- **高速モード（試験的、M5）**: DiTのアテンションをccvのint8アテンション
  カーネルで計算します。15秒・20ステップの動画で約1.44倍速く、短い動画では
  ほとんど速くなりません。既定はオフ。詳細は[高速モード](#高速モード試験的)。
- **プロジェクト**: プロジェクトはフォルダ（既定は`~/Movies/H3cApp/<名前>`）で、
  `project.json`（プロンプトとすべての設定）、`references/`（使った画像・動画・
  音声のコピー）、生成した動画（`<日付>-<時刻>_seed<シード>.mp4`と、その
  リクエストをそのまま記録した`.json`）が入ります。動画は削除するまで残ります
  （削除はゴミ箱へ）。プロジェクトを開くとフォームが保存時の状態に戻り、動画の
  「設定を戻す」はシードを固定して同じ動画をつくり直せるようにします。生成した
  動画を参照に使え、プロンプト欄の本数で、シードを変えて続けて生成しプロジェクトに
  保存できます（シード固定なら1本目だけそのシード、2本目以降はランダム）。プロジェクトを開いていないときは、
  生成するときに新規作成するか既存のものを選び、既存のプロジェクトの保存済みの
  プロンプトと設定が違えば上書きしてよいか確認します。2本目以降はエンコード
  済みのプロンプトと参照を使い回します（テキストのみで約8秒、参照画像ありで
  約14秒の短縮）。プロジェクトなしでは
  従来どおり一時ファイルです。「プロジェクトとして保存」で、今のフォームから名前を
  付けてプロジェクトを作り、その動画を記録と一緒に移せます（使った入力ファイルも
  コピーします）。同じリクエストとシードなら、生成されるフレームは
  ビット単位で同じです。ただしH.264ファイル自体は、ハードウェアエンコーダーが
  ビット単位では再現しないため、見分けられない程度（PSNR約57〜60dB）に違うことが
  あります。
- **サイズ**: プロンプト欄で画面の形（横長・正方形・縦長）と大きさ（小・中・大）を
  選び、それぞれ生成後に2倍に拡大することもできます（Accelerateの高品質
  リサンプリング。増えるのは数秒です）。生成サイズは正方形256・512・768、
  横長448×256・672×384・960×544、縦長はその縦横を入れ替えたもので、拡大した
  最大の出力は1536×1536と1920×1088・1088×1920です。詳細設定では完成サイズを
  カスタムで指定できます（32の倍数。2倍拡大ではその半分で生成するため64の倍数）。
  生成するサイズは60万ピクセル以下（1辺256〜1344）です。また、このMacの過去の
  実績から全体の所要時間の目安を表示し、1時間を超える見込みなら警告します。時間はおおむね生成ピクセル数に比例し、24GBのM5で
  1秒・20ステップは512×512で107秒、960×544で207秒、768×768で234秒でした。
- **ウィンドウ**: プレビューをウィンドウ全体に表示し、プロンプトはその上に
  浮かぶパネルで入力します（Enterで生成、Shift+Enterで改行、Tabで入力例を
  入力。生成中はパネルが小さくなり、プロンプトの行をクリックすると開き
  ます）。詳細設定ダイアログ（⌘,）、名前を付けて保存できる設定プリセット
  （起動時は前回使ったプリセットを適用）、プレビューのピンチ／マウス
  ホイールでの拡大縮小、書き出し先の既定をダウンロードフォルダにし前回の
  フォルダを記憶、にも対応しています。全画面表示では上のツールバー（プロジェクト
  メニューとタイトル）を隠してプレビューを画面全体に表示し、マウスを画面の上端に
  移動すると表示します。
- **M5でのint8動画VAE**: 動画VAEデコーダーのTransformer線形層をint8
  TensorOpsカーネルで計算し、F32比で約3倍速いデコードをPSNR 46dBの画質で
  実現しています（`H3_VAE_INT8=0`でF32に戻せます）。
- **モデルマネージャー**: 固定の1パスではなく、複数のH3チェックポイント
  ディレクトリとLoRAファイルを登録・切り替え可能。FL2VA/Ref2VAは外部依存
  なし（Python/huggingface_hub不要）でHugging Faceから直接ダウンロード
  でき、初回起動時に自動的に案内されます — 素朴だが再開可能なURLSession
  ベースのダウンローダーです。詳細は
  [ModelDownloader.swift](native/H3Spike/Sources/H3cApp/ModelDownloader.swift)。
- **組み込みローカルAPI**: `127.0.0.1`上で動くプレーンなHTTP/JSON API。
  生のPOSIXソケット上でアプリ自身が提供しており（Network.framework・
  サードパーティサーバー不要）、ウィンドウと全く同じジョブ/エンジン状態を
  操作します。詳細は[API](#api)。

## 計算方式

DiTの約37GiBのBF16重みは、生成のたびに何らかの形でGPUに供給する必要が
あります。互いに排他的な3つの方式があり、いずれもアプリの計算方式選択
（またはAPIの`compute_mode`フィールド）から選べます。

| | int8アテンションキャッシュ | 常駐モード | SSDストリーミング |
|---|---|---|---|
| アプリ／APIでの値 | 「高速（int8キャッシュ）」／`attentionCache` | 「常駐（大容量メモリ向け、キャッシュ不要）」／`resident` | 「省メモリ（SSDストリーミング）」／`ssdStreaming` |
| 準備 | 一度だけのキャッシュ作成（必要な場合はアプリのウィンドウ内で提案されます。チェックポイントごとに約28秒、約18GiBのint8キャッシュファイルを作成） | ディスクへの準備は不要 — 起動のたびにその場で量子化（キャッシュ作成と同じ約28秒のコストを、起動・モデル切り替えのたびに払う） | 不要 — チェックポイントをそのまま読む |
| 速度 | 最速の実測経路 | キャッシュ方式と同程度（こちらもステップごとのディスクI/Oなし） | より遅い。22フレーム/512正方形のクリップで141秒 vs. キャッシュ方式の約78秒（いずれも20ステップ） |
| メモリ | int8量子化された重みをダブルバッファリングで常時ストリーミング | 全DiTブロックを常時メモリに展開 — Tensor演算ユニット搭載GPUではint8量子化で約18GiB、それ以外では完全なBF16のまま約37GiB | 常時DiTブロック2個分のみ常駐（追跡ストレージ約2GiB） |
| 必要条件 | M5クラスGPU（Metal 4 TensorOps／int8経路） | 任意のApple Silicon GPU（Tensor演算ユニットがない場合はBF16のまま常駐）— 大容量メモリのMac向け | 任意のApple Silicon GPU |
| LoRA | 対応 | 対応 | 対応 |

キャッシュ形式の詳細 — バージョン管理、`model_kind`/`model_id`の不一致
検出 — は[h3_attention_cache.c](h3_attention_cache.c)にあります。
キャッシュは独立したステップとして
`build_attention_cache <FL2VA/transformer dir> <output file>`
（`make build_attention_cache`）で作成することもでき、モデルの
ルートディレクトリを指定すればFL2VA/Ref2VA両方のキャッシュを一度に
作成できます。

生成時間が無制限であることと、参照動画が大きすぎることは、どちらも同じ
制約に突き当たります — 次節を参照してください。

### ライブラリレベルの設定

エンジン自体（`h3_dit.c`）は、いくつかの環境変数を直接読み取ります —
`H3_ATTENTION_CACHE`、`H3_ATTENTION_CACHE_DIR`、`H3_INT8_STREAM_MLP`、
`H3_QWEN_PREFETCH*`、`H3_ZERO_COPY_WEIGHTS`、`H3_VAE_TILE_PIXELS`、
`H3_VAE_INT8`、`H3_DIT_COMMAND_BLOCKS`、`H3_PROFILE`、および多数の
`H3_DISABLE_*`/`H3_USE_SLOWER_*`系のA/B診断用スイッチです。これらは
`libh3.a`を直接リンクする人向けのもので、アプリ自身は通常の利用では
これらに頼らず、同じ内部オプションを自前のUIと`H3GenerationParams`
経由で操作しています。それぞれソース中の使用箇所
（[h3_dit.c](h3_dit.c)、[h3_gpu.m](h3_gpu.m)、
[h3_attention_cache.c](h3_attention_cache.c)から辿るのがおすすめ）に
ドキュメント化されています。

## 速度モード

計算方式とは独立に、詳細設定の「速度」（またはAPIの`speed_mode`）で
エンジンの近似の度合いを選べます。

| モード | エンジン設定 | M5で512×512・124フレーム（5秒）・20ステップのDiT部分 |
|---|---|---|
| 標準（`quality`） | 近似なし。ノイズ除去の再利用（`reuse`）は設定どおり（既定2） | reuse 1で557秒 |
| 高速（`fast`） | Transformerコアを4ステップ間再利用、`reuse` 1 | 204秒（約2.7倍） |
| 最速（`fastest`） | 高速＋トークン削減 | 189秒（約2.9倍） |

高速・最速は生成の経過が変わるため、同じシードでも構図は標準と異なります。
コア再利用はステップ数に応じて調整され（ステップ数÷5、最大4）、4ステップの
Turbo LoRAではトークン削減のみが効きます。エンジンはコア再利用と2以上の
`reuse`を併用できないため、高速・最速は`reuse` 1で動きます。保存されている
`reuse`の設定は変わらず、標準に戻すと再び使われます。

どのモードも50個のDiTブロックをすべて使います。0.2.0までは高速・最速が
ゲートに基づいて5ブロックを省略（50中45）していましたが、これで音声が壊れる
（大きな広帯域ノイズと帯状の音）ことが分かったため、0.3.0で外しました。
ブロック数は詳細設定の「使用する層数」（35〜50、既定50、APIの`dit_layers`）
で単独に指定でき、減らすと音声が壊れることがある旨の警告が出ます。

### 高速モード（試験的）

詳細設定の「高速モード（試験的）」（APIの`fast_attention`）は、DiTの
アテンションをエンジン自身の経路ではなく
[ccv](https://github.com/liuliu/ccv)のint8アテンションカーネル（入出力は
BF16）で計算します。既定はオフで、使える環境でだけ表示されます。使うには、
アプリがccvをリンクしてビルドされていること（リリースの`.dmg`はリンク済み。
[ネイティブアプリ](#ネイティブアプリ)参照）と、GPUにニューラル行列演算
ユニットがあること（M5）が必要です。それ以外の環境で指定するとエラーになり、
黙って通常経路に切り替わることはありません。

M5・24GBのMacで768×768・15秒・20ステップ・シード7の実測（コマンドライン
ツール）: 13727秒 → 9506秒（1.44倍、70分短縮）、ピークのメモリ使用量は
21.09 → 22.35GiB。効果は画面が大きく動画が長いほど大きく、256×256では
ほぼありません。速度プリセットと組み合わせても効きます（下の表を参照）。
同じシードでも標準とは別の動画になります。いくつかのプロンプトとシードで
並べて比較した範囲では一貫した画質の低下は見られませんでしたが、同等の
画質の証明ではありません。

### 生成時間の実測

M5・24GBのMacで、アプリのAPIからグリッドサーチで測った値です（2026-10-02〜03）。
正方形、int8キャッシュ、20ステップ、プロンプト「A cat playing with a ball of
yarn.」、シード7、各1回、リクエストから動画ファイル完成までの時間。標準は
既定の`reuse` 2、高速・最速は`reuse` 1とコア再利用4で動きます。各欄は
「高速モードなし / あり」です。

| 大きさ | 長さ | 標準 | 高速 | 最速 |
|---|---|---|---|---|
| 256×256 | 5秒 | 1:34 / 1:39 | 1:08 / 1:09 | 0:58 / 0:58 |
| 256×256 | 10秒 | 2:54 / 2:54 | 1:48 / 1:48 | 1:28 / 1:28 |
| 256×256 | 15秒 | 4:24 / 4:19 | 2:44 / 2:44 | 2:08 / 2:08 |
| 512×512 | 5秒 | 5:59 / 5:39 | 3:44 / 3:29 | 2:44 / 2:38 |
| 512×512 | 10秒 | 15:56 / 13:35 | 9:25 / 8:10 | 6:14 / 5:39 |
| 512×512 | 15秒 | 30:08 / 24:27 | 17:16 / 14:10 | 11:05 / 9:25 |

- 速度プリセットの効果は、画面が大きく動画が長いほど大きくなります。標準に
  対して、高速は1.4〜1.75倍、最速は1.6〜2.7倍速くなりました。
- 高速モードは256×256では効きません（5秒の標準ではかえって5%遅い）。
  512×512ではプリセットと組み合わせても効き、5秒で3〜7%、10秒で9〜15%、
  15秒で15〜19%短くなりました。512×512・15秒で最も速いのは最速＋高速
  モードの9分25秒で、標準の30分8秒の約3.2倍速です。
- 時間は、256×256では長さにほぼ比例し（5・10・15秒で1 : 1.9 : 2.8）、
  512×512ではそれより大きく伸びます（標準で1 : 2.7 : 5.0）。
- スワップは36回を通じて559〜715MiBにとどまり、512×512・15秒も24GBに
  収まりました。

## LoRA

LoRAをキャッシュファイルへ融合することはありません。[h3_lora.c](h3_lora.c)が
生成ごとに各ファイルを一度読み、テンソルをエンジン自身の重みの並びに対応付け
（ComfyUIとdiffusersはアテンションのq/k/vを連続して格納しますが、公式
チェックポイントはヘッドごとに交互に並んでいます。diffusersはさらにSwiGLUの
`fc1`の前半と後半が逆です）、重ねるLoRAをrank方向に連結します。ブロックの
重みがGPUに載るたびに — 常駐する重みはロード時に1回、ストリームされる重み
（int8キャッシュ・SSD）は毎ステップ — 投影ごとに1回の行列積で
`Σ 強さ_i · alpha_i/rank_i · B_i A_i`をその場で加えます。この差分はint8や
BF16の1刻みよりはるかに小さいため確率的丸めで加えます（最近接丸めでは大半が
消えてしまいます）。丸めのノイズは決定的なので、同じシードからは同じ動画が
得られます。

対象は50個のDiTブロックと2個のトークンリファイナーブロックの`qkv`・`out`・
`fc1`・`fc2`です。ファイル内のそれ以外（`adaln_proj`、`final_layer`など）は
無視され、その件数はモデルマネージャーに表示されます。M5・512x512・39フレーム・
4ステップでの実測では、lightx2vのTurbo LoRA（rank 128、全50ブロック）の追加
コストはint8キャッシュで1ステップあたり約1.8秒（7.5 → 9.3秒）、SSD
ストリーミングで約2秒（11.9 → 13.9秒）でした。rank 16のLoRAをさらに重ねても
差は測定できない程度です。同じTurbo LoRAのdiffusers版とComfyUI版からは
ビット単位で同一の動画が得られます。

## 参照条件付けのコスト：なぜ参照動画を大きくすると遅くなるのか

Ref2VAの参照は一度エンコードされてキャッシュされるのではなく、DiTが
全レイヤー・全denoisingステップで、メインの動画/音声潜在表現と共に
毎回再度アテンション計算します。アテンションコストはトークン総数に対して
少なくとも2乗で増加するため、参照動画を大きく・長くすると生成**全体**の
コストが跳ね上がります（一度きりのエンコードコストではありません）。
大きい・長い参照動画では実測で4時間以上かかったケースがあります（正常に
完了はします）。上記のメモリ連動の上限は極端なケースのリスクを減らします
が、このスケーリング自体は変わりません。

**参照動画使用時に生成が想定より遅い場合**: より小さい・短い参照動画に
するか、denoisingステップ数を減らすか、`reuse`の値を上げてください。
極端なケースは、メモリ不足を招く前に明確なエラーで拒否されます
（MPSGraphフォールバック経路の512MiBアテンションマスク上限、
[h3_gpu.m](h3_gpu.m)参照）。

## リポジトリ構成

```
h3.c, h3_dit.c, h3_gpu.m, ...   コア推論エンジン（C + Objective-C/Metal）。libh3.aをビルド
h3.h                            公開C API（h3_load_dir、h3_generate、h3_build_attention_cache、...）
h3_shaders.metal                すべてのMetalコンピュートカーネル
h3_build_attention_cache.c      h3_build_attention_cache()のCLIラッパー -> build_attention_cache
h3_lora.c                       LoRAの読み込み（形式の正規化）とGPU上での重みへの適用
tests/                          Cテストスイート（make test / make parity）
native/H3Spike/                 ネイティブmacOSアプリ（SwiftPM）
  Sources/CH3                   libh3.aのC APIをSwiftへ橋渡しするCシム
  Sources/H3Engine              C APIのSwift非同期ラッパー（AsyncThrowingStreamベースの進捗/キャンセル）
  Sources/H3cApp                SwiftUIアプリ本体（H3cApp.app）。ModelLibrary/ModelManagerView（登録済み
                                 モデル/LoRA）、ModelDownloader（Hugging Faceダウンロード）、
                                 HTTPServer/GenerationViewModel+API（組み込み自動化API）を含む
  Sources/H3Spike                H3Engine用の最小限のプロセス内スパイク/参照クライアント。配布アプリ本体ではない
  package_app.sh                H3cApp.appをビルド・パッケージング。署名・公証も行う（下記参照）
  make_dmg.sh                   ビルド済みH3cApp.appを配布用.dmgにまとめる
```

## ソースからのビルド

### ライブラリ

```sh
make -j8 libh3.a     # C/Objective-Cエンジンをビルド
make test             # 決定論的なホストテスト一式（+ フィクスチャがあればMetal/MLXパリティ）
make parity            # Metal/MLXの数値比較チェックのみ
```

`make build_attention_cache`は、上記の独立したキャッシュ準備ツールを
ビルドします。デフォルトの`make`ターゲットには含まれません。

### ネイティブアプリ

```sh
cd native/H3Spike
./package_app.sh
```

`libh3.a`はこのスクリプトでは再ビルドされません — リポジトリ直下で
`make libh3.a`を先に（エンジンのコードを変更した後も同様に）実行して
ください。`package_app.sh`はその後`swift build -c release`を実行し、
`H3cApp.app`を組み立てます。SwiftPM標準のリポジトリ内`.build`ではなく
リポジトリ外のスクラッチディレクトリ（`${TMPDIR}h3c-app-build-scratch`）
にビルドします。これは、リポジトリが同期フォルダ（Google Drive、
iCloud Drive、Dropboxなど）配下にあると、その同期デーモンがSwiftPMの
`build.db`のロックを保持してしまい、見せかけの「disk I/O error」で
ビルドが断続的に失敗することがあるためです — このエラーを見たら、
単なる不安定さと決めつける前に、チェックアウト先が同期対象ディレクトリ
内でないか確認してください。

高速モードにはccvが必要です。[tools/ccv_eval/README.md](tools/ccv_eval/README.md)
の手順でパッチを当てたMPS対応のccvをビルドし、そのパスを両方の手順に渡します
（リリースの`.dmg`はこの方法でビルドしています）。

```sh
make -j8 libh3.a CCV_DIR=/path/to/ccv
cd native/H3Spike
CCV_DIR=/path/to/ccv ./package_app.sh
```

`CCV_DIR`なしでは、これまでどおり高速モードなしでビルドされます。

画面は英語と日本語に対応し、macOSの言語設定に従います。Swiftソース中の
日本語の文字列がキーで、訳は`native/H3Spike/Packaging/{en,ja}.lproj/Localizable.strings`
にあり、`package_app.sh`がアプリに入れます。翻訳されるのは、SwiftUIの
リテラル（`Text("…")`、`Button("…")`など）か`String(localized: "…")`として
画面に渡る文字列だけです。`native/H3Spike`の`./check_localizations.sh`は
コンパイラでキーを抽出し、どちらかの表に無いものを一覧にします。別の言語で
試すには`open .build/H3cApp.app --args -AppleLanguages '(en)'`。

#### コード署名と公証

署名と公証は2つの環境変数によるオプトイン方式なので、どちらも設定
しない`./package_app.sh`は未署名の開発ビルドを生成します。

```sh
H3C_SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" \
H3C_NOTARY_PROFILE="some-keychain-profile" \
./package_app.sh
```

- `H3C_SIGN_IDENTITY` — `security find-identity -v -p codesigning`で
  確認できる`Developer ID Application`証明書。Apple Developer Programへの
  登録と、この証明書の作成が必要です（Xcode → Settings → Accounts →
  Manage Certificates → `+` → Developer ID Application — Xcodeプロジェクト
  は不要で、アカウント管理機能を使うだけです）。署名だけ（
  `H3C_NOTARY_PROFILE`なし）でも、Hardened Runtime有効な状態でローカル
  実行するには十分です。
- `H3C_NOTARY_PROFILE` — `xcrun notarytool store-credentials <profile>
  --apple-id ... --team-id ...`で一度だけ保存するプロファイル名
  （通常のApple IDパスワードではなく[App用パスワード](https://appleid.apple.com)
  が必要）。両方の変数を設定すると、スクリプトはさらにzip化・公証提出・
  完了待ち・チケットのステープルまで行い、生成された`.app`はどのMacでも
  Gatekeeper（`spctl -a -vvv -t exec`）を通過するようになります。

ここでは気にする必要のあるネストされたフレームワークや埋め込みdylibは
ありません — `H3cApp`がリンクしているのはAppleのシステムフレームワークと
静的リンクされた`libh3.a`（`CCV_DIR`指定時は`libccv.a`も）だけなので、バンドルに対する単純な
`codesign --deep`一回で十分です。

#### .dmgとして配布する

```sh
./make_dmg.sh
```

すでにビルド済みの`H3cApp.app`を、Applicationsへのドラッグ用ショート
カット付きの圧縮`.dmg`にまとめます。`hdiutil`のみを使用（サードパーティ製
のdmg作成ツールは不使用）。先に`package_app.sh`を実行しておいてください。
`H3C_SIGN_IDENTITY`と`H3C_NOTARY_PROFILE`が設定されていれば、
`make_dmg.sh`は`.dmg`ファイル自体にも署名・公証・ステープルを行います
— Gatekeeperの実際のチェックは起動時に`.app`自身の署名・ステープルに
対して行われるため必須ではありませんが、ダウンロード時の体験を完全に
クリーンにするため`.dmg`自体もGatekeeperを通過するようにできます。

## テスト

```sh
make test
make parity
```

`make test`は決定論的なホストテスト一式を実行し、（gitで無視されている）
MLXフィクスチャが`misc/fixtures/`に配置されていれば、Metalソースを
実行時にコンパイルして、トイのH3ブロックを名前付きMLX出力と比較検証
します — Irisにならい意図的に実行時コンパイルとしており、Xcodeの
オプションであるオフラインMetalツールチェーンを必要としません。
`make parity`はそのMetal/MLXチェックのみを実行します。

同じシードでの再現性：`make test`には`h3_determinism_tests`（threadgroupで
集計するカーネルを繰り返し実行し、ビット単位で一致することを確認）が
含まれます。公開済みの重みがあれば、`make h3_repro_check CCV_DIR=...`の後に
`./h3_repro_check`を実行すると、短いRef2VA動画を3回生成し（約4分）、
エンコーダーに渡すRGBフレームとデコード後の音声が一致しなければ失敗します。
`--run`で実行ごとに条件を変えられ、`cache`以外が同じ実行どうしを比較します
（例：`--run cache=0 --run cache=conditioning --run cache=conditioning`で、
キャッシュなし・初回・再利用を比べます）。

## API

`H3cApp.app`が起動している間、`http://127.0.0.1:8420`でプレーンな
JSON APIを提供します — 生のPOSIXソケットで実装されており
（[HTTPServer.swift](native/H3Spike/Sources/H3cApp/HTTPServer.swift)参照）、
Network.frameworkやサードパーティ製サーバーは使っていません。別プロセスの
起動・停止も不要です。このAPIは、アプリの主となる`GenerationViewModel`/
エンジンのインスタンス（最初のウィンドウが表示するもの）を操作します
（[AppModels.swift](native/H3Spike/Sources/H3cApp/AppModels.swift)、
[GenerationViewModel+API.swift](native/H3Spike/Sources/H3cApp/GenerationViewModel+API.swift)参照）。
アプリが起動していれば、ウィンドウをすべて閉じても受け付けます。ウィンドウを
閉じるとそのプロジェクトも閉じるので、必要なら`POST /api/project/open`で
開いてください。ジョブは常に1つだけで、そのインスタンスを表示している
ウィンドウと共有されます — 生成ボタンを押すのと`POST /api/generate`は同じ枠を
奪い合い、負けた方には明確な`409`が返ります。

クライアントとサーバーは常に同じMac上にあるため、メディア入力はアップ
ロードではなく、単なるファイルシステムパスです。

| メソッド | パス | 内容 |
|---|---|---|
| `POST` | `/api/generate` | ジョブを開始。JSONボディで現在のドラフトを完全に置き換えます（下記参照）。開始できれば`202`、リクエスト不正なら`400`、ジョブ実行中なら`409`。 |
| `GET` | `/api/status` | エンジン/ジョブの状態、進捗割合、フェーズ/ステージ文言、エラーメッセージ、結果が用意できているか。 |
| `POST` | `/api/cancel` | 実行中のジョブがあれば中止。 |
| `GET` | `/api/result/video` | 現在の結果を`video/mp4`としてストリーミング。結果が無いか、次のジョブの`generate()`呼び出しで削除された後は`404`。 |
| `GET` | `/api/models` | 登録済みH3モデルディレクトリ一覧（id・名前・パス・アクティブかどうか）。 |
| `GET` | `/api/loras` | 登録済みLoRAファイル一覧（id・名前・パス・強さ・推奨ステップ数・オンかどうか）。 |
| `GET` | `/api/project` | 開いているプロジェクト（名前・パス・本数・動画一覧（ファイル・シード・完成日時・生成時間））。 |
| `POST` | `/api/project/new` | `{"name", "directory"?}` 今のフォームからプロジェクトを作って開く（既定の場所は`~/Movies/H3cApp`）。 |
| `POST` | `/api/project/open` | `{"path"}` プロジェクトのフォルダを開く。フォームは保存時の状態になる。 |
| `POST` | `/api/project/close` | 閉じる。結果は再び一時ファイルになる。 |
| `POST` | `/api/project/restore` | `{"video"}` その動画のリクエストをシード固定でフォームに戻す。 |
| `POST` | `/api/project/use-as-reference` | `{"video"}` その動画を参照動画としてフォームに追加する。 |
| `POST` | `/api/project/delete-video` | `{"video"}` 動画と記録をゴミ箱に移す。 |

`POST /api/generate`のボディフィールド（`prompt`以外はすべて省略可）:

| フィールド | 既定値 | 備考 |
|---|---|---|
| `prompt` | — | 必須。 |
| `size_profile` | `"square"` | プリセット`<形>-<大きさ>`（形は`square`・`landscape`・`portrait`、大きさは`small`・`medium`・`large`）。2倍に拡大した出力は末尾に`-x2`（例: `landscape-large-x2`は960×544を1920×1088に拡大）。または完成サイズを表す`custom-<幅>x<高さ>`／`custom-<幅>x<高さ>-x2`（32の倍数。`-x2`はその半分で生成するため64の倍数）。生成するサイズは60万ピクセル以下で1辺256〜1344。以前の名前`smallSquare`、`square`、`landscapeNative`、`landscapeUpscaled`、`portraitNative`、`portraitUpscaled`もそのまま使え、対応する小・中のプリセットを表します（[GenerationModels.swift](native/H3Spike/Sources/H3cApp/GenerationModels.swift)の`SizeProfile`参照）。 |
| `seconds` | `5` | 1〜15。 |
| `steps` | `20`、またはオンにした最初のTurbo LoRAの推奨ステップ数 | 3〜40。 |
| `reuse` | `2` | 1〜3。高速・最速ではこの値にかかわらず1で動く。 |
| `compute_mode` | このGPUでのアプリの既定値 | `attentionCache`、`resident`、`ssdStreaming`のいずれか。 |
| `speed_mode` | `"quality"` | `quality`、`fast`、`fastest`のいずれか（[速度モード](#速度モード)参照）。 |
| `dit_layers` | `50` | 35〜50。減らすと音声が壊れることがある。 |
| `fast_attention` | `false` | `true`で[高速モード](#高速モード試験的)。使えない環境では`400`。 |
| `seed` | ランダム | 0〜18446744073709551615。数値か10進数の文字列（JSONの数値を倍精度で扱うクライアントでも、文字列なら大きなシードが丸められない）。 |
| `first_frame_path` / `last_frame_path` | なし | FL2VAのアンカー。`reference_paths`とは併用不可。 |
| `reference_paths` | `[]` | 順序付きのRef2VA参照。画像/動画/音声はパスごとに自動判定。音声パスを含める場合、画像か動画を最低1つ含める必要あり。 |
| `loras` | `[]` | このジョブで重ねるLoRA。`GET /api/loras`の名前、または`{"name": ..., "strength": 0.8}`形式（強さは学習時の効き具合に対する倍率。省略するとそのLoRAの保存済みの強さ）。省略すると、ウィンドウ側でオンにしていてもこのジョブではLoRAなし扱いになる。 |
| `lora_name` / `lora_scale` | なし | `loras`の旧・単一LoRA形式。`loras`を指定した場合は無視される。 |
| `count` | `1` | 1〜20。シードを変えて続けて生成する本数。2以上はプロジェクトを開いているときだけ。 |
| `from_form` | `false` | `true`でフォームの今の状態から生成する（`/api/project/restore`の後など）。併用できるのは`count`だけ。 |

範囲外の値や型の違う値（整数の項目に`true`、`"5"`、`2.5`など）は、使える
範囲を示した`400`になります。黙って範囲内に収めたり丸めたりはしません。

```sh
curl -X POST http://127.0.0.1:8420/api/generate -H "Content-Type: application/json" -d '{
  "prompt": "A red fox walks through fresh snow in a pine forest.",
  "size_profile": "square", "seconds": 5, "steps": 20
}'

curl http://127.0.0.1:8420/api/status
curl http://127.0.0.1:8420/api/result/video -o fox.mp4
```

## ライセンス

MIT — [LICENSE](LICENSE)を参照。サードパーティ表示（`h3_shaders.metal`内の
一部設計はccvのFlashAttention実装を基にしています）は
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)にあります。推論エンジンは
Salvatore Sanfilippo氏による
[antirez/h3.c](https://github.com/antirez/h3.c)を起源としています。
