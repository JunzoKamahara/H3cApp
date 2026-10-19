# GGUF（量子化モデル）のダウンロードと取り込み

[English](GGUF.md) | [日本語](GGUF.ja.md)

H3cApp v0.6.0 から、公式のモデル（BF16、FL2VA だけで約134GiB）の代わりに、
量子化されたモデル（GGUF）を使えます。ダウンロードとディスクの使用量が
大きく減ります（おすすめの組み合わせで約45GB、最小で約35GB）。

この文書では、GGUF をダウンロードしてモデルとして使えるようにする方法と、
手元にある GGUF を取り込む方法、その仕組みと注意点を説明します。

- [GGUF で用意するもの・しないもの](#gguf-で用意するものしないもの)
- [方法1：ダウンロードウィザードを使う（おすすめ）](#方法1ダウンロードウィザードを使うおすすめ)
- [方法2：手元の GGUF を取り込む](#方法2手元の-gguf-を取り込む)
- [取り込んだあと](#取り込んだあと)
- [配布元と対応している形式](#配布元と対応している形式)
- [動作を確かめた組み合わせ](#動作を確かめた組み合わせ)
- [モデルの削除とディスク容量](#モデルの削除とディスク容量)
- [仕組み](#仕組み)
- [API](#api)
- [ライセンス](#ライセンス)

## GGUF で用意するもの・しないもの

MiniMax H3 のモデルは、いくつかの部品でできています。GGUF で配られているのは
重みの大きな部品だけで、残りは公式のモデルから取ります。

| 部品 | 用意の仕方 | 大きさの目安 |
|---|---|---|
| DiT（FL2VA：文章から・最初と最後の画像から） | GGUF | Q8_0 で 21.4GB、Q4_K で 11.4GB |
| DiT（Ref2VA：参照画像・動画・音声から） | GGUF（任意） | FL2VA と同じくらい |
| 文章の部分（テキストエンコーダー、Qwen3-VL-32B） | GGUF | H3専用 Q4_K_M で 18.2GB、Qwen 公式 Q8_0 で 34.8GB |
| 画像の部分（ビジョンタワー、mmproj） | GGUF（文章の部分に入っていれば不要） | F16 で 1.2GB |
| トークナイザー・設定 | 公式のモデルから（GGUF はない） | 数十MB |
| 映像・音声の VAE | 公式のモデルか、unsloth の safetensors（GGUF はない） | 公式 11GB、unsloth 5.9GB |

トークナイザーは重みではなく数MBの辞書なので、GGUF にする意味がなく、配られていません。
VAE も GGUF では配られていませんが、unsloth と Comfy-Org が単体の safetensors
（映像 VAE を FP16 にした小さいもの）を配っており、H3cApp はそれも読めます。

## 方法1：ダウンロードウィザードを使う（おすすめ）

1. 最初の起動時に出る「MiniMax-H3 モデルの用意」を使います。モデル管理の
   「ダウンロードして追加…」からも開けます。
2. 「ダウンロードするもの」で「GGUF（量子化）と必要な部品」を選びます。
3. 部品ごとに、配布元とファイルを選びます。

   | 部品 | おすすめ | 小さくしたいとき |
   |---|---|---|
   | DiT（FL2VA） | unsloth の Q8_0（21.4GB、動作確認済み） | Q4_K（11.4GB） |
   | 文章の部分 | unsloth の H3専用 Q4_K_M（18.2GB、50層・画像の部分入り） | — |
   | VAE | unsloth の VAE（5.9GB、動作確認済み） | — |

   - 文章の部分の画質をいちばん確実にするなら、Qwen 公式の Q8_0（34.8GB、動作確認済み）
     も選べます。この場合は取り込みのときに使う50層だけに切り詰めて書き出す
     （約27GB）ので、そのぶんの空きも必要です。画像の部分が入っていないので、
     mmproj（F16、1.2GB）も自動で加わります。
   - 参照画像・動画・音声を使う生成もしたいときは「Ref2VA も含める」をオンにします。
4. 「保存する場所」と「フォルダ名」を決めます。ここが取り込んだモデルのフォルダに
   なり、ダウンロードしたファイルもいったんこの中（`.downloads`）に置きます。
   外付けディスクを選んでも、取り込みは同じディスクの中の複製なので容量は二重に
   なりません。
5. ライセンスの注意書きを読み、「ダウンロードして取り込む（〜GB）」を押します。
   ダウンロード、取り込み、モデル管理への登録まで自動で進みます。
   中断しても、次回は続きからダウンロードします。

ボタンを押す前に、ダウンロードの量・取り込みで書き出す量・保存先のディスクの
空き容量を表示します。足りないときはボタンが押せません（2GB の余裕を残します）。

### 配布元を変える

各部品の配布元（Hugging Face のリポジトリ）は変えられます。

- 「≡」から、その部品を配っている既知の配布元を選べます。
- 「↓」（または入力欄で Return）を押すと、その配布元の全ファイルの先頭を読み、
  中身を見てその部品のファイルだけを並べます。ファイル名ではなく中身で見分けるので、
  配布元ごとに名前の付け方が違っても正しく分かれます。一度読んだ結果は保存され、
  2回目からはすぐ出ます。
- 「≡」→「カスタム…」で、リポジトリ名か Hugging Face の URL を入力できます。
  リポジトリ（`https://huggingface.co/owner/name`）、フォルダ
  （`…/tree/main/vae`）、1つのファイル（`…/blob/main/xxx.gguf`、
  `…/resolve/main/xxx.gguf`）、`main` 以外のブランチのどれでも構いません。
  入力した場所は部品ごとに「カスタム」として残り、「カスタムを削除」で消せます。
- 選んだ配布元は部品ごとに記憶され、次に開いたときの初期値になります。

アプリで使えないファイルも一覧には出し、「（未対応：理由）」と添えます（選んでも
選ばれません）。理由は[対応している形式](#対応している形式)を参照してください。

## 方法2：手元の GGUF を取り込む

すでに GGUF ファイルを持っている場合は、モデル管理の「GGUFを取り込む…」を使います。

1. **ファイルを選ぶ**：「GGUFファイルを追加…」でファイルを、「フォルダから追加…」で
   フォルダごと（3階層下まで、リンクになったフォルダもたどる）追加します。
   追加したファイルは中身を調べて、DiT・文章の部分・画像の部分のどれかを表示します。
   DiT の FL2VA／Ref2VA は中身では区別できないので、名前に `ref2va` を含むものを
   Ref2VA とみなし、切り替えもできます。「必要な部品」「足りない部品」の欄に、
   入手先の例とリンク、「ダウンロード…」ボタンが出ます。
2. **内容の確認**：部品ごとに、どのファイルを使うか、足りないものは何かを表示します。
   - 同じ部品のファイルが複数あるときは、使うものを選びます。同じファイルへのリンク
     や、大きさと形式が同じファイルは、その旨を表示します。
   - 使われないファイルには、その理由を表示します。
   - 「部品の複製元」で、トークナイザー・VAE・設定をどこから取るかを選びます。
     モデル管理に登録したモデル、「フォルダを登録…」で選んだフォルダ、または
     「共通部品をダウンロード…」で取った公式の共通部品（約6〜11GB）です。
   - 新しく書き出す量（文章の部分の切り詰めなど）と空き容量を表示します。
3. **取り込み**：「取り込む」を押すと、`~/Library/Application Support/H3cApp/Imported/`
   にモデルのフォルダを作り、モデル管理に登録します。

## 取り込んだあと

- モデル管理の「ⓘ」で、モデルの中身（各部品の形式・大きさ、AdaLN の形、層の数、
  キャッシュの有無、取り込みの記録）と、このモデルでできることを確かめられます。
- M5 で速く生成したいときは、詳細設定の「キャッシュを作成…」で int8 キャッシュ
  （約19GB）を作ります。GGUF のモデルでも作れます。M1〜M4 では SSD ストリーミングで
  動き、GGUF の量子化された重みのまま読み込んで GPU で展開するので、読み込む量が
  減ります。
- GGUF の DiT（曲線形の AdaLN を含むもの）は、条件の計算の近似をファイル自体に
  持っているので、別の近似ファイルは要りません。
- Ref2VA なしで取り込んだモデルには、あとから Ref2VA を加えられます。モデル管理の
  「…」→「Ref2VAを追加…」で、Ref2VA の DiT（GGUF）を Hugging Face からダウンロード
  するか、手元のファイルを選びます。文章の部分・VAE・設定はそのモデルの FL2VA から
  複製するので、増えるのは DiT の分だけです。

## 配布元と対応している形式

### 既知の配布元

| 配布元 | 部品 | メモ |
|---|---|---|
| [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) | DiT（FL2VA・Ref2VA）、H3専用の文章の部分、VAE | 標準の配布元。DiT は曲線形の AdaLN |
| [Qwen/Qwen3-VL-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF) | 文章の部分（64層）、mmproj | 公式。取り込み時に50層へ切り詰め |
| [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) | DiT | 曲線形の AdaLN |
| [Abiray/MiniMax-H3-GGUF](https://huggingface.co/Abiray/MiniMax-H3-GGUF) | DiT、文章の部分 | DiT は密な AdaLN |
| [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF) | DiT、文章の部分 | 曲線形と密な形の両方 |
| [molbal/MiniMax-H3-GGUF](https://huggingface.co/molbal/MiniMax-H3-GGUF) | DiT | Q8_CR・U16G は未対応 |
| [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) | DiT、文章の部分 | DiT は密な AdaLN（中身は Abiray と同じ重み） |
| [joeygambino/MiniMax-H3-x-Z-Image-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-GGUF) | DiT | Z-Image との合成モデル。`curve` 付きが曲線形 |
| [nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF](https://huggingface.co/nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF) | 文章の部分（50層）、mmproj | IQ 系は未対応 |
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | VAE | int8 版は未対応 |

いずれも、ダウンロードの画面で先頭を読んだ結果に基づく分類です。どの配布元も、
Q4 前後の代表的なファイルで短い動画を作れることを確かめています（下の
「動作を確かめた組み合わせ」）。

### 対応している形式

- **量子化形式**：Q8_0、Q4_0、Q4_1、Q5_0、Q5_1、Q2_K〜Q6_K、F16、BF16、F32。
  unsloth の「UD-○○_XL」のような混在も、これらだけでできていれば使えます。
- **未対応**（一覧に理由付きで出ます）：
  - IQ 系（IQ4_XS、IQ3_XXS など）、I8 などを含むファイル
  - 分割された GGUF（`-00001-of-00002` など）
  - safetensors 形式の DiT・文章の部分（取り込めるのは GGUF だけ）
  - 読めない数値形式（F8、int8 など）の VAE
  - 層が50に足りない文章の部分
- **AdaLN の形**：DiT の GGUF には、AdaLN が「曲線形」（ComfyUI の pruned 形式。
  約24GiBの重みを小さな曲線で表したもの）と「密な形」の2種類があります。どちらも
  使えます。密な形はファイルが8GBほど大きくなります。ComfyUI-GGUF が行の長さを
  揃えるために並べ替えたテンソル（`comfy.gguf.orig_shape`）は、元の形に戻して読みます。
- **文章の部分**：エンジンが使うのは Qwen3-VL-32B の最初の50層です。64層のものは
  取り込みのときに50層だけに切り詰めて書き出します。H3専用（50層）のものは
  そのまま使います。

## 動作を確かめた組み合わせ

| 組み合わせ | 確かめたこと |
|---|---|
| DiT unsloth Q8_0 ＋ 文章の部分 Qwen Q8_0（50層へ切り詰め）＋ mmproj F16 | 公式の重みと値を照合（誤差は量子化相当）、生成 |
| DiT unsloth Q4_K ＋ 文章の部分 unsloth H3専用 Q4_K_M ＋ unsloth の VAE | ダウンロード・取り込み・生成 |
| unsloth の VAE（映像 FP16・音声） | 同じ潜在を公式の VAE と比べて、映像 PSNR 平均 51.7dB、音声 SNR 72dB（見分けがつかない） |
| unsloth の H3専用 文章の部分 | テンソル名と形が公式の重みと一致。言語の部分の誤差は Q4_K で相対 7%、画像の部分はビット単位で一致 |

既知の配布元ごとに、なるべく同じ配布元・同じ量子化の DiT と文章の部分を組み合わせて、
3秒・512×512・SSD ストリーミングで同じプロンプトと seed の動画を作り、映像と音声
（話し声）を確かめました（2026年10月19日、Apple M5・24GB）。配布元に文章の部分がない
ものは unsloth の Q4_K_M を、文章の部分だけの配布元は DiT に unsloth の Q4_K を使っています。

| 配布元 | DiT | 文章の部分 |
|---|---|---|
| unsloth | `minimax_h3_fl2va_pruned-Q4_K`（曲線形） | `qwen3vl_32b_minimax_h3-Q4_K_M` |
| leejet（剪定版） | unsloth と同じファイル | unsloth と同じファイル |
| leejet | `minimax_h3_fl2va-Q4_K_M`（密な形） | unsloth と同じファイル |
| Abiray | `MiniMax-H3-FL2VA-Q4_K_M`（密な形） | `qwen3vl_32b_minimax_h3-Q4_K_M` |
| Abiray（Pruned） | `MiniMax-H3-FL2VA-Pruned-Q4_K_M`（曲線形） | Abiray の `Q4_K_M` |
| realrebelai | `MiniMax-H3-FL2VA-Q4_K_M`（密な形） | `qwen3vl-32B-MiniMax-H3-Q4_K_M` |
| molbal | `minimax_h3_fl2va_pruned_fp8_Q4_0`（曲線形） | unsloth の `Q4_K_M` |
| joeygambino | `MiniMax-H3-fl2va-curve-zs05-Q4_0`（曲線形） | unsloth の `Q4_K_M` |
| Qwen | unsloth の `Q4_K` | `Qwen3VL-32B-Instruct-Q4_K_M`（50層へ切り詰め）＋ mmproj F16 |
| nif0 | unsloth の `Q4_K` | `…-L0-49-Q4_K_M` ＋ mmproj BF16 |

画像の部分（mmproj）は、参照画像を使う生成でしか使わないため、この確認では読み込み
までです。

ただし、Q2_K だけでできた DiT（unsloth の `minimax_h3_fl2va_pruned-Q2_K`）は、人物が溶け、
背景が格子状になり、音声がうなりになるなど、動画が大きく崩れます。ブロックの行列がすべて
Q2_K で、Q8_0 との相対誤差が 27.7%（Q3_K は 15.6%、Q4_K は 7.4%）あるためです。Q3_K_M
以上を使ってください。H3cApp はこうしたファイルに警告を出します。Q2_K の文章の部分は
問題なく使え、「UD-Q2_K_XL」のような混在は確かめていません。

## モデルの削除とディスク容量

GGUF の取り込みは、同じディスク上のファイルを APFS の複製（クローン）で置くので、
取り込み自体ではほとんど容量を使いません。その代わり、モデルどうしやダウンロード
した元のファイルとデータを共有することがあります。

モデル管理の「…」→「削除…」では、次のように動きます。

- 登録を外すだけの場合でも、そのモデル用の生成キャッシュ（int8 キャッシュなど）は
  すぐに削除します（作り直せます）。
- 「モデルファイルも削除する」をオンにすると、モデルのフォルダをゴミ箱に移します。
  もう一度確認してから実行します。ゴミ箱を空にすると解放される容量を、APFS の複製の
  情報から計算して表示します。ほかのモデルや元のファイルと共有している部分は、
  そちらで使われているので残ります。
- 最後の1つのモデルを削除するときは、ダウンロードした共通部品・GGUF と、使われていない
  キャッシュも削除します。

## 仕組み

- **ダウンロード**：Hugging Face から直接、ファイルごとに再開できる形でダウンロード
  します。配布元の一覧は、各ファイルの先頭を HTTP の範囲指定で読み（GGUF は
  キーと値・テンソル表、safetensors は JSON の見出し）、エンジンと同じ検査で部品を
  見分けます。
- **取り込み**：DiT・文章の部分・画像の部分の GGUF と、複製元のトークナイザー・
  VAE・設定から、公式と同じ構成のモデルフォルダ（`FL2VA/`、必要なら `Ref2VA/`）を
  作ります。文章の部分は使う50層だけに切り詰めます。取り込みの記録は
  `h3c-import.json` に残します。
- **生成**：GGUF の量子化された重みは、使うときに GPU（Metal）で BF16 に展開してから、
  既存の行列演算で計算します。量子化のまま行列演算をするわけではありません。
  SSD ストリーミングでは量子化されたまま読み込むので、読み込む量が減ります。

エンジン側の詳細は [README の曲線形の AdaLN](README.ja.md#曲線形のadaln試験的h3_adaln_curve) と
`h3_gguf.c`・`h3_quant.c`・`h3_shaders.metal` を参照してください。

## API

アプリの JSON API（`127.0.0.1:8420`）からも同じ操作ができます。

| エンドポイント | 内容 |
|---|---|
| `GET /api/models/import/downloads?kind=dit&repo=…` | 部品（`dit`・`ref2va`・`textEncoder`・`vision`・`shared`）のダウンロード候補。`repo` にはリポジトリ名か URL |
| `POST /api/models/import` | GGUF の取り込み。`files` または `folder`、`base_model` か `base_path`、`name`、`dry_run` |
| `GET /api/models/detail?id=…` | モデルの中身（詳細画面と同じ内容） |
| `POST /api/models/remove` | 登録を外す（キャッシュは削除）。`delete_files` でファイルもゴミ箱へ、`dry_run` で計算だけ |
| `POST /api/models/add` | フォルダをモデルとして登録 |
| `POST /api/models/add_ref2va` | Ref2VA のないモデルに Ref2VA を加える。GGUF のモデルは `file`（Ref2VA の DiT）、`dry_run` で計画だけ（公式のモデルの分は計画のみ） |

## ライセンス

- MiniMax H3 のモデル（公式のもの、unsloth などの GGUF）には
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
  が適用されます。
- Qwen の文章の部分・画像の部分（Qwen3-VL-32B-Instruct）には
  [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF) が適用されます。
- それぞれの配布元のライセンスも確認してください。ダウンロードと利用の前に各ライセンスを
  読み、その条件に従ってください。
