# H3cApp ローカルAPIリファレンス

[English](API.md) | [日本語](API.ja.md)

同じMac上のスクリプトなどからH3cAppを操作する開発者向けの仕様です。H3cApp本体を起動して利用します。インストールは[README](README.ja.md)をご覧ください。

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
| `GET` | `/api/status` | エンジン/ジョブの状態、進捗割合、フェーズ/ステージ文言、エラーメッセージ、結果が用意できているか。複数本の生成やつなげて生成の間は、`batch`に`index`・`total`・`chained`と、全体の`elapsed_seconds`・`estimated_remaining_seconds`（今の1本の見積りに、残りの本数分の見積りをこれまでの実際の速さで補正して足したもの）が入る。最上位の`estimated_remaining_seconds`は今の1本の値。 |
| `POST` | `/api/cancel` | 実行中のジョブがあれば中止。 |
| `GET` | `/api/result/video` | 現在の結果を`video/mp4`としてストリーミング。結果が無いか、次のジョブの`generate()`呼び出しで削除された後は`404`。 |
| `GET` | `/api/models` | 登録済みH3モデルディレクトリ一覧（id・名前・パス・アクティブかどうか）。 |
| `GET` | `/api/loras` | 登録済みLoRAファイル一覧（id・名前・パス・強さ・推奨ステップ数・オンかどうか、モデル管理で設定したステップの進め方`sampler`と`audio_shift`、ファイルの説明による推奨`recommended_sampling`。ファイルを読み込み中は`null`）。 |
| `GET` | `/api/project` | 開いているプロジェクト（名前・パス・本数・動画一覧（ファイル・シード・完成日時・生成時間・`continued_from`・`can_continue`、つないだ動画なら`joined_clips`・`planned_clips`））。 |
| `POST` | `/api/project/new` | `{"name", "directory"?}` 今のフォームからプロジェクトを作って開く（既定の場所は`~/Movies/H3cApp`）。 |
| `POST` | `/api/project/open` | `{"path"}` プロジェクトのフォルダを開く。フォームは保存時の状態になる。 |
| `POST` | `/api/project/close` | 閉じる。結果は再び一時ファイルになる。 |
| `POST` | `/api/project/restore` | `{"video"}` その動画のリクエストをシード固定でフォームに戻す。 |
| `POST` | `/api/project/use-as-reference` | `{"video"}` その動画を参照動画としてフォームに追加する。 |
| `POST` | `/api/project/delete-video` | `{"video"}` 動画と記録をゴミ箱に移す。 |
| `POST` | `/api/project/join` | `{"video"}` その動画で終わる続きの連なりを、引き継いだ部分のクロスフェードでプロジェクトの`<名前>_joined.mp4`（記録つき）につなぎ、終わったら`{"path", "clips"}`を返す。 |

`POST /api/generate`のボディフィールド（`prompt`以外はすべて省略可）:

| フィールド | 既定値 | 備考 |
|---|---|---|
| `prompt` | — | 必須。 |
| `size_profile` | `"square"` | プリセット`<形>-<大きさ>`（形は`square`・`landscape`・`portrait`、大きさは`small`・`medium`・`large`）。2倍に拡大した出力は末尾に`-x2`（例: `landscape-large-x2`は960×544を1920×1088に拡大）。または完成サイズを表す`custom-<幅>x<高さ>`／`custom-<幅>x<高さ>-x2`（32の倍数。`-x2`はその半分で生成するため64の倍数）。生成するサイズは60万ピクセル以下で1辺256〜1344。以前の名前`smallSquare`、`square`、`landscapeNative`、`landscapeUpscaled`、`portraitNative`、`portraitUpscaled`もそのまま使え、対応する小・中のプリセットを表します（[GenerationModels.swift](native/H3Spike/Sources/H3cApp/GenerationModels.swift)の`SizeProfile`参照）。 |
| `seconds` | `5` | 1〜15。 |
| `steps` | `20`、またはオンにした最初のTurbo LoRAの推奨ステップ数 | 3〜40。 |
| `reuse` | `2` | 1〜3。高速・最速ではこの値にかかわらず1で動く。 |
| `compute_mode` | このGPUでのアプリの既定値 | `attentionCache`、`resident`、`ssdStreaming`のいずれか。 |
| `speed_mode` | `"quality"` | `quality`、`fast`、`fastest`のいずれか（[速度モード](TECHNICAL.ja.md#速度モード)参照）。 |
| `dit_layers` | `50` | 35〜50。減らすと音声が壊れることがある。 |
| `fast_attention` | `false` | `true`で[高速モード](TECHNICAL.ja.md#高速モード試験的)。使えない環境では`400`。 |
| `seed` | ランダム | 0〜18446744073709551615。数値か10進数の文字列（JSONの数値を倍精度で扱うクライアントでも、文字列なら大きなシードが丸められない）。 |
| `first_frame_path` / `last_frame_path` | なし | FL2VAのアンカー。`reference_paths`とは併用不可。 |
| `seconds` / `chained` / `chain_clip_seconds` | `5` / `false` / `7` | 1〜15秒は1回の生成で、16〜90秒（`"chained": true`なら15秒も）はつなげて生成（5秒か7秒の動画をつなぐ）で、つないだ動画1本をつくる。つなげて生成には、開いているプロジェクトと`count` 1が必要。 |
| `clip_prompts` | なし | つなげて生成のときだけ使う。文字列の配列で、1本目から順に、その動画で`prompt`の後ろに足す指示（空の文字列や足りない分は`prompt`だけでつくる）。 |
| `clip_references` | なし | 参照画像を使うつなげて生成のときだけ。true/falseの配列で、1本目から順に、その動画に参照画像を使うか（足りない分は使う）。 |
| `clips` | なし | カスタムクリップ。1本ずつのオブジェクトの配列で、`seconds`（3〜15、省略時7）、`prompt`（`prompt`の後ろに足す指示）、2本目からは`images`（`same`／`references`／`frames`）、`reference_paths`、`first_frame_path`（その画像から始めてカットでつなぐ）、`last_frame_path`、`audio`（`carry_on`／`file`／`none`）、`audio_path`、`audio_offset`。1本目はリクエスト自体の画像とリップシンクでつくる。開いているプロジェクトが必要。 |
| `continue_from` | なし | 続きを作る元の動画（`GET /api/project`のファイル名、開いているプロジェクトの動画）。最後の22コマとその音を引き継ぐ。同じ大きさが必要で、最初・最後の画像とは併用不可（参照画像は使える）。 |
| `forced_audio_path` / `forced_audio_offset` | なし / `0` | 音声に合わせる（リップシンク）。このファイルの音声（音声ファイル、または動画の音）を`forced_audio_offset`秒目から使い、生成のあいだ音として固定して、映像だけをそれに合わせてつくる。出来上がった動画の音は、この音声そのもの。どの作り方とも併用できる。 |
| `reference_paths` | `[]` | 順序付きのRef2VA参照。画像/動画/音声はパスごとに自動判定。音声パスを含める場合、画像か動画を最低1つ含める必要あり。 |
| `loras` | `[]` | このジョブで重ねるLoRA。`GET /api/loras`の名前、または`{"name": ..., "strength": 0.8}`形式（強さは学習時の効き具合に対する倍率。省略するとそのLoRAの保存済みの強さ）。オブジェクトには`"sampler"`（`"auto"`・`"euler"`・`"renoise"`）と`"audio_shift"`（0は自動）も書け、強さと同じくそのLoRAの設定として保存される。`loras`を省略すると、ウィンドウ側でオンにしていてもこのジョブではLoRAなし扱いになる。 |
| `sampler` / `audio_shift` | LoRAの設定に従う | [ステップの進め方と音声シフト](TECHNICAL.ja.md#ステップの進め方と音声シフト)を、このジョブだけ上書きする（`"auto"`・`"euler"`・`"renoise"`、0は自動）。比較テスト用。 |
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
