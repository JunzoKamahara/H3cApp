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

別のオリジンのWebページからのPOSTリクエストには`403`が返ります: ブラウザが
付ける`Origin`ヘッダーが`http://127.0.0.1:8420`・`http://localhost:8420`
以外のもの、または`Sec-Fetch-Site: cross-site`/`same-site`を持つリクエストは
拒否されるので、閲覧中のサイトがアプリを操作することはできません。スクリプト
や`curl`はどちらのヘッダーも送らないので影響を受けません。`GET`リクエストは
従来どおり応答します。

`Host`ヘッダーが`localhost`かIPアドレス（ポートは問いません）以外の名前の
リクエストには`403`が返ります — DNSリバインディングでWebページから操作される
のを防ぐためです（自分のドメイン名を127.0.0.1に向けたページも、`Host`には
そのドメイン名を送ります）。curlやスクリプト用ライブラリはURLから`Host`を
設定するので、`http://127.0.0.1:8420`・`http://[::1]:8420`・
`http://localhost:8420`でそのまま使えます。

リクエストにはすべて、アプリのAPIトークンを`Authorization: Bearer <トークン>`
として付けてください。付いていない、または違うトークンのリクエストには、
`GET`も含めてどのエンドポイントも`401`を返します。`127.0.0.1`には自分だけで
なく、そのMacのすべてのアカウント・すべてのプロセスから接続でき、APIは
プロンプト・パス・結果を読み出し、あなたの権限でファイルを書き込めるため、
トークンで自分だけが使えるようにしています。トークンはアプリが起動する
たびに新しく作られ、自分のアカウントだけが読めるファイル（モード`0600`）
`~/Library/Application Support/H3cApp/api-token`に書き込まれます。アプリを
普通に終了するとこのファイルは削除されるので、トークンを控えておかず、
スクリプトを始めるたびにこのファイルを読んでください。
**H3cApp > 全体設定…**（⌥⌘,）には、アドレスとトークンがコピー用のボタン
（curlの例も）とともに表示され、**作り直す…**でトークンをすぐに作り直せます。トークンをコマンド
ラインやURLに書かないでください。そのMacの他のアカウントは`ps`で
すべてのプロセスのコマンドラインを読めます。curlでは下の例のように
ヘッダーを標準入力から渡します（`printf`はシェルの組み込みコマンドなので
`ps`には出ません）。他の言語のスクリプトでは、ファイルを読んで
ヘッダーをコードの中で設定してください。

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
| `POST` | `/api/project/upscale-video` | `{"video", "upscaler"?, "scale"?}` そのプロジェクトの動画を拡大した動画（2倍なら`<名前>_x2.mp4`、音声とメタデータはそのまま）をつくる。`scale`は1.5・2（省略時）・3・4。`upscaler`は`standard`（省略時。大きさの×2と同じ高品質な拡大）、`metalFX`（MetalFXの拡大）、`realESRGANGeneral`・`realESRGANAnime`（Real-ESRGANのAI超解像、アプリに同梱）、または`swiftVR`（SwiftVRの一段AI動画超解像。重みを入れて変換してあるときだけ。UPSCALE_PLAN.ja.md 参照）。拡大する前に`{"video"}`（付く名前）を返す。一辺が4096ピクセルを超えるときは`400`、ほかの拡大中は`409`。`GET /api/status`（と`GET /api/project`）の`upscaling_video`・`upscale_progress`（0〜1）・`message`で進み具合と結果が分かる。 |
| `POST` | `/api/upscale-file` | `{"path", "output", "upscaler"?, "scale"?}` どの動画ファイルでも、拡大して`output`（`.mp4`のパス）に書き出す。拡大の方法と倍率は上と同じ。`output`の隣の隠しファイルに書き出し、終わってから置き換えるので、拡大に失敗したり中止したりしても、既にある`output`はそのまま残る。`output`が入力そのもの（`..`やシンボリックリンク、ハードリンクでの別名も含む）なら`400`。別のサイトのWebページから送られたもの（別の`Origin`、または`Sec-Fetch-Site: cross-site`/`same-site`）には`403`を返す。スクリプトや`curl`は影響を受けない。すぐに返り、進み具合と結果は上と同じく分かる。 |
| `POST` | `/api/upscale/cancel` | 実行中の拡大を止める。書きかけのファイルは消す。実行中でなければ`409`。 |
| `POST` | `/api/project/import-video` | `{"path", "size_profile"?, "carry_frames"?, "crop_position"?}` ほかで作った動画の続きをつくるため、その動画を、動画の大きさに近い大きさ（`size_profile`が`"current"`なら今の大きさ、大きさの名前ならその大きさ）と24fpsに変換してプロジェクトに取り込み、フォームをその続きにする。`carry_frames`は引き継ぐコマ数（22・39・56、省略時22）。`crop_position`（`[x, y]`、それぞれ0〜1、省略時`[0.5, 0.5]`）は、縦横比が違うときに残す部分の位置（`[0, 0.5]`で左端）。取り込む前に`{"video"}`（付く名前）を返し、変換が終わると`GET /api/project`に並んで`continue_from`に使える。 |

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
| `continue_from` | なし | 続きを作る元の動画（`GET /api/project`のファイル名、開いているプロジェクトの動画）。最後の22コマとその音を引き継ぐ。同じ大きさが必要で、最初・最後の画像とは併用不可（参照画像は使える）。拡大した動画は、元の動画の大きさで続きをつくり、できた動画を同じ方法・倍率で拡大して、その続きになる拡大した動画もつくる（そのあいだは`GET /api/status`の`upscaling_video`が真）。つないだ長さとは併用不可。 |
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
TOKEN_FILE="$HOME/Library/Application Support/H3cApp/api-token"

printf 'Authorization: Bearer %s\n' "$(cat "$TOKEN_FILE")" |
  curl -X POST http://127.0.0.1:8420/api/generate -H @- -H "Content-Type: application/json" -d '{
  "prompt": "A red fox walks through fresh snow in a pine forest.",
  "size_profile": "square", "seconds": 5, "steps": 20
}'

printf 'Authorization: Bearer %s\n' "$(cat "$TOKEN_FILE")" | curl -H @- http://127.0.0.1:8420/api/status
printf 'Authorization: Bearer %s\n' "$(cat "$TOKEN_FILE")" | curl -H @- http://127.0.0.1:8420/api/result/video -o fox.mp4
```

標準入力はヘッダーに使うので、ファイルに入れたボディは`-d @ファイル名`で
送ります。

```sh
TOKEN_FILE="$HOME/Library/Application Support/H3cApp/api-token"
printf 'Authorization: Bearer %s\n' "$(cat "$TOKEN_FILE")" |
  curl -X POST http://127.0.0.1:8420/api/generate -H @- -H "Content-Type: application/json" -d @request.json
```

### ほかの機器から使う

**全体設定 > 同じネットワークのほかの機器からAPIを使えるようにする**を
オンにしない限り、APIは`127.0.0.1`でしか受け付けません。オンにすると、
Macのネットワークのアドレス（`http://192.168.1.5:8420`や
`http://<名前>.local:8420`など、全体設定に表示されます）でも受け付けるので、
同じネットワークのほかのMacやスマートフォンから、同じトークンで使えます。
オンのあいだ、トークンはアプリを終了しても変わりません（ほかの機器は
トークンのファイルを読めないため）。オフに戻すとトークンは作り直されるので、
ほかの機器に渡したトークンは使えなくなります。macOSが一度、H3cAppが外部から
の接続を受け付けてよいか尋ねることがあります。

通信は暗号化されない素のHTTPです。同じネットワークの機器なら、行き来する
トークン・プロンプト・動画を読み取れます。自宅など信頼できるネットワーク
でだけ使ってください。それ以外の場所では、SSHのトンネル
（`ssh -L 8420:127.0.0.1:8420 <ユーザー名>@<Mac名>.local`、Mac側で
「リモートログイン」をオン）を使えば、APIは`127.0.0.1`のままで使えます。
リクエストの中のパスは、H3cAppが動いているMacの上のパスです。
1つの接続がリクエストを送り終えるまでの時間は最大30秒、本文は最大1MBで、
同時に扱う接続は32本までです。
