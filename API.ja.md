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
| `GET` | `/api/refmods` | RefModの一覧（モデル管理のRefMod）。各ファイルの`id`・`name`・`path`と、ヘッダから`format_version`・`bundle`・`file_name`・`members`（`index`、`kind`はimage/video/audio、`mode`はfull/compressed、`latent_t`/`latent_h`/`latent_w`、`tokens`、`name`、`source`、`pool`、`optimize_steps`、`supported`）。ファイルを読み込み中は`null`、読めないときは`error`も。 |
| `POST` | `/api/refmods/extract` | `{"path", "name"?, "mode"?, "resolution_cap"?, "latent_frames"?, "max_seconds"?}` 画像・動画・音声のファイルからFullのRefModをRefModフォルダに作る。`mode`は`full`だけ（`compressed`は`400`。H3cAppは圧縮のRefModを作らなくなった。読み込みはできる）、`resolution_cap`は短辺の上限ピクセル（省略時1024）、`latent_frames`は動画から使うコマ数（省略時16）、`max_seconds`は使う音声の秒数（省略時15）。`"path"`の代わりに`"paths": [...]`で2〜32枚の静止画を1つのRefModにまとめる（上流の"stack"。1枚が潜在の1コマになり順に並ぶ。`resolution_cap`の省略時は512。1枚目が画面の大きさを決め、ほかはそれを覆うよう拡大縮小して中央を切り取る）。書く前に`202`で`{"output"}`（できるファイル）を返す。ほかを作成中は`409`。生成が走っていればエンジンの待ち行列でその後に始まり、`GET /api/status`の`refmod_extracting`・`refmod_extract_phase`・`refmod_extract_progress`（その段階の0〜1）・`refmod_extract_name`で進み具合、`message`で結果が分かる。 |
| `POST` | `/api/refmods/extract/cancel` | 作成中のRefModを止める（なければ`409`）。 |
| `GET` | `/api/loras` | 登録済みLoRAファイル一覧（id・名前・パス・強さ・推奨ステップ数・オンかどうか、モデル管理で設定したステップの進め方`sampler`と`audio_shift`、ファイルの説明による推奨`recommended_sampling`。ファイルを読み込み中は`null`）。 |
| `GET` | `/api/upscalers` | 拡大の方法の一覧：`id`・`label`・`available`・`min_scale`・`max_scale`（`scale`に指定できる範囲。動画によってはさらに小さくなる）と、選べないときの`reason`（このMacでは使えない、モデルが入っていない）、生成・キャッシュ作成・取り込みの間だけ待たせるときの`busy_reason`。`swiftVR`にはさらに`model_directory`・`model_installed`（元の重みがあるか）・`original_removed`（変換したあとで元の重みを削除したか）・`cache`（動かすときに読む変換済み`.h3vc`、無ければnull）・`license_accepted`・`setup`（`idle`／`listing`／`downloading`／`verifying`／`converting n/30`／`failed: …`／`done`）と、固定した取得元：`revision`（ダウンロードを固定しているH-oliday/SwiftVRのrevision。モデルカードが重みをApache License 2.0と宣言していた最後のもの）・`declared_license`・`verified`（ここにあるファイルをそのrevisionとSHA-256で照合済みか）・`verified_at`。ダウンロード（約20GB）と変換はアプリ側（動画を拡大 → モデルを準備…、またはモデル管理 → 拡大モデル）か`POST /api/upscalers/swiftVR/setup`で始める。 |
| `POST` | `/api/upscalers/swiftVR/setup` | `{"confirm_license": true}`で、準備の画面のボタンと同じ流れを始める：固定したrevisionの一覧を取り、足りないファイルをダウンロードし、各ファイルのSHA-256をそのrevisionと照合し（違うファイルは削除して`setup`に名前を出す）、ライセンス本文と記録を重みの隣に保存し、このMac用に変換する。`{"ok", "setup"}`を返す。確認なしは400（アプリが示すライセンスの説明を読んだことの代わり）、すでに動いているとき、または最初の変換の前で生成・キャッシュ作成・取り込みがGPUを使っているときは409。進み具合は`GET /api/upscalers`の`setup`で追える。 |
| `POST` | `/api/upscalers/swiftVR/remove-original` | 変換したあとで、ダウンロードした元の重み（約20GB）を削除します。変換したキャッシュは残り、SwiftVRはそのまま使えます。`{"ok", "freed_bytes", "cache"}`を返します。元の重みが無いとき、このMac用にまだ変換していないとき、準備の最中は409。変換し直すには、もう一度ダウンロードが必要です。 |
| `POST` | `/api/upscalers/seedVR2/setup` | `{"confirm_license": true}`で、mlx-community/SeedVR2-3B-mlx-int8の固定したrevisionの一覧を取り、足りないファイル（約4.7GB、中断しても続きから）をダウンロードし、各ファイルのSHA-256をそのrevisionと照合し（違うファイルは削除して`setup`に名前を出す）、Apache License 2.0の本文と記録を重みの隣に保存します。変換はありません（エンジンがそのファイルを読みます）。`{"ok", "setup"}`を返し、確認なしは400、すでに動いているときは409。`seedVR2`と`seedVR2Video`については`GET /api/upscalers`が`model_directory`・`model_installed`・`license_accepted`・`setup`・`revision`・`declared_license`・`verified`・`verified_at`を返し、`seedVR2Video`はさらに動画の切り方`window_frames`（17）・`overlap_frames`（5）・`blend`（`hann`）を返します。 |
| `POST` | `/api/upscalers/openH3/setup` | `{"confirm_license": true}`で、Efficient-Large-Model/Open-H3-Regenerate-2Kの固定したrevisionの一覧を取り、足りないファイル（結合LoRA FP32 約12.5GB、潜在アップスケーラ 691MB、ライセンス等。中断しても続きから。LoRAフォルダにLightX2V Turbo v1.2の同じ版が無ければそれも1.4GB）をダウンロードし、各ファイルのSHA-256をそのrevisionと照合し（違うファイルは削除して`setup`に名前を出す）、結合LoRAから精緻化の部分（ランク0〜383）だけをBF16で取り出して（`extracting`、約4.2GB）結合ファイルを削除し、記録を保存します。`{"ok", "setup"}`を返し、確認なしは400、すでに動いているときは409。`openH3`については`GET /api/upscalers`が`model_directory`・`model_installed`・`license_accepted`・`setup`（`extracting`を含む）・`revision`・`declared_license`（MiniMax H3 Community License）・`verified`・`verified_at`・`turbo`（使うTurboのパス）・`max_pixels`を返します。 `GET /api/upscalers`の`openH3`には、選べる強さの`strengths`（`id`・`label`・`sigmas`）と`default_strength`も入る。 |
| `GET` | `/api/project` | 開いているプロジェクト（名前・パス・`untitled`（ウィンドウの無題のプロジェクト。保存するまで Application Support/H3cApp/Untitled に置かれる）・本数・動画一覧（ファイル・シード・完成日時・生成時間・`continued_from`・`can_continue`、つないだ動画なら`joined_clips`・`planned_clips`））。 |
| `POST` | `/api/project/new` | `{"name", "directory"?}` 今のフォームからプロジェクトを作って開く（既定の場所は`~/Movies/H3cApp`）。 |
| `POST` | `/api/project/open` | `{"path"}` プロジェクトのフォルダを開く。フォームは保存時の状態になる。 |
| `POST` | `/api/project/close` | 閉じる。ウィンドウは新しい無題のプロジェクトで続く（`{"ok": true, "untitled": true}`）。動画のない無題のプロジェクトは削除される。 |
| `POST` | `/api/project/save` | `{"name", "directory"?}` 開いている無題のプロジェクトに名前を付ける。フォルダが動画・記録・潜在ファイルごと `directory/name`（既定の directory は `~/Movies/H3cApp`）に移り、プロジェクトがその名前になる。移動を待ってプロジェクトを返す。名前のあるプロジェクトや既にあるフォルダには `400`、生成・拡大・取り込み・書き出し・ほかの移動の間は `409`。 |
| `POST` | `/api/project/move-video` | `{"video", "project"}` その動画をプロジェクトのフォルダ `project` に移す。記録と潜在ファイル、つなげて生成した動画ならもとの動画のフォルダ（`chains/<名前>`）も一緒に移り、記録が使う入力ファイルは移動先の `references` にコピーされる。移動先で使われている名前には `-2`、`-3`… が付く。移動を待って `{"ok", "project", "video"}`（移動先での名前）を返す。理由付きの `400`（プロジェクトでない、同じプロジェクト）、ファイルの書き込み中は `409`。 |
| `POST` | `/api/project/restore` | `{"video"}` その動画のリクエストをシード固定でフォームに戻す。 |
| `POST` | `/api/project/use-as-reference` | `{"video"}` その動画を参照動画としてフォームに追加する。 |
| `POST` | `/api/project/delete-video` | `{"video"}` 動画と記録をゴミ箱に移す。 |
| `POST` | `/api/project/join` | `{"video"}` その動画で終わる続きの連なりを、引き継いだ部分のクロスフェードでプロジェクトの`<名前>_joined.mp4`（記録つき）につなぎ、終わったら`{"path", "clips"}`を返す。 |
| `POST` | `/api/project/upscale-video` | `{"video", "upscaler"?, "scale"?, "strength"?}` そのプロジェクトの動画を拡大した動画（2倍なら`<名前>_x2.mp4`、音声とメタデータはそのまま）をつくる。`scale`は小数第2位までの数値で省略時は2。下限はAIの方法が1（等倍。大きさを変えずに描き直す）、`standard`と`metalFX`が1.1で、上限はその2つが8、ほかの方法が4で、動画によっては一辺4096ピクセル・GPUメモリ・Open-H3の上限1344×768でさらに小さくなり、超えると使える範囲つきの`400`。`upscaler`は`standard`（省略時。大きさの×2と同じ高品質な拡大）、`metalFX`（MetalFXの拡大）、`realESRGANGeneral`・`realESRGANAnime`（Real-ESRGANのAI超解像、アプリに同梱）、`swiftVR`（SwiftVRの一段AI動画超解像。重みを入れて変換してあるときだけ。UPSCALE_PLAN.ja.md 参照）、または`swiftVRFourX`（同じモデルに4倍で復元させ、大きいフレームは重なりを持たせたタイルで処理して、選んだ倍率に縮める。質感が穏やかで元に忠実、時間は約5倍）、`seedVR2`（SeedVR2-3Bで1フレームずつ。最も細部を描き、最も遅い。重みが入っているときだけ。試験的）、`seedVR2Video`（同じモデルに17フレームの窓で動画として描き起こさせ、窓どうしは5フレームを重ねてクロスフェード。ちらつきが減り、時間は約2.3倍、24GBのMacでは1344×768まで。試験的）、`openH3`（Open-H3 再生成：その動画の潜在を広げてH3自身が2〜3ステップで描き直す。描き直しの強さは`strength`で`light`（省略時、σ 0.4）、`medium`（σ 0.7）、`strong`（σ 0.909、配布元の手順）から選び、ほかの方法に指定すると`400`。大きさはH3の上限1344×768までに丸められ、潜在ファイルのある生成した動画だけ。`GET /api/upscalers`の`model_installed`が真のときだけ。できない動画や倍率は理由つきの`400`。試験的）。拡大する前に`{"video"}`（付く名前）を返す。一辺が4096ピクセルを超えるときは`400`、ほかの拡大中は`409`。`GET /api/status`（と`GET /api/project`）の`upscaling_video`・`upscale_progress`（0〜1）・`message`で進み具合と結果が分かる。`swiftVR`・`seedVR2`・`seedVR2Video`は、生成・キャッシュ作成・取り込みの間は`409`（GPUを独占するため）。 |
| `POST` | `/api/upscale-file` | `{"path", "output", "upscaler"?, "scale"?}` どの動画ファイルでも、拡大して`output`（`.mp4`のパス）に書き出す。拡大の方法と倍率は上と同じ。`output`の隣の隠しファイルに書き出し、終わってから置き換えるので、拡大に失敗したり中止したりしても、既にある`output`はそのまま残る。`output`が入力そのもの（`..`やシンボリックリンク、ハードリンクでの別名も含む）なら`400`。別のサイトのWebページから送られたもの（別の`Origin`、または`Sec-Fetch-Site: cross-site`/`same-site`）には`403`を返す。スクリプトや`curl`は影響を受けない。`409`は上と同じ。すぐに返り、進み具合と結果は上と同じく分かる。 |
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
| `reference_paths` | `[]` | 順序付きのRef2VA参照。画像/動画/音声はパスごとに自動判定し、`.safetensors`のパスはRefMod（保存した参照の潜在。`GET /api/refmods`参照）。RefModは`{"path", "strength"?, "member"?}`のオブジェクトでも指定でき、`strength`は0より大きく1以下（省略時1）、`member`はバンドルのメンバー番号（0始まり。バンドルでは必須、単体ファイルでは省略時の`-1`）。音声パスを含める場合、画像か動画かRefModを最低1つ含める必要あり。 |
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
