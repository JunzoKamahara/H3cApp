# H3cApp 開発者向けビルド・テストガイド

[English](BUILD.md) | [日本語](BUILD.ja.md)

ソースからのビルド、テスト、署名、配布を行う開発者向けです。通常の利用では[署名済みの配布版](https://github.com/JunzoKamahara/H3cApp/releases/latest)を使ってください。ソースからのビルドにはXcode Command Line Toolsが必要です。

## クイックスタート

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/H3cApp.app
```

初回起動時、まだモデルが登録されていなければ、アプリがHugging Faceから
MiniMax-H3を直接ダウンロードするか尋ねます（[機能](TECHNICAL.ja.md#機能)を参照）。すでに
チェックポイントを持っている場合は「モデル管理」からその場所を指定して
ください。

対応GPUでは、計算方式選択からint8アテンションキャッシュを作成（高速経路）
するか、「省メモリ（SSDストリーミング）」を選んでこの作成手順自体を省略し、
元のBF16重みをストリーミングすることもできます。

## リポジトリ構成

```
h3.c, h3_dit.c, h3_gpu.m, ...   コア推論エンジン（C + Objective-C/Metal）。libh3.aをビルド
h3.h                            公開C API（h3_load_dir、h3_generate、h3_build_attention_cache、...）
h3_shaders.metal                すべてのMetalコンピュートカーネル
h3_build_attention_cache.c      h3_build_attention_cache()のCLIラッパー -> build_attention_cache
h3_adaln_curve.c                曲線形AdaLNファイル（読み込みと作成）。h3_build_adaln_curve.c -> build_adaln_curve
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

ソースからビルドするには、Xcode Command Line Tools（`clang`、`swift`、`ar`）が
必要です（ビルド済みの`.dmg`では不要）。

### ライブラリ

```sh
make -j8 libh3.a     # C/Objective-Cエンジンをビルド
make test             # 決定論的なホストテスト一式（+ フィクスチャがあればMetal/MLXパリティ）
make parity            # Metal/MLXの数値比較チェックのみ
```

`make build_attention_cache`は上記の独立したキャッシュ準備ツールを、
`make build_adaln_curve`は曲線形AdaLNのツールをビルドします。
どちらもデフォルトの`make`ターゲットには含まれません。

### ネイティブアプリ

```sh
cd native/H3Spike
./package_app.sh
```

`libh3.a`はこのスクリプトでは再ビルドされません — リポジトリ直下で
`make libh3.a`を先に（エンジンのコードを変更した後も同様に）実行して
ください。`package_app.sh`はその後`swift build -c release`を実行し、
`H3cApp.app`を組み立てます。SwiftPM標準のリポジトリ内`.build`ではなく
リポジトリ外のスクラッチディレクトリ（`${TMPDIR}H3cApp-build-scratch`）
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
`make parity`はそのMetal/MLXチェックのみを実行します。SwiftVRの照合テスト
（`h3_real_vsr_test`）も同様に、ダウンロード済みのSwiftVRの重みと
`tools/dump_swiftvr_fixture.py`のfixtureがあるときだけ走り、なければ
`skip:`と表示します。

同じシードでの再現性：`make test`には`h3_determinism_tests`（threadgroupで
集計するカーネルを繰り返し実行し、ビット単位で一致することを確認）が
含まれます。公開済みの重みがあれば、`make h3_repro_check CCV_DIR=...`の後に
`./h3_repro_check`を実行すると、短いRef2VA動画を3回生成し（約4分）、
エンコーダーに渡すRGBフレームとデコード後の音声が一致しなければ失敗します。
`--run`で実行ごとに条件を変えられ、`cache`以外が同じ実行どうしを比較します
（例：`--run cache=0 --run cache=conditioning --run cache=conditioning`で、
キャッシュなし・初回・再利用を比べます）。
