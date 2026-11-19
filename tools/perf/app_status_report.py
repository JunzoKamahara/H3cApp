"""Writes the report of an app_status.py run: one table of what worked, how
long it took and how much memory it used, one of the memory by stage, the
failures in full, and a contact sheet of a frame of every video.

    python3 tools/perf/app_status_report.py <work dir> <report.md> [<contact sheet.jpg>]

<work dir>/notes.md, when there, is put at the top as the reading of the
numbers (written by hand; the tables are only the measurements).

The work dir may have been made on another Mac and brought here: the Mac and
the app's version are the ones in the records, and the videos are looked for
under this work dir. The contact sheet needs ffmpeg; without it the report
is written without one."""
import json, os, re, shutil, subprocess, sys

work, report = sys.argv[1:3]
sheet = sys.argv[3] if len(sys.argv) > 3 else None
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
records = {}
for line in open(os.path.join(work, "results.jsonl")):
    record = json.loads(line)
    records[record["name"]] = record
records = list(records.values())


def shell(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def result_text(record):
    if record.get("skipped"):
        return "スキップ"
    if record.get("refused"):
        return "断られた"
    if record["ok"]:
        return "中止できた" if record["name"] == "cancel" else "成功"
    return "**失敗**"


def here(path):
    """A video's path in this work dir, when the records came from another."""
    if os.path.exists(path) or "/home/Library/" not in path:
        return path
    return os.path.join(work, "home", "Library", path.split("/home/Library/", 1)[1])


def distinct(values):
    return "／".join(dict.fromkeys(v for v in values if v)) or "?"


def peak(record):
    """The largest footprint of the job and the stage it was in: the engine's
    own figure for each stage, or the sampled one when that is higher."""
    stages = record.get("stage_peaks_gib") or {}
    name, value = max(stages.items(), key=lambda item: item[1]) if stages else ("-", 0.0)
    sampled = record.get("footprint_peak_gib") or 0.0
    return (sampled, "（採取）") if sampled > value + 0.05 else (value, name)


def seconds(record, *names):
    return sum(t for phase, t in record.get("phases", []) if any(phase.startswith(n) for n in names))


def time_text(value):
    value = float(value or 0)
    return "%d分%02d秒" % (value // 60, value % 60) if value >= 60 else "%.0f秒" % value


STAGE_NAMES = {"text encoder": "テキストエンコーダ", "denoise enqueue": "デノイズ", "denoise": "デノイズ",
               "video VAE load": "動画VAE", "encode": "書き出し", "load transformer core": "DiTの読み込み",
               "refine text": "テキストの精製", "precompute AdaLN": "AdaLNの準備", "audio VAE": "音声VAE",
               "video encoder": "動画VAEエンコーダ", "vision encoder": "画像エンコーダ", "DiT": "DiT",
               "video decoder": "動画デコーダ", "（採取）": "採取値"}


def stage_text(name):
    return STAGE_NAMES.get(name, name)


def output_text(record):
    videos = record.get("videos") or []
    if not videos:
        return "—"
    first = videos[0]
    text = "%s×%s、%sコマ（%.1f秒）%s" % (first.get("width"), first.get("height"), first.get("frames"),
                                      first.get("seconds") or 0, "、音声あり" if first.get("audio") else "、**音声なし**")
    return text + ("、ほか%d本" % (len(videos) - 1) if len(videos) > 1 else "")


lines = []
ok = [r for r in records if r["ok"]]
skipped = [r for r in records if r.get("skipped")]
refused = [r for r in records if r.get("refused")]
failed = [r for r in records if not r["ok"] and not r.get("skipped") and not r.get("refused")]
# Records from before the Mac and the app were written in them are this
# Mac's, and the app is this checkout as it was when the run began (not as it
# is now: a report is made again after later commits).
runs = [r["run"] for r in records if r.get("run")]
apps = distinct(run.get("app") for run in runs) if runs else "開発ビルド、コミット %s" % (
    shell("git", "-C", repo, "log", "-1", "--format=%h", "--before=" + min(r["at"] for r in records)) or "?")
macs = distinct("%s（%s）、メモリ %d GB、macOS %s" % (
    run["machine"]["chip"], run["machine"]["model"], run["machine"]["memory_gb"], run["machine"]["macos"])
    for run in runs) if runs else "%s、メモリ %d GB、macOS %s" % (
    shell("sysctl", "-n", "machdep.cpu.brand_string"), int(shell("sysctl", "-n", "hw.memsize")) // 2 ** 30,
    shell("sw_vers", "-productVersion"))
models = distinct(r.get("model") for r in records)
lines += ["# H3cApp：アプリでの動画生成の状況", "",
          "`tools/perf/app_status.py`が、アプリをローカルAPIから動かして取った記録である。"
          "表は測った値だけで、読み取りは「所見」にまとめた。", "",
          "| 項目 | 値 |", "|---|---|",
          "| 実行日時 | %s 〜 %s |" % (min(r["at"] for r in records), max(r["at"] for r in records)),
          "| アプリ | %s |" % apps,
          "| Mac | %s |" % macs]
if models != "?":
    lines += ["| モデル | %s |" % models]
lines += ["| 電源 | %s |" % "、".join(sorted({r.get("power", "?") for r in records})),
          "| 結果 | %d件のうち、成功 %d件、失敗 %d件%s%s |" % (
              len(records), len(ok), len(failed),
              "、メモリ計画が断った %d件" % len(refused) if refused else "",
              "、スキップ %d件" % len(skipped) if skipped else ""), ""]
notes = os.path.join(work, "notes.md")
if os.path.exists(notes):
    lines += [open(notes, encoding="utf-8").read().rstrip(), ""]

lines += ["## 設定ごとの結果", "",
          "特に書いていない項目は、512×512・5秒・20ステップ・reuse 2・品質優先（フォームとAPIの既定）である。"
          "計算方式は、書いていなければそのMacの既定（int8キャッシュを使えるMacではint8キャッシュ）で、"
          "実際に使われた方式は「実際に使われた設定」の表にある。プロンプトとseed（7）は全部同じ。", "",
          "- **生成時間**：アプリが動画に記録した値（生成の開始から書き出しまで）。複数本のときは全体の所要時間。",
          "- **ピーク**：生成中のアプリのメモリ使用量（phys_footprint）の最大と、その段階。",
          "- **計画**：生成の前にエンジンが見積もったピーク。", "",
          "| 設定 | 結果 | 生成時間 | うちテキスト | うちデノイズ | うちVAE・書き出し | ピーク | 計画 | 出力 |",
          "|---|---|---|---|---|---|---|---|---|"]
for r in records:
    videos = r.get("videos") or []
    generation = videos[0].get("generation_seconds") if len(videos) == 1 else None
    value, where = peak(r)
    plan = r.get("plan")
    result = result_text(r)
    lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        r["label"], result,
        time_text(generation or r.get("wall_seconds")) if r.get("wall_seconds") else "—",
        time_text(seconds(r, "tokenizer", "text encoder", "refine", "video encoder", "vision", "Qwen")) if r.get("phases") else "—",
        time_text(seconds(r, "precompute", "load transformer", "denoise")) if r.get("phases") else "—",
        time_text(seconds(r, "audio VAE", "video VAE", "encode", "upscal")) if r.get("phases") else "—",
        "%.2f GiB（%s）" % (value, stage_text(where)) if value else "—",
        "%.1f GiB（%s）" % (plan["gib"], stage_text(plan["stage"])) if plan else "—",
        output_text(r)))

columns = ["text encoder", "denoise enqueue", "video VAE load", "encode"]
lines += ["", "## 段階ごとのメモリ", "",
          "エンジンが段階ごとに記録した最大（GiB）。複数本つくる設定では、各段階の最大。"
          "「生成の前」「生成の後」は、生成していないときのアプリの使用量（生成の5秒後）。", "",
          "| 設定 | テキストエンコーダ | デノイズ | 動画VAE | 書き出し | 生成の前 | 生成の後 | スワップ（前→後） |",
          "|---|---|---|---|---|---|---|---|"]
for r in records:
    stages = r.get("stage_peaks_gib") or {}
    denoise = max(stages.get("denoise enqueue", 0), stages.get("denoise", 0))
    cells = [stages.get("text encoder"), denoise or None, stages.get("video VAE load"), stages.get("encode")]
    lines.append("| %s | %s | %s | %s | %s |" % (
        r["label"], " | ".join("%.2f" % c if c else "—" for c in cells),
        "%.2f" % r["idle_before_gib"] if r.get("idle_before_gib") is not None else "—",
        "%.2f" % r["idle_after_gib"] if r.get("idle_after_gib") is not None else "—",
        "%.1f→%.1f" % (r.get("swap_before_gib") or 0, r.get("swap_after_gib") or 0)))

lines += ["", "## 実際に使われた設定", "",
          "アプリが動画に記録した値。", "",
          "| 設定 | 計算方式 | 速度モード | reuse | ステップ | コマ数 | CCVの呼び出し | latentのSHA-256（先頭16桁） |",
          "|---|---|---|---|---|---|---|---|"]
for r in records:
    for video in (r.get("videos") or [])[:1]:
        made = video.get("as_made") or {}
        lines.append("| %s | %s | %s | %s | %s | %s | %s | `%s` |" % (
            r["label"], made.get("compute_mode"), made.get("speed_mode"), made.get("reuse"), made.get("steps"),
            made.get("frames"), made.get("ccv_calls"), video.get("latent_sha256", "—")))

if failed or skipped or refused or any(r.get("log_notes") or r.get("fallback") for r in records):
    lines += ["", "## 失敗・スキップと、ログに出た注意", ""]
    for r in records:
        problems = [r.get("error"), r.get("error_message")] if not r["ok"] else []
        problems = [p for p in problems if p]
        if r.get("skipped"):
            problems = ["スキップ：" + r["skipped"]]
        if r.get("fallback"):
            problems.append("int8キャッシュが無いので、%sで生成した" % r["fallback"])
        if not problems and not r.get("log_notes"):
            continue
        lines.append("- **%s**（`%s`）%s" % (r["label"], r["name"], "：" + " / ".join(problems) if problems else ""))
        for note in r.get("log_notes") or []:
            lines.append("  - ログ：`%s`" % note.replace("`", "'"))

if sheet:
    thumbs = os.path.join(work, "thumbs")
    shutil.rmtree(thumbs, ignore_errors=True)
    os.makedirs(thumbs, exist_ok=True)
    shown = []
    for r in records:
        for video in (r.get("videos") or [])[:1]:
            target = os.path.join(thumbs, "%02d.png" % len(shown))
            middle = max(0, (video.get("frames") or 2) // 2)
            shell("ffmpeg", "-v", "error", "-y", "-i", here(video["path"]), "-vf",
                  "select=eq(n\\,%d),scale=192:192:force_original_aspect_ratio=decrease,"
                  "pad=192:192:(ow-iw)/2:(oh-ih)/2" % middle, "-vframes", "1", target)
            if os.path.exists(target):
                shown.append(r["label"])
    if shown:
        columns_count = 6
        rows = (len(shown) + columns_count - 1) // columns_count
        shell("ffmpeg", "-v", "error", "-y", "-i", os.path.join(thumbs, "%02d.png"), "-vf",
              "tile=%dx%d" % (columns_count, rows), "-frames:v", "1", "-q:v", "3", sheet)
        lines += ["", "## 映像の確認", "",
                  "各動画のまんなかのコマを、左上から右へ、次の順に並べた一覧（`%s`）。"
                  % sheet.replace(os.path.expanduser("~"), "~"), ""]
        lines += ["%d. %s" % (index + 1, label) for index, label in enumerate(shown)]
    else:
        lines += ["", "## 映像の確認", "", "コマの一覧は作っていない（ffmpegが無いか、動画がこの作業フォルダに無い）。"]

lines += ["", "## 記録の場所", "",
          "`%s`：`results.jsonl`（1行が1件の記録）、`app.log`（アプリとエンジンの出力）、"
          "`home/`（切り離したホーム。プロジェクト、動画、latent）。" % work.replace(os.path.expanduser("~"), "~")]
open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("wrote", report, "-", len(ok), "ok,", len(failed), "failed,", len(refused), "refused,", len(skipped), "skipped")
